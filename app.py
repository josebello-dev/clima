"""
================================================================================
 CARIBE WEATHER ANALYTICS v2.0  |  Barranquilla / Caribe Colombiano
================================================================================
 Sistema de pronóstico de lluvias con diagnóstico sinóptico:
   - Baja presión (anomalía + tendencia barométrica real)
   - Vaguadas / ejes de baja en altura (500 y 700 hPa)
   - Ondas tropicales del Este (firma clásica de paso de onda)
   - Capa de Aire Sahariano / intrusión seca (supresor de convección)
   - Termodinámica convectiva (CAPE, CIN, Lifted Index, agua precipitable)
   - Ciclo diurno y brisa marina

 Ejecutar:  streamlit run app_clima_caribe.py
 Requiere:  streamlit pandas numpy plotly requests
================================================================================
"""

import io
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components


# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================

LAT, LON = 10.968, -74.781          # Barranquilla
TZ = "America/Bogota"
ZONA = ZoneInfo(TZ)
API_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo entrega vertical_velocity en m/s con positivo = ascenso.
# Si notas que el diagnóstico de ascenso sale invertido, cambia a -1.
SIGNO_ASCENSO = 1

# --- Pesos del algoritmo (AJÚSTALOS con tu bitácora de verificación) ---------
# Se mantienen en "puntos" para que sean fáciles de leer, pero YA NO se suman
# directo al porcentaje: se convierten a log-odds (ver combinar_probabilidad).
# Así 10 factores a favor acercan la probabilidad a 90%, sin pasar de 100%.
PESOS = {
    # Sinópticos / dinámicos (independientes entre sí: se suman)
    "baja_presion_absoluta": 18,
    "anomalia_presion": 14,
    "caida_barometrica": 12,
    "vaguada_500": 16,
    "vaguada_700": 12,
    "onda_tropical": 22,
    "ascenso_700": 10,
    # Termodinámicos (muy correlacionados: cuenta el mayor + 30% del resto)
    "conveccion_cape": 16,
    "lifted_index": 12,
    "humedad_profunda": 12,
    "agua_precipitable": 10,
    "gatillo_termico": 10,
    "ciclo_diurno": 8,
    # Trayectoria de tormentas
    "tormenta_en_camino": 30,     # lluvia aguas arriba según el flujo director
    "flujo_aleja": -14,           # flujo fuerte y nada aguas arriba
    # Análisis oficial del NHC (ondas y bajas analizadas por meteorólogos)
    "onda_nhc": 20,
    "baja_nhc": 8,
    # --- Supresores (restan) ---
    "aire_sahariano": -25,
    "cin_fuerte": -15,
    "subsidencia": -12,
    "alta_presion": -12,
}

PUNTOS_POR_LOGIT = 25.0      # 25 puntos = 1 unidad de log-odds
PROB_MIN, PROB_MAX = 3.0, 92.0   # nunca 0% ni 100%: la atmósfera no da certezas
VEL_TORMENTA_ORGANIZADA = 30.0   # km/h: a esta velocidad manda lo que viene de lejos

# --- Umbrales climatológicos para el Caribe colombiano -----------------------
UMBRALES = {
    "pres_baja_abs": 1009.0,      # hPa a nivel del mar: baja relevante en trópico
    "pres_alta_abs": 1014.5,      # dorsal / alta bien marcada
    "anom_pres_baja": -2.0,       # hPa bajo la media móvil de 5 días
    "caida_3h": -1.2,             # hPa en 3 h
    "gph500_anom": -12.0,         # metros geopotenciales bajo la media
    "gph700_anom": -8.0,
    "rh700_humedo": 60.0,         # % humedad en capa media = combustible
    "rh700_seco": 35.0,           # % por debajo -> intrusión seca / SAL
    "pw_alto": 50.0,              # kg/m² agua precipitable (trópico húmedo)
    "pw_bajo": 38.0,
    "cape_moderado": 800.0,
    "cape_alto": 1800.0,
    "li_inestable": -2.0,
    "cin_bloqueante": -75.0,      # J/kg: tapa que impide el disparo
    "temp_gatillo": 31.0,
    "hum_gatillo": 72.0,
}

st.set_page_config(
    page_title="Caribe Weather Analytics v2 | Barranquilla",
    page_icon="⛈️",
    layout="wide",
)



# ==============================================================================
# TRAYECTORIA DE TORMENTAS (antes en trayectoria_tormentas.py)
# Flujo director 850-500 hPa + anillo de vigilancia de 100 km + advección.
# ==============================================================================

LAT_BQ, LON_BQ = 10.968, -74.781
API_FORECAST = "https://api.open-meteo.com/v1/forecast"

RADIOS_KM = (30.0, 60.0, 100.0)
N_RUMBOS = 16
KM_POR_GRADO = 111.32

# Peso de cada nivel en el flujo director
PESOS_NIVEL = {"850": 0.25, "700": 0.40, "500": 0.35}

# Pérdida de confianza por cada hora extra de viaje (las celdas también mueren)
DECAIMIENTO = {1: 1.00, 2: 0.80, 3: 0.60}

PUNTOS_CARDINALES = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]


def rumbo_a_texto(grados: float) -> str:
    if pd.isna(grados):
        return "s/d"
    return PUNTOS_CARDINALES[int(((grados % 360) + 11.25) // 22.5) % 16]


# ==============================================================================
# 1. GEOMETRÍA
# ==============================================================================

def _desplazar(lat: float, lon: float, este_km: float, norte_km: float):
    dlat = norte_km / KM_POR_GRADO
    dlon = este_km / (KM_POR_GRADO * np.cos(np.deg2rad(lat)))
    return lat + dlat, lon + dlon


def puntos_anillo() -> pd.DataFrame:
    filas = []
    for r in RADIOS_KM:
        for k in range(N_RUMBOS):
            rumbo = k * 360.0 / N_RUMBOS               # 0 = norte, 90 = este
            este = r * np.sin(np.deg2rad(rumbo))
            norte = r * np.cos(np.deg2rad(rumbo))
            la, lo = _desplazar(LAT_BQ, LON_BQ, este, norte)
            filas.append({"pid": len(filas), "radio": r, "rumbo": rumbo,
                          "este": este, "norte": norte,
                          "lat": round(la, 3), "lon": round(lo, 3)})
    return pd.DataFrame(filas)


ANILLO = puntos_anillo()


def punto_mas_cercano(este_km: float, norte_km: float) -> int:
    dist = np.hypot(ANILLO["este"] - este_km, ANILLO["norte"] - norte_km)
    return int(ANILLO.loc[dist.idxmin(), "pid"])


# ==============================================================================
# 2. DESCARGA
# ==============================================================================

@st.cache_data(ttl=900, show_spinner="Vigilando el anillo de 100 km…")
def cargar_anillo() -> pd.DataFrame:
    """Formato largo: tiempo | pid | precipitation | cape."""
    params = {
        "latitude": ",".join(str(x) for x in ANILLO["lat"]),
        "longitude": ",".join(str(x) for x in ANILLO["lon"]),
        "hourly": "precipitation,cape",
        "models": "gfs_seamless",
        "timezone": TZ,
        "past_days": 1,
        "forecast_days": 3,
    }
    r = requests.get(API_FORECAST, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]

    piezas = []
    for pid, punto in enumerate(data):
        h = punto["hourly"]
        piezas.append(pd.DataFrame({
            "tiempo": pd.to_datetime(h["time"]),
            "pid": pid,
            "precipitation": pd.to_numeric(pd.Series(h["precipitation"]),
                                           errors="coerce"),
            "cape": pd.to_numeric(pd.Series(h.get("cape", [np.nan] * len(h["time"]))),
                                  errors="coerce"),
        }))
    return pd.concat(piezas, ignore_index=True)


# ==============================================================================
# 3. FLUJO DIRECTOR
# ==============================================================================

def flujo_director(d: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a d (que ya trae u_850, v_850, u_700… en km/h, convención
    'hacia dónde va') el vector de desplazamiento de las tormentas.
    """
    u = sum(PESOS_NIVEL[n] * d[f"u_{n}"] for n in PESOS_NIVEL)
    v = sum(PESOS_NIVEL[n] * d[f"v_{n}"] for n in PESOS_NIVEL)
    d = d.copy()
    d["u_dir"] = u
    d["v_dir"] = v
    d["vel_tormenta"] = np.hypot(u, v)                       # km/h
    # Rumbo HACIA el que se mueven (0 = norte) y DESDE donde vienen
    d["rumbo_hacia"] = (np.rad2deg(np.arctan2(u, v)) + 360) % 360
    d["rumbo_desde"] = (d["rumbo_hacia"] + 180) % 360
    return d


# ==============================================================================
# 4. ADVECCIÓN AGUAS ARRIBA
# ==============================================================================

def advectar(d: pd.DataFrame, anillo: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada hora t, mira la lluvia que había 1, 2 y 3 h antes en el punto
    desde donde viaja el aire hacia la ciudad.

    Columnas nuevas:
      senal_adv      0-1  lluvia en camino (1 = aguacero entrando)
      lluvia_arriba  mm   la mayor lluvia aguas arriba encontrada
      lluvia_lado    mm   la mayor lluvia del anillo FUERA de esa trayectoria
      eta_h          h    horas que faltan para que llegue
    """
    tabla = anillo.pivot_table(index="tiempo", columns="pid",
                               values="precipitation").sort_index()
    salida = {"senal_adv": [], "lluvia_arriba": [], "lluvia_lado": [],
              "eta_h": [], "pid_arriba": []}

    for _, r in d.iterrows():
        t = r["tiempo"]
        mejor, mm_mejor, eta, pid_mejor = 0.0, 0.0, np.nan, np.nan
        usados = set()

        if pd.notna(r["u_dir"]) and pd.notna(r["v_dir"]):
            for tau, peso in DECAIMIENTO.items():
                este = -r["u_dir"] * tau
                norte = -r["v_dir"] * tau
                if np.hypot(este, norte) < RADIOS_KM[0] * 0.5:
                    continue            # flujo muy débil: no hay "aguas arriba"
                pid = punto_mas_cercano(este, norte)
                usados.add(pid)
                t_origen = t - pd.Timedelta(hours=tau)
                if t_origen not in tabla.index:
                    continue
                mm = tabla.at[t_origen, pid]
                if pd.isna(mm):
                    continue
                s = (1 - np.exp(-mm / 1.5)) * peso
                if s > mejor:
                    mejor, mm_mejor, eta, pid_mejor = s, mm, tau, pid

        lado = 0.0
        if t in tabla.index:
            fila = tabla.loc[t].drop(labels=list(usados), errors="ignore")
            lado = float(fila.max()) if fila.notna().any() else 0.0

        salida["senal_adv"].append(mejor)
        salida["lluvia_arriba"].append(mm_mejor)
        salida["lluvia_lado"].append(lado)
        salida["eta_h"].append(eta)
        salida["pid_arriba"].append(pid_mejor)

    out = d.copy()
    for k, v in salida.items():
        out[k] = v
    return out


# ==============================================================================
# 5. TRAYECTORIA RECIENTE DE LA LLUVIA
# ==============================================================================

def trayectoria_reciente(anillo: pd.DataFrame, ahora: pd.Timestamp,
                         horas: int = 6) -> dict:
    """
    Centroide de la lluvia del anillo en las últimas `horas`. Si se desplaza
    de forma consistente, da el movimiento real que tuvieron los núcleos
    (según el modelo) para compararlo con el flujo director.
    """
    ini = ahora - pd.Timedelta(hours=horas)
    sub = anillo[(anillo["tiempo"] > ini) & (anillo["tiempo"] <= ahora)]
    sub = sub.merge(ANILLO[["pid", "este", "norte"]], on="pid")

    cent = []
    for t, g in sub.groupby("tiempo"):
        w = g["precipitation"].clip(lower=0)
        if w.sum() >= 1.0:
            cent.append({"tiempo": t,
                         "este": float((g["este"] * w).sum() / w.sum()),
                         "norte": float((g["norte"] * w).sum() / w.sum()),
                         "mm": float(w.sum())})
    if len(cent) < 3:
        return {"valida": False, "centroides": pd.DataFrame(cent)}

    c = pd.DataFrame(cent)
    hrs = (c["tiempo"] - c["tiempo"].iloc[0]).dt.total_seconds() / 3600
    ue = np.polyfit(hrs, c["este"], 1)[0]
    vn = np.polyfit(hrs, c["norte"], 1)[0]
    return {
        "valida": True,
        "centroides": c,
        "vel_kmh": float(np.hypot(ue, vn)),
        "rumbo_hacia": float((np.rad2deg(np.arctan2(ue, vn)) + 360) % 360),
        "dist_actual_km": float(np.hypot(c["este"].iloc[-1], c["norte"].iloc[-1])),
        "rumbo_actual": float((np.rad2deg(np.arctan2(c["este"].iloc[-1],
                                                     c["norte"].iloc[-1])) + 360) % 360),
        "se_acerca": bool(np.hypot(c["este"].iloc[-1], c["norte"].iloc[-1])
                          < np.hypot(c["este"].iloc[0], c["norte"].iloc[0]) - 5),
    }


# ==============================================================================
# RAYOS OBSERVADOS  |  GOES-19 GLM (Geostationary Lightning Mapper)
# ------------------------------------------------------------------------------
# El modelo dice dónde DEBERÍAN estar las tormentas; el GLM dice dónde ESTÁN.
# Donde hay rayos hay núcleo con lluvia (no solo yunque). Siguiendo los grupos
# de rayos cada 10 min se obtiene la trayectoria REAL de cada celda y se
# proyecta si pasa sobre Barranquilla en las próximas 3 h (nowcasting).
#
# Datos: repositorio público de NOAA en AWS, sin clave. Un archivo cada 20 s.
# Se toma 1 de cada 3 (≈1 por minuto) para que cargue rápido: el conteo se
# multiplica por 3 para estimar la tasa real.
# ==============================================================================

GLM_BUCKET = "noaa-goes19"
GLM_RADIO_KM = 160          # zona vigilada alrededor de la ciudad
GLM_MINUTOS = 60            # historia usada para la trayectoria
GLM_PASO = 3                # 1 de cada 3 archivos
GLM_BIN_MIN = 10            # ventanas de 10 min para seguir las celdas
GLM_CELDA_KM = 4            # rejilla para agrupar rayos
GLM_UNION_KM = 10           # rayos a <10 km = misma tormenta
GLM_SALTO_KM = 10           # desplazamiento máximo de una celda en 10 min (60 km/h)
GLM_IMPACTO_KM = 12         # pasa "sobre la ciudad" si su centro se acerca a menos de esto
GLM_ENCIMA_KM = 10          # rayos a <10 km de la ciudad = tormenta encima
GLM_MIN_BINS = 4            # trayectoria "observada" solo con >=4 ventanas (>=30 min)
GLM_MAX_RMS_KM = 6          # dispersión máxima alrededor de la recta de movimiento
GLM_PTS_MAX = 30            # aporte máximo de los rayos a la probabilidad
UTC_A_COL = pd.Timedelta(hours=-5)


def _atributo(ds, nombre):
    v = ds.attrs.get(nombre)
    if v is None:
        return None
    v = np.asarray(v).ravel()
    if v.size == 0:
        return None
    v = v[0]
    return v.decode() if isinstance(v, bytes) else v


def _leer_variable(ds) -> np.ndarray:
    """Lee una variable netCDF4 con h5py aplicando _FillValue, _Unsigned y escala."""
    crudo = ds[()]
    mascara = np.zeros(crudo.shape, dtype=bool)
    relleno = _atributo(ds, "_FillValue")
    if relleno is not None:
        mascara = crudo == relleno
    if str(_atributo(ds, "_Unsigned")).lower() == "true" and crudo.dtype.kind == "i":
        crudo = crudo.view(crudo.dtype.str.replace("i", "u"))
    out = crudo.astype("float64")
    escala = _atributo(ds, "scale_factor")
    offset = _atributo(ds, "add_offset")
    if escala is not None:
        out = out * float(escala)
    if offset is not None:
        out = out + float(offset)
    out[mascara] = np.nan
    return out


def _inicio_archivo(clave: str):
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", clave)
    if not m:
        return None
    a, dj, h, mi, s = (int(x) for x in m.groups())
    return (pd.Timestamp(year=a, month=1, day=1) + pd.Timedelta(days=dj - 1,
            hours=h, minutes=mi, seconds=s))


def _listar_glm(t_utc: pd.Timestamp) -> list[str]:
    prefijo = f"GLM-L2-LCFA/{t_utc:%Y}/{t_utc.dayofyear:03d}/{t_utc:%H}/"
    r = requests.get(f"https://{GLM_BUCKET}.s3.amazonaws.com/",
                     params={"list-type": "2", "prefix": prefijo}, timeout=20)
    r.raise_for_status()
    return re.findall(r"<Key>([^<]+)</Key>", r.text)


def _rayos_de_archivo(clave: str):
    import h5py
    r = requests.get(f"https://{GLM_BUCKET}.s3.amazonaws.com/{clave}", timeout=30)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as f:
        if "flash_lat" not in f:
            return np.array([]), np.array([])
        return _leer_variable(f["flash_lat"]), _leer_variable(f["flash_lon"])


def _a_km(lat, lon):
    norte = (np.asarray(lat) - LAT_BQ) * KM_POR_GRADO
    este = (np.asarray(lon) - LON_BQ) * KM_POR_GRADO * np.cos(np.deg2rad(LAT_BQ))
    return este, norte


@st.cache_data(ttl=240, show_spinner="Descargando rayos del GOES-19…")
def cargar_rayos(marca_tiempo: str):
    """Rayos de los últimos GLM_MINUTOS en GLM_RADIO_KM. Devuelve (df, info)."""
    ahora = pd.Timestamp.now("UTC").tz_localize(None)
    claves = []
    for h in (1, 0):
        try:
            claves += _listar_glm(ahora - pd.Timedelta(hours=h))
        except Exception:
            pass
    pares = [(c, _inicio_archivo(c)) for c in claves]
    pares = sorted((c, t) for c, t in pares
                   if t is not None and t >= ahora - pd.Timedelta(minutes=GLM_MINUTOS))
    pares.sort(key=lambda x: x[1])
    if not pares:
        raise RuntimeError("no hay archivos GLM recientes en noaa-goes19")
    elegidos = pares[::-1][::GLM_PASO][::-1]      # siempre incluye el más nuevo

    filas = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futuros = {ex.submit(_rayos_de_archivo, c): t for c, t in elegidos}
        for fut, t in futuros.items():
            try:
                lat, lon = fut.result()
            except Exception:
                continue
            if len(lat) == 0:
                continue
            este, norte = _a_km(lat, lon)
            dist = np.hypot(este, norte)
            sel = np.isfinite(dist) & (dist <= GLM_RADIO_KM)
            for e, n in zip(este[sel], norte[sel]):
                filas.append((t + UTC_A_COL, float(e), float(n)))

    df = pd.DataFrame(filas, columns=["tiempo", "este", "norte"])
    if not df.empty:
        df["dist"] = np.hypot(df["este"], df["norte"])
        df["rumbo"] = (np.rad2deg(np.arctan2(df["este"], df["norte"])) + 360) % 360
    info = {
        "archivos": len(elegidos),
        "ultimo": pares[-1][1] + UTC_A_COL,
        "latencia_min": (ahora - pares[-1][1]).total_seconds() / 60,
    }
    return df, info


def _agrupar(este, norte):
    """Agrupa rayos en tormentas: rejilla de 4 km + unión de celdas a <15 km."""
    if len(este) == 0:
        return []
    cel = pd.DataFrame({"i": np.floor(np.asarray(este) / GLM_CELDA_KM),
                        "j": np.floor(np.asarray(norte) / GLM_CELDA_KM),
                        "e": este, "n": norte})
    g = cel.groupby(["i", "j"]).agg(e=("e", "mean"), n=("n", "mean"),
                                    c=("e", "size")).reset_index()
    pe, pn, pc = g["e"].values, g["n"].values, g["c"].values
    padre = list(range(len(g)))

    def raiz(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    d2 = (pe[:, None] - pe[None, :]) ** 2 + (pn[:, None] - pn[None, :]) ** 2
    ii, jj = np.where(np.triu(d2 <= GLM_UNION_KM ** 2, 1))
    for a, b in zip(ii, jj):
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[ra] = rb
    grupos = {}
    for k in range(len(g)):
        grupos.setdefault(raiz(k), []).append(k)
    out = []
    for idx in grupos.values():
        w = pc[idx]
        out.append({"este": float((pe[idx] * w).sum() / w.sum()),
                    "norte": float((pn[idx] * w).sum() / w.sum()),
                    "n": int(w.sum())})
    return out


def seguir_celdas(rayos: pd.DataFrame, u_mod: float, v_mod: float,
                  ahora_local: pd.Timestamp) -> list[dict]:
    """
    Identifica cada tormenta activa y reconstruye su trayectoria de la última
    hora. Una trayectoria solo se acepta como "observada" si es CONSISTENTE:
    al menos 30 min de historia, puntos alineados sobre una recta y una
    dirección compatible con el flujo del modelo.

    Por qué tantas exigencias: las celdas tropicales viven 20-40 min. Cuando
    una muere y nace otra a 10-15 km, un seguidor ingenuo las une y "ve" un
    desplazamiento falso, a veces directo hacia la ciudad. Eso producía falsos
    positivos de +25 a +40 puntos con el cielo despejado.
    """
    if rayos.empty:
        return []
    fin = rayos["tiempo"].max()
    rayos = rayos.assign(bin=((fin - rayos["tiempo"]).dt.total_seconds()
                              // (GLM_BIN_MIN * 60)).astype(int))
    por_bin = {b: _agrupar(g["este"].values, g["norte"].values)
               for b, g in rayos.groupby("bin")}
    if 0 not in por_bin:
        return []
    vel_mod = np.array([u_mod, v_mod], dtype=float)

    celdas = []
    for c in por_bin[0]:
        pista = [(0, c["este"], c["norte"])]
        actual = np.array([c["este"], c["norte"]])
        vel_est = None                                  # km/h, hacia el pasado se resta
        usados = {b: set() for b in por_bin}
        for b in range(1, max(por_bin) + 1):
            cand = por_bin.get(b, [])
            if not cand:
                continue
            salto = b - pista[-1][0]
            # Posición esperada: con 2+ puntos se extrapola la velocidad estimada
            esperada = actual - (vel_est * salto * GLM_BIN_MIN / 60
                                 if vel_est is not None else 0)
            dist = [np.hypot(x["este"] - esperada[0], x["norte"] - esperada[1])
                    for x in cand]
            k = int(np.argmin(dist))
            if dist[k] <= GLM_SALTO_KM * salto and k not in usados[b]:
                usados[b].add(k)
                actual = np.array([cand[k]["este"], cand[k]["norte"]])
                pista.append((b, actual[0], actual[1]))
                if len(pista) >= 2:
                    hrs = np.array([p[0] for p in pista]) * GLM_BIN_MIN / 60
                    vel_est = np.array([
                        -np.polyfit(hrs, [p[1] for p in pista], 1)[0],
                        -np.polyfit(hrs, [p[2] for p in pista], 1)[0]])

        fuente, confianza = "modelo", 0.3
        u, v = float(vel_mod[0]), float(vel_mod[1])
        if len(pista) >= GLM_MIN_BINS:
            hrs = -np.array([p[0] for p in pista]) * GLM_BIN_MIN / 60
            pe = np.polyfit(hrs, [p[1] for p in pista], 1)
            pn = np.polyfit(hrs, [p[2] for p in pista], 1)
            res = np.hypot(np.polyval(pe, hrs) - [p[1] for p in pista],
                           np.polyval(pn, hrs) - [p[2] for p in pista])
            rms = float(np.sqrt(np.mean(res ** 2)))
            vo = np.array([pe[0], pn[0]])
            rapidez = float(np.hypot(*vo))
            # Coherencia con el flujo del modelo (si el flujo no es casi nulo)
            nm = float(np.hypot(*vel_mod))
            cos_ang = float(vo @ vel_mod / (rapidez * nm)) if rapidez > 1 and nm > 3 else 1.0
            if rms <= GLM_MAX_RMS_KM and rapidez <= 60 and cos_ang >= 0.0:
                u, v = float(vo[0]), float(vo[1])
                fuente, confianza = "observada", 1.0
            elif rms <= GLM_MAX_RMS_KM and rapidez <= 60:
                # Alineada pero contra el flujo: posible propagación discreta
                u, v = float(vo[0]), float(vo[1])
                fuente, confianza = "dudosa", 0.4

        p0 = np.array([c["este"], c["norte"]])
        vel = np.array([u, v])
        v2 = float(vel @ vel)
        # Celdas jóvenes no se proyectan lejos: la mayoría muere antes de 45 min
        horizonte = 3.0 if fuente == "observada" else 0.75
        t_min = float(np.clip(-(p0 @ vel) / v2, 0, horizonte)) if v2 > 1 else 0.0
        d_min = float(np.linalg.norm(p0 + vel * t_min))
        d_ahora = float(np.linalg.norm(p0))
        celdas.append({
            "este": c["este"], "norte": c["norte"],
            "dist": d_ahora,
            "rumbo": float((np.rad2deg(np.arctan2(*p0)) + 360) % 360),
            "rayos_10min": c["n"] * GLM_PASO,
            "u": u, "v": v, "vel": float(np.hypot(u, v)),
            "rumbo_hacia": float((np.rad2deg(np.arctan2(u, v)) + 360) % 360),
            "fuente": fuente, "confianza": confianza,
            "d_min": d_min,
            "eta_h": t_min,
            "viene": d_min <= GLM_IMPACTO_KM and t_min > 0 and d_ahora > GLM_ENCIMA_KM,
            "pista": pista,
            "edad_min": float((ahora_local - fin).total_seconds() / 60),
        })
    return sorted(celdas, key=lambda x: x["dist"])


def rayos_encima(rayos: pd.DataFrame, minutos: int = 15) -> int:
    """Rayos reales (estimados) a menos de GLM_ENCIMA_KM en los últimos minutos."""
    if rayos is None or rayos.empty:
        return 0
    fin = rayos["tiempo"].max()
    rec = rayos[(rayos["tiempo"] >= fin - pd.Timedelta(minutes=minutos))
                & (rayos["dist"] <= GLM_ENCIMA_KM)]
    return int(len(rec) * GLM_PASO)


def puntos_nowcast(celdas: list[dict], hay_datos: bool, hora_rel: int,
                   rayos_hora: int = 0, n_encima: int = 0):
    """Puntos (log-odds) que aportan los rayos observados a la hora relativa 0-3."""
    if not hay_datos or hora_rel > 3:
        return 0.0, None

    # 1. Tormenta realmente encima: rayos a <10 km en los últimos 15 min
    if hora_rel == 0 and n_encima >= 3:
        pts = GLM_PTS_MAX * (1 - np.exp(-n_encima / 15))
        return pts, (f"⚡ **Tormenta activa sobre la ciudad:** ~{n_encima} rayos "
                     f"a menos de {GLM_ENCIMA_KM} km en los últimos 15 min")

    if not celdas:
        if hora_rel <= 1 and rayos_hora > 0:
            return -5.0, ("⚡ **Tormentas cercanas apagándose:** hubo rayos en la "
                          "última hora, pero ninguno en los últimos 10 min")
        if hora_rel <= 1:
            return -10.0, (f"⚡ **Sin rayos en {GLM_RADIO_KM} km** durante la última "
                           "hora (GOES-19): no hay tormentas activas cerca")
        return 0.0, None

    # 2. Tormentas que vienen, ponderadas por la confianza de su trayectoria
    mejor, texto = 0.0, None
    for c in celdas:
        if not c["viene"] or not (hora_rel <= c["eta_h"] < hora_rel + 1):
            continue
        intensidad = 1 - np.exp(-c["rayos_10min"] / 20)
        cercania = 1 - min(c["d_min"], GLM_IMPACTO_KM) / (GLM_IMPACTO_KM * 1.5)
        tau = 2.0 if c["fuente"] == "observada" else 0.5
        vida = np.exp(-c["eta_h"] / tau)
        pts = GLM_PTS_MAX * intensidad * cercania * vida * c["confianza"]
        if pts > mejor:
            mejor = pts
            texto = (f"⚡ **Tormenta en camino** desde el "
                     f"{rumbo_a_texto(c['rumbo'])} a {c['dist']:.0f} km, "
                     f"llega en ~{c['eta_h']*60:.0f} min "
                     f"(trayectoria {c['fuente']}, {c['vel']:.0f} km/h)")
    if mejor >= 2:
        return mejor, texto
    if any(c["viene"] and c["fuente"] == "observada" for c in celdas):
        return 0.0, None          # viene, pero en otra hora: no penalizar
    if hora_rel <= 1:
        cerca = celdas[0]
        return -8.0, (f"⚡ **Rayos cerca pero no vienen:** celda a {cerca['dist']:.0f} km "
                      f"al {rumbo_a_texto(cerca['rumbo'])}, sin trayectoria hacia "
                      "la ciudad")
    return 0.0, None


# ==============================================================================
# ONDAS TROPICALES SEGÚN EL NHC  |  Tropical Weather Discussion (texto oficial)
# ------------------------------------------------------------------------------
# Los meteorólogos del Centro Nacional de Huracanes analizan a mano las ondas
# tropicales con satélite, sondeos y boyas, y publican su posición 4 veces al
# día en texto. Es la referencia más confiable de dónde está cada onda: el
# detector de la app (que mira solo el modelo GFS) puede no verlas o ubicarlas
# mal. Aquí se lee ese texto, se extrae longitud y velocidad de cada onda y se
# calcula cuándo pasa su eje por Barranquilla.
# ==============================================================================

NHC_URLS = {
    "atlantico": "https://www.nhc.noaa.gov/text/MIATWDAT.shtml",
    "pacifico": "https://www.nhc.noaa.gov/text/MIATWDEP.shtml",
}
NHC_RETRASO_ANALISIS_H = 3     # el análisis es ~3 h anterior a la emisión
NHC_BAJA_RADIO_KM = 350
MESES_EN = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _texto_plano(html: str) -> str:
    m = re.search(r"<pre[^>]*>(.*?)</pre>", html, flags=re.S | re.I)
    t = m.group(1) if m else html
    t = re.sub(r"<[^>]+>", "", t)
    return (t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("\r", ""))


def _hora_emision(texto: str):
    m = re.search(r"(\d{3,4})\s*UTC\s+\w{3}\s+(\w{3})\s+(\d{1,2})\s+(\d{4})", texto)
    if not m:
        return None
    hhmm, mes, dia, anio = m.groups()
    hhmm = hhmm.zfill(4)
    try:
        return pd.Timestamp(year=int(anio), month=MESES_EN[mes.upper()[:3]],
                            day=int(dia), hour=int(hhmm[:2]), minute=int(hhmm[2:]))
    except (KeyError, ValueError):
        return None


def _seccion(texto: str, titulo: str) -> str:
    m = re.search(r"^\s*\.{3}\s*" + titulo + r"\s*\.{3}\s*$(.*?)(?=^\s*\.{3}[A-Z][A-Z /]+\.{3}\s*$|\Z)",
                  texto, flags=re.S | re.M | re.I)
    return m.group(1) if m else ""


def interpretar_ondas(texto: str) -> list[dict]:
    """Extrae cada onda tropical: longitud del eje, extensión, velocidad, actividad."""
    seccion = _seccion(texto, "TROPICAL WAVES")
    ondas = []
    for parrafo in re.split(r"\n\s*\n", seccion):
        p = " ".join(parrafo.split())
        if not re.search(r"tropical wave", p, re.I):
            continue
        m_lon = (re.search(r"(?:axis|along|near|is at)\D{0,30}?(\d{2,3}(?:\.\d)?)\s*W\b", p, re.I)
                 or re.search(r"(\d{2,3}(?:\.\d)?)\s*W\b", p))
        if not m_lon:
            continue
        lon_w = float(m_lon.group(1))

        lat_s, lat_n = 0.0, 20.0
        m_ext = re.search(r"(\d{1,2})\s*N\s*(?:-|to)\s*(\d{1,2})\s*N", p, re.I)
        m_sur = re.search(r"south of\s+(\d{1,2})\s*N", p, re.I)
        m_nor = re.search(r"north of\s+(\d{1,2})\s*N", p, re.I)
        if m_ext and m_ext.start() < m_lon.end() + 80:
            lat_s, lat_n = sorted((float(m_ext.group(1)), float(m_ext.group(2))))
        else:
            if m_sur:
                lat_n = float(m_sur.group(1))
            if m_nor:
                lat_s, lat_n = float(m_nor.group(1)), max(lat_n, 25.0) if not m_sur else lat_n

        if re.search(r"stationary|nearly stationary", p, re.I):
            kt = 0.0
        else:
            m_v = (re.search(r"(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*(?:kt|knots)", p, re.I)
                   or re.search(r"(?:at|around|near|about)\s+(\d{1,2})\s*(?:kt|knots)", p, re.I))
            if m_v and m_v.lastindex == 2:
                kt = (float(m_v.group(1)) + float(m_v.group(2))) / 2
            elif m_v:
                kt = float(m_v.group(1))
            else:
                kt = 15.0                       # típico en el Caribe

        if re.search(r"no significant|limited|isolated", p, re.I) and \
                not re.search(r"numerous|scattered", p, re.I):
            actividad = 0.4
        elif re.search(r"numerous|strong", p, re.I):
            actividad = 1.0
        else:
            actividad = 0.75
        ondas.append({"lon_w": lon_w, "lat_s": lat_s, "lat_n": lat_n, "kt": kt,
                      "actividad": actividad, "texto": p[:400]})
    return ondas


def interpretar_bajas(texto: str) -> list[dict]:
    bajas = []
    p = " ".join(texto.split())
    for m in re.finditer(r"(\d{4})\s*mb\s+low[^.]{0,80}?(\d{1,2}(?:\.\d)?)\s*N\s*"
                         r"(\d{2,3}(?:\.\d)?)\s*W", p, re.I):
        bajas.append({"mb": float(m.group(1)), "lat": float(m.group(2)),
                      "lon_w": float(m.group(3))})
    return bajas


@st.cache_data(ttl=1800, show_spinner="Leyendo el análisis del NHC…")
def cargar_nhc() -> dict:
    textos, emision = {}, None
    for clave, url in NHC_URLS.items():
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "caribe-weather"})
            r.raise_for_status()
            textos[clave] = _texto_plano(r.text)
            e = _hora_emision(textos[clave])
            if clave == "atlantico":
                emision = e
        except Exception:
            continue
    if "atlantico" not in textos:
        raise RuntimeError("no se pudo descargar la discusión del NHC")
    bajas = []
    for t in textos.values():
        bajas += interpretar_bajas(t)
    return {"ondas": interpretar_ondas(textos["atlantico"]), "bajas": bajas,
            "emision_utc": emision}


def paso_ondas(nhc: dict) -> list[dict]:
    """Hora (local) a la que cada onda cruza la longitud de Barranquilla."""
    if not nhc.get("emision_utc"):
        return []
    valida = nhc["emision_utc"] - pd.Timedelta(hours=NHC_RETRASO_ANALISIS_H) + UTC_A_COL
    lon_bq_w = -LON_BQ
    km_por_grado_lon = KM_POR_GRADO * np.cos(np.deg2rad(LAT_BQ))
    salida = []
    for o in nhc["ondas"]:
        if not (o["lat_s"] - 1.5 <= LAT_BQ <= o["lat_n"] + 1.5):
            continue                            # la onda no llega a esta latitud
        grados = lon_bq_w - o["lon_w"]          # >0: la onda está al este
        if abs(grados) > 30:
            continue
        vel_deg_h = o["kt"] * 1.852 / km_por_grado_lon
        if vel_deg_h <= 0:
            if abs(grados) <= 1.5:
                salida.append({**o, "paso": valida, "dist_km": abs(grados) * km_por_grado_lon})
            continue
        horas = grados / vel_deg_h
        salida.append({**o, "paso": valida + pd.Timedelta(hours=float(horas)),
                       "dist_km": abs(grados) * km_por_grado_lon,
                       "al_este": grados > 0})
    return sorted(salida, key=lambda x: abs((x["paso"] - valida).total_seconds()))


def puntos_nhc(t: pd.Timestamp, pasos: list[dict], bajas: list[dict],
               valida: pd.Timestamp | None = None):
    """Puntos por onda/baja del NHC para la hora local t."""
    mejor, texto = 0.0, None
    for o in pasos:
        dt = (t - o["paso"]).total_seconds() / 3600
        # Máximo ~3 h después del paso del eje; ventana útil de -12 a +18 h
        w = float(np.exp(-((dt - 3) / 9) ** 2))
        pts = PESOS["onda_nhc"] * w * o["actividad"]
        if pts > mejor and pts >= 1:
            mejor = pts
            cuando = ("pasó hace " if dt >= 0 else "llega en ") + f"{abs(dt):.0f} h"
            texto = (f"🌊 **Onda tropical (NHC)** con eje en {o['lon_w']:.0f}°W, "
                     f"{cuando} a Barranquilla")
    for b in bajas:
        dn = (b["lat"] - LAT_BQ) * KM_POR_GRADO
        de = (-b["lon_w"] - LON_BQ) * KM_POR_GRADO * np.cos(np.deg2rad(LAT_BQ))
        dist = float(np.hypot(dn, de))
        if dist < NHC_BAJA_RADIO_KM:
            pts = PESOS["baja_nhc"] * (1 - dist / NHC_BAJA_RADIO_KM)
            if valida is not None:      # una baja analizada envejece: ~1 día de vigencia
                horas = max(0.0, (t - valida).total_seconds() / 3600)
                pts *= float(np.exp(-horas / 24))
            if pts > 2:
                mejor += pts
                texto = ((texto + " · ") if texto else "") + \
                    (f"📉 **Baja de {b['mb']:.0f} hPa (NHC)** a {dist:.0f} km "
                     f"({b['lat']:.0f}°N {b['lon_w']:.0f}°W)")
    return mejor, texto


# ==============================================================================
# 2. EXTRACCIÓN DE DATOS
# ==============================================================================
# NOTA IMPORTANTE SOBRE MODELOS:
#   - precipitation_probability, cape, convective_inhibition, lifted_index y
#     total_column_integrated_water_vapour NO existen en ecmwf_ifs025.
#     Open-Meteo los calcula/entrega para GFS. Por eso GFS es el modelo base.
#   - ECMWF se usa como segunda opinión en presión y humedad de capa media,
#     que es donde su desempeño sinóptico es superior.

VARS_GFS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "pressure_msl", "surface_pressure", "precipitation",
    "precipitation_probability", "cloud_cover", "cloud_cover_low",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "cape", "convective_inhibition", "lifted_index",
    "total_column_integrated_water_vapour",
    "geopotential_height_500hPa", "geopotential_height_700hPa",
    "relative_humidity_700hPa", "relative_humidity_850hPa",
    "relative_humidity_500hPa",
    "wind_speed_700hPa", "wind_direction_700hPa",
    "wind_speed_500hPa", "wind_direction_500hPa",
    "wind_speed_850hPa", "wind_direction_850hPa",
    "vertical_velocity_700hPa",
    "temperature_850hPa", "temperature_500hPa",
]

VARS_ECMWF = [
    "pressure_msl", "precipitation", "temperature_2m",
    "relative_humidity_2m", "relative_humidity_700hPa",
    "geopotential_height_500hPa",
    "wind_speed_10m", "wind_direction_10m",
]


@st.cache_data(ttl=900, show_spinner=False)
def _pedir(modelo: str, variables: list[str]) -> dict:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(variables),
        "timezone": TZ,
        "models": modelo,
        "past_days": 7,        # 7 días: necesarios para medir la marea diaria
        "forecast_days": 4,
    }
    r = requests.get(API_URL, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def _a_dataframe(payload: dict, sufijo: str = "") -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame({"tiempo": pd.to_datetime(h["time"])})
    for k, v in h.items():
        if k == "time":
            continue
        df[f"{k}{sufijo}"] = pd.to_numeric(pd.Series(v), errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner="Descargando modelos GFS y ECMWF…")
def cargar_datos() -> tuple[pd.DataFrame, list[str]]:
    avisos = []

    base = _a_dataframe(_pedir("gfs_seamless", VARS_GFS))

    try:
        ecmwf = _a_dataframe(_pedir("ecmwf_ifs025", VARS_ECMWF), sufijo="_ec")
        base = base.merge(ecmwf, on="tiempo", how="left")
        # ECMWF 0.25 viene cada 3 h: interpolamos para comparar hora a hora.
        cols_ec = [c for c in base.columns if c.endswith("_ec")]
        base[cols_ec] = base[cols_ec].interpolate(limit_direction="both")
    except Exception as e:  # la app sigue viva aunque falle el segundo modelo
        avisos.append(f"ECMWF no disponible ({e}). Se usa solo GFS.")

    return base, avisos


def col(df: pd.DataFrame, nombre: str) -> pd.Series:
    """Devuelve la columna o una serie de NaN si el modelo no la entregó."""
    if nombre in df.columns:
        return df[nombre]
    return pd.Series(np.nan, index=df.index, dtype="float64")


# ==============================================================================
# 3. ÍNDICES DERIVADOS (aquí está la física)
# ==============================================================================

def calcular_indices(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # --- Presión: usar SIEMPRE pressure_msl, no surface_pressure -------------
    # surface_pressure depende de la altura del punto; para análisis sinóptico
    # (bajas, vaguadas, ondas) el campo correcto es el reducido al nivel del mar.
    p_cruda = col(d, "pressure_msl")
    d["pres"] = p_cruda              # lo que se muestra al usuario

    # --- MAREA ATMOSFÉRICA ---------------------------------------------------
    # En el trópico la presión baja 2-3 hPa TODOS los días entre las 10:00 y
    # las 16:00 y vuelve a subir en la noche (marea semidiurna). Sin quitarla,
    # cada tarde se disparaban "baja presión", "anomalía" y "caída barométrica"
    # (44 puntos) aunque no hubiera ningún sistema. Se resta el ciclo medio
    # por hora del día y se trabaja con la presión "desmareada".
    hora = d["tiempo"].dt.hour
    marea = p_cruda.groupby(hora).transform("mean") - p_cruda.mean()
    p = p_cruda - marea
    d["marea"] = marea
    d["pres_dm"] = p

    # Media móvil de ~5 días centrada = "normal local" dinámica.
    d["pres_base"] = p.rolling(120, min_periods=24, center=True).mean()
    d["pres_anom"] = p - d["pres_base"]

    # Tendencia SIN marea: solo cuenta la caída que no explica la hora del día
    d["pres_tend_3h"] = p.diff(3)
    d["pres_tend_12h"] = p.diff(12)
    d["pres_tend_3h_cruda"] = p_cruda.diff(3)

    # Mínimo local de presión: corazón del eje de onda / baja
    ventana = 9
    minimo_local = p.rolling(ventana, center=True, min_periods=5).min()
    d["es_minimo_presion"] = (p <= minimo_local + 0.15) & (
        p.rolling(25, center=True, min_periods=9).max() - p >= 1.0
    )

    # --- Altura geopotencial: vaguadas en niveles medios y altos -------------
    for niv, alias in (("500hPa", "500"), ("700hPa", "700")):
        g = col(d, f"geopotential_height_{niv}")
        d[f"gph{alias}"] = g
        d[f"gph{alias}_base"] = g.rolling(120, min_periods=24, center=True).mean()
        d[f"gph{alias}_anom"] = g - d[f"gph{alias}_base"]
        # curvatura temporal: un mínimo en la serie = paso del eje de vaguada
        d[f"gph{alias}_curv"] = g.diff().diff()

    # --- Viento: componentes y cizalladura ----------------------------------
    for niv, alias in (("10m", "sfc"), ("850hPa", "850"),
                       ("700hPa", "700"), ("500hPa", "500")):
        spd = col(d, f"wind_speed_{niv}")
        dr = np.deg2rad(col(d, f"wind_direction_{niv}"))
        d[f"u_{alias}"] = -spd * np.sin(dr)   # componente zonal (+ = del oeste)
        d[f"v_{alias}"] = -spd * np.cos(dr)   # componente meridional (+ = del sur)
        d[f"dir_{alias}"] = col(d, f"wind_direction_{niv}")
        d[f"spd_{alias}"] = spd

    # Cizalladura 850-500 hPa: mucha cizalladura organiza, demasiada desorganiza
    d["cizalladura_850_500"] = np.hypot(d["u_500"] - d["u_850"],
                                        d["v_500"] - d["v_850"])

    # Giro ciclónico de la componente meridional en 700 hPa:
    # en una onda del Este, delante del eje el viento rola del ENE al ESE/SE
    # (v aumenta = flujo del sur) y detrás regresa al NE.
    d["dv700_6h"] = d["v_700"].diff(6)

    # --- Humedad y movimiento vertical --------------------------------------
    d["rh700"] = col(d, "relative_humidity_700hPa")
    d["rh850"] = col(d, "relative_humidity_850hPa")
    d["rh500"] = col(d, "relative_humidity_500hPa")
    d["rh_profunda"] = d[["rh850", "rh700", "rh500"]].mean(axis=1)
    d["rh700_tend_12h"] = d["rh700"].diff(12)

    d["pw"] = col(d, "total_column_integrated_water_vapour")
    d["omega700"] = col(d, "vertical_velocity_700hPa") * SIGNO_ASCENSO  # + = ascenso

    # --- Termodinámica -------------------------------------------------------
    d["cape"] = col(d, "cape")
    d["cin"] = col(d, "convective_inhibition")
    d["li"] = col(d, "lifted_index")
    d["prob_base"] = col(d, "precipitation_probability")
    d["lluvia"] = col(d, "precipitation")

    # --- Consenso de modelos: divergencia GFS vs ECMWF en presión -----------
    if "pressure_msl_ec" in d.columns:
        d["pres_ec"] = d["pressure_msl_ec"]
        d["desacuerdo_pres"] = (d["pres"] - d["pres_ec"]).abs()
    else:
        d["pres_ec"] = np.nan
        d["desacuerdo_pres"] = np.nan

    d["hora_local"] = d["tiempo"].dt.hour

    # --- Flujo director: hacia dónde y a qué velocidad viajan las tormentas --
    d = flujo_director(d)
    return d


# ==============================================================================
# 4. DETECTORES SINÓPTICOS
# ==============================================================================

def detectar_onda_tropical(d: pd.DataFrame, i: int) -> tuple[float, list[str]]:
    """
    Firma clásica del paso de una onda del Este sobre el Caribe:
      1. Mínimo relativo de presión con amplitud >= 1 hPa.
      2. Rolada del viento: de ENE a ESE/SE (componente meridional sube)
         delante del eje, y regreso al NE detrás.
      3. Humedecimiento marcado de la capa media (RH700 sube y supera 60%).
      4. Vaguada en 700 hPa (anomalía negativa de altura geopotencial).
      5. Ascenso en 700 hPa.
    Se exige un mínimo de 3 señales para declarar onda probable.
    """
    r = d.iloc[i]
    señales, detalle = 0, []

    if bool(r.get("es_minimo_presion", False)):
        señales += 1
        detalle.append("mínimo barométrico relativo")

    if pd.notna(r["dv700_6h"]) and r["dv700_6h"] >= 2.0:
        señales += 1
        detalle.append("rolada del viento en 700 hPa hacia el SE")

    if pd.notna(r["rh700"]) and pd.notna(r["rh700_tend_12h"]):
        if r["rh700"] >= UMBRALES["rh700_humedo"] and r["rh700_tend_12h"] >= 12:
            señales += 1
            detalle.append("humedecimiento rápido de la capa media")

    if pd.notna(r["gph700_anom"]) and r["gph700_anom"] <= UMBRALES["gph700_anom"]:
        señales += 1
        detalle.append("vaguada en 700 hPa")

    if pd.notna(r["omega700"]) and r["omega700"] > 0.03:
        señales += 1
        detalle.append("ascenso forzado en niveles medios")

    if señales >= 3:
        # confianza proporcional al número de señales coincidentes
        return señales / 5.0, detalle
    return 0.0, detalle


FACTOR_TERMO = 0.6   # el GFS ya incluye la termodinámica en su probabilidad:
                     # sumarla completa otra vez la contaría dos veces


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _sigmoide(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def evaluar_hora(d: pd.DataFrame, i: int) -> dict:
    """
    Devuelve probabilidad combinada, detonantes y supresores para la hora i.

    Cambios frente a la versión anterior (que llegaba a 100% con facilidad):
      1. Los puntos se suman en log-odds, no en porcentaje directo.
      2. Los factores termodinámicos (CAPE, LI, PW, humedad, calor) miden casi
         lo mismo: cuenta el mayor y solo el 30% de los demás.
      3. Con flujo director fuerte, la convección local pesa menos y manda lo
         que viene aguas arriba. Una tormenta cercana que el viento aleja ya
         no sube la probabilidad: la baja.
      4. Tope de 92% y piso de 3%.
    """
    r = d.iloc[i]
    sinopticos, termo, negativos = [], [], []   # (puntos, texto)

    # ---------- 1. Sistemas de baja presión ----------
    if pd.notna(r["pres_dm"]):
        if r["pres_dm"] <= UMBRALES["pres_baja_abs"]:
            sinopticos.append((PESOS["baja_presion_absoluta"],
                f"**Baja presión activa (L):** {r['pres_dm']:.1f} hPa sin marea diaria"))
        elif r["pres_dm"] >= UMBRALES["pres_alta_abs"]:
            negativos.append((PESOS["alta_presion"],
                f"**Dorsal anticiclónica:** {r['pres_dm']:.1f} hPa sin marea, aire estable"))

    if pd.notna(r["pres_anom"]) and r["pres_anom"] <= UMBRALES["anom_pres_baja"]:
        sinopticos.append((PESOS["anomalia_presion"],
            f"**Anomalía negativa de presión:** {r['pres_anom']:+.1f} hPa "
            "bajo la media de 5 días"))

    if pd.notna(r["pres_tend_3h"]) and r["pres_tend_3h"] <= UMBRALES["caida_3h"]:
        sinopticos.append((PESOS["caida_barometrica"],
            f"**Caída barométrica:** {r['pres_tend_3h']:+.1f} hPa en 3 h"))

    # ---------- 2. Vaguadas en altura ----------
    if pd.notna(r["gph500_anom"]) and r["gph500_anom"] <= UMBRALES["gph500_anom"]:
        sinopticos.append((PESOS["vaguada_500"],
            f"**Vaguada en 500 hPa:** {r['gph500_anom']:+.0f} mgp de anomalía"))

    if pd.notna(r["gph700_anom"]) and r["gph700_anom"] <= UMBRALES["gph700_anom"]:
        sinopticos.append((PESOS["vaguada_700"],
            f"**Vaguada en 700 hPa:** {r['gph700_anom']:+.0f} mgp de anomalía"))

    # ---------- 3. Onda tropical ----------
    conf_onda, detalle_onda = detectar_onda_tropical(d, i)
    # Si hay análisis del NHC, manda el NHC: el detector del modelo pesa la mitad
    f_onda_mod = 0.5 if bool(r.get("nhc_ok", False)) else 1.0
    if conf_onda > 0:
        sinopticos.append((PESOS["onda_tropical"] * conf_onda * f_onda_mod,
            f"🌀 **Onda tropical probable** ({int(conf_onda*100)}% de firma): "
            + ", ".join(detalle_onda)))

    n_pts = r.get("nhc_pts", np.nan)
    if pd.notna(n_pts) and n_pts >= 1:
        sinopticos.append((float(n_pts), r["nhc_txt"]))

    # ---------- 4. Dinámica vertical ----------
    if pd.notna(r["omega700"]):
        if r["omega700"] > 0.05:
            sinopticos.append((PESOS["ascenso_700"],
                "**Ascenso en 700 hPa:** forzamiento dinámico activo"))
        elif r["omega700"] < -0.08:
            negativos.append((PESOS["subsidencia"],
                "**Subsidencia marcada:** el aire baja y seca la columna"))

    # ---------- 5. Termodinámica (grupo correlacionado) ----------
    if pd.notna(r["cape"]):
        if r["cape"] >= UMBRALES["cape_alto"]:
            termo.append((PESOS["conveccion_cape"],
                f"**CAPE alto:** {r['cape']:.0f} J/kg, energía para tormentas severas"))
        elif r["cape"] >= UMBRALES["cape_moderado"]:
            termo.append((PESOS["conveccion_cape"] * 0.5,
                f"**CAPE moderado:** {r['cape']:.0f} J/kg"))

    if pd.notna(r["li"]) and r["li"] <= UMBRALES["li_inestable"]:
        termo.append((PESOS["lifted_index"],
            f"**Lifted Index {r['li']:.1f}:** atmósfera inestable"))

    if pd.notna(r["rh_profunda"]) and r["rh_profunda"] >= 65:
        termo.append((PESOS["humedad_profunda"],
            f"**Columna húmeda:** {r['rh_profunda']:.0f}% medio en 850-500 hPa"))

    if pd.notna(r["pw"]):
        if r["pw"] >= UMBRALES["pw_alto"]:
            termo.append((PESOS["agua_precipitable"],
                f"**Agua precipitable alta:** {r['pw']:.0f} kg/m²"))
        elif r["pw"] <= UMBRALES["pw_bajo"]:
            negativos.append((-PESOS["agua_precipitable"],
                f"**Columna seca:** {r['pw']:.0f} kg/m² de agua precipitable"))

    if (pd.notna(r["temperature_2m"]) and pd.notna(r["relative_humidity_2m"])
            and r["temperature_2m"] >= UMBRALES["temp_gatillo"]
            and r["relative_humidity_2m"] >= UMBRALES["hum_gatillo"]):
        termo.append((PESOS["gatillo_termico"],
            f"**Gatillo térmico:** {r['temperature_2m']:.1f} °C con "
            f"{r['relative_humidity_2m']:.0f}% de humedad"))

    h = int(r["hora_local"])
    if 14 <= h <= 21:
        termo.append((PESOS["ciclo_diurno"],
            "**Ventana convectiva de la tarde** (14:00-21:00)"))
    elif 8 <= h <= 11:
        negativos.append((-PESOS["ciclo_diurno"] * 0.5,
            "**Media mañana:** mínimo convectivo del día"))

    # ---------- 6. Supresores fuertes ----------
    if pd.notna(r["cin"]) and r["cin"] <= UMBRALES["cin_bloqueante"]:
        negativos.append((PESOS["cin_fuerte"],
            f"**Inhibición convectiva {r['cin']:.0f} J/kg:** tapa que frena el disparo"))

    if pd.notna(r["rh700"]) and r["rh700"] <= UMBRALES["rh700_seco"]:
        negativos.append((PESOS["aire_sahariano"],
            f"**Intrusión seca / polvo del Sahara:** RH700 en {r['rh700']:.0f}%. "
            "Mata la convección aunque haya calor"))

    # ---------- 7. Trayectoria: ¿la tormenta viene o pasa de largo? ----------
    vel = r.get("vel_tormenta", np.nan)
    peso_local = 1.0
    adv_pts = 0.0
    texto_adv = None
    if pd.notna(vel):
        # Flujo débil -> las tormentas nacen y mueren en el sitio (brisa marina).
        # Flujo fuerte -> lo local importa menos que lo que viene de lejos.
        peso_local = float(np.clip(1 - vel / VEL_TORMENTA_ORGANIZADA, 0.3, 1.0))
        desde = rumbo_a_texto(r.get("rumbo_desde", np.nan))
        senal = r.get("senal_adv", np.nan)
        # Con rayos observados para esta hora, la advección del MODELO pesa la mitad
        f_mod = 0.5 if bool(r.get("glm_ok", False)) else 1.0

        if pd.notna(senal) and senal >= 0.1:
            adv_pts = PESOS["tormenta_en_camino"] * senal * f_mod
            texto_adv = (f"🎯 **Lluvia en camino desde el {desde}:** "
                         f"{r['lluvia_arriba']:.1f} mm aguas arriba, llega en "
                         f"~{r['eta_h']:.0f} h a {vel:.0f} km/h")
            sinopticos.append((adv_pts, texto_adv))
        elif vel >= 15 and pd.notna(senal):
            lado = r.get("lluvia_lado", 0.0) or 0.0
            factor = 1.0 if lado >= 2.0 else 0.5
            adv_pts = PESOS["flujo_aleja"] * factor * f_mod
            if lado >= 2.0:
                texto_adv = (f"↗️ **Tormenta cercana que NO viene:** hay "
                             f"{lado:.0f} mm en el anillo de 100 km, pero el flujo "
                             f"del {desde} a {vel:.0f} km/h la desvía")
            else:
                texto_adv = (f"💨 **Aguas arriba despejado:** flujo del {desde} "
                             f"a {vel:.0f} km/h sin lluvia en camino")
            negativos.append((adv_pts, texto_adv))

    # ---------- 7b. Rayos observados (GOES-19 GLM), solo 0-3 h ----------
    g_pts = r.get("glm_pts", np.nan)
    if pd.notna(g_pts) and g_pts != 0:
        (sinopticos if g_pts > 0 else negativos).append((g_pts, r["glm_txt"]))

    # ---------- 8. Combinación en log-odds ----------
    termo_orden = sorted((p for p, _ in termo), reverse=True)
    termo_pts = (termo_orden[0] + 0.3 * sum(termo_orden[1:])) if termo_orden else 0.0
    termo_pts *= FACTOR_TERMO * peso_local

    total_pts = (sum(p for p, _ in sinopticos) + termo_pts
                 + sum(p for p, _ in negativos))

    base = r["prob_base"] if pd.notna(r["prob_base"]) else 20.0
    base_c = float(np.clip(base, 5, 90)) / 100
    prob = _sigmoide(_logit(base_c) + total_pts / PUNTOS_POR_LOGIT) * 100
    prob = float(np.clip(prob, PROB_MIN, PROB_MAX))

    fmt = lambda p, t: f"{t} `({p:+.0f})`"
    detonantes = [fmt(p, t) for p, t in sinopticos]
    if termo:
        detonantes += [fmt(p, t) for p, t in termo]
        detonantes.append(
            f"_Grupo termodinámico: aporta {termo_pts:+.0f} pts efectivos "
            f"(se solapan entre sí; peso local {peso_local:.0%} por el flujo)_")
    supresores = [fmt(p, t) for p, t in negativos]

    return {
        "prob_base": base,
        "ajuste": prob - base,
        "prob_calibrada": prob,
        "detonantes": detonantes,
        "supresores": supresores,
        "confianza_onda": conf_onda,
        "peso_local": peso_local,
    }


def nivel_alerta(prob: float, cape: float | None) -> tuple[str, str, str]:
    if prob >= 75 or (prob >= 60 and (cape or 0) >= UMBRALES["cape_alto"]):
        return ("ROJA", "error",
                "Alta probabilidad de aguaceros con tormenta eléctrica y vendavales.")
    if prob >= 50:
        return ("NARANJA", "warning",
                "Sistema organizado en aproximación. Lluvias fuertes probables.")
    if prob >= 30:
        return ("AMARILLA", "warning",
                "Condiciones inestables. Chubascos aislados a dispersos.")
    return ("VERDE", "info", "Tiempo seco o nubosidad dispersa sin desarrollo.")


# ==============================================================================
# 5. INTERFAZ
# ==============================================================================

st.title("⛈️ Caribe Weather Analytics v2")
st.caption("Diagnóstico sinóptico y pronóstico calibrado para Barranquilla "
           "y el Caribe colombiano · GFS + ECMWF vía Open-Meteo")

try:
    crudo, avisos = cargar_datos()
    for a in avisos:
        st.warning(a)

    d = calcular_indices(crudo)

    # --- Anillo de vigilancia y advección aguas arriba ----------------------
    try:
        anillo = cargar_anillo()
        d = advectar(d, anillo)
    except Exception as e:
        anillo = None
        st.warning(f"No se pudo cargar el anillo de vigilancia ({e}). "
                   "La probabilidad se calcula sin trayectoria de tormentas.")
        for c in ("senal_adv", "lluvia_arriba", "lluvia_lado", "eta_h"):
            d[c] = np.nan

    # --- Índice de la hora actual (NO iloc[0]: eso es medianoche) -----------
    ahora_exacto = pd.Timestamp(datetime.now(ZONA)).tz_localize(None)
    ahora = ahora_exacto.floor("h")
    idx_ahora = int((d["tiempo"] - ahora).abs().idxmin())

    # --- Análisis oficial del NHC: ondas tropicales y bajas -----------------
    d["nhc_pts"], d["nhc_txt"], d["nhc_ok"] = np.nan, None, False
    nhc, pasos_nhc = None, []
    try:
        nhc = cargar_nhc()
        pasos_nhc = paso_ondas(nhc)
        valida_nhc = (nhc["emision_utc"] - pd.Timedelta(hours=NHC_RETRASO_ANALISIS_H)
                      + UTC_A_COL) if nhc.get("emision_utc") is not None else None
        # mostrar solo ondas que todavía importan (desde hace 24 h en adelante)
        pasos_nhc = [o for o in pasos_nhc
                     if (o["paso"] - ahora_exacto).total_seconds() > -24 * 3600]
        for k in range(len(d)):
            pts, txt = puntos_nhc(d.at[k, "tiempo"], pasos_nhc, nhc["bajas"], valida_nhc)
            d.at[k, "nhc_pts"], d.at[k, "nhc_txt"] = pts, txt
        d["nhc_ok"] = True
    except Exception as e:
        st.caption(f"⚠️ Análisis del NHC no disponible ({e}); se usa solo el "
                   "detector de ondas del modelo.")

    # --- Rayos observados GOES-19 GLM: nowcasting 0-3 h ----------------------
    d["glm_pts"], d["glm_txt"], d["glm_ok"] = np.nan, None, False
    d_sin_rayos = d.copy()
    rayos, info_glm, celdas, n_enc = None, None, [], 0
    try:
        rayos, info_glm = cargar_rayos(ahora_exacto.floor("4min").isoformat())
        r_now = d.iloc[idx_ahora]
        celdas = seguir_celdas(rayos, float(r_now.get("u_dir", 0) or 0),
                               float(r_now.get("v_dir", 0) or 0), ahora_exacto)
        n_enc = rayos_encima(rayos)
        if info_glm["latencia_min"] > 20:      # datos viejos: no usarlos
            raise RuntimeError(f"datos con {info_glm['latencia_min']:.0f} min de retraso")
        for k in range(4):
            if idx_ahora + k < len(d):
                pts, txt = puntos_nowcast(celdas, True, k, len(rayos), n_enc)
                d.at[idx_ahora + k, "glm_pts"] = pts
                d.at[idx_ahora + k, "glm_txt"] = txt
                d.at[idx_ahora + k, "glm_ok"] = True
    except Exception as e:
        info_glm, celdas, n_enc = None, [], 0
        d["glm_pts"], d["glm_txt"], d["glm_ok"] = np.nan, None, False
        st.warning(f"No se pudieron cargar los rayos del GOES-19 ({e}). "
                   "La probabilidad se calcula solo con modelos.")

    # --- Ventana de trabajo: próximas 48 h ---------------------------------
    fin = min(idx_ahora + 48, len(d) - 1)
    futuro = d.iloc[idx_ahora:fin].reset_index(drop=True)

    # --- Evaluación hora a hora --------------------------------------------
    evaluaciones = [evaluar_hora(d, i) for i in range(idx_ahora, fin)]
    futuro["prob_calibrada"] = [e["prob_calibrada"] for e in evaluaciones]
    futuro["ajuste"] = [e["ajuste"] for e in evaluaciones]
    futuro["confianza_onda"] = [e["confianza_onda"] for e in evaluaciones]

    actual = evaluaciones[0]
    r0 = d.iloc[idx_ahora]

    # ---------------------------------------------------------------- MÉTRICAS
    st.subheader("📌 Estado actual y pronóstico calibrado")
    st.caption(f"Hora de referencia: {d['tiempo'].iloc[idx_ahora]:%A %d/%m %H:%M} "
               f"(hora de Colombia)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Probabilidad calibrada", f"{actual['prob_calibrada']:.0f}%",
              f"{actual['ajuste']:+.0f} pts vs modelo crudo")
    c2.metric("Presión (nivel del mar)", f"{r0['pres']:.1f} hPa",
              f"{r0['pres_tend_3h_cruda']:+.1f} hPa/3 h "
              f"({r0['pres_tend_3h']:+.1f} sin marea)"
              if pd.notna(r0["pres_tend_3h"]) else None,
              delta_color="inverse")
    c3.metric("CAPE", f"{r0['cape']:.0f} J/kg" if pd.notna(r0["cape"]) else "s/d",
              f"LI {r0['li']:.1f}" if pd.notna(r0["li"]) else None)
    c4.metric("Humedad capa media (700 hPa)",
              f"{r0['rh700']:.0f}%" if pd.notna(r0["rh700"]) else "s/d",
              f"{r0['rh700_tend_12h']:+.0f} pts / 12 h"
              if pd.notna(r0["rh700_tend_12h"]) else None)
    c5.metric("Temperatura", f"{r0['temperature_2m']:.1f} °C",
              f"HR {r0['relative_humidity_2m']:.0f}%")

    nivel, estilo, mensaje = nivel_alerta(actual["prob_calibrada"], r0["cape"])
    getattr(st, estilo)(f"**ALERTA {nivel}:** {mensaje}")

    # Pico de las próximas 12 h: lo que de verdad interesa planificar
    prox12 = futuro.head(12)
    if not prox12.empty:
        j = int(prox12["prob_calibrada"].idxmax())
        st.caption(
            f"🕐 Pico esperado en las próximas 12 h: "
            f"**{prox12['prob_calibrada'].iloc[j]:.0f}%** hacia las "
            f"{prox12['tiempo'].iloc[j]:%H:%M}."
        )

    # Dato para la bitácora: máximo de la tarde (14:00-21:00) de hoy
    tarde = futuro[(futuro["tiempo"].dt.date == ahora.date())
                   & futuro["tiempo"].dt.hour.between(14, 21)]
    if not tarde.empty:
        st.info(f"📓 **Para tu bitácora:** probabilidad máxima de esta tarde "
                f"(14:00-21:00) = **{tarde['prob_calibrada'].max():.0f}%**")

    st.divider()

    # -------------------------------------------- TRAYECTORIA DE TORMENTAS
    st.subheader("🧭 Trayectoria de tormentas: ¿viene o pasa de largo?")
    if anillo is not None and pd.notna(r0.get("vel_tormenta", np.nan)):
    
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Las tormentas vienen del",
                  rumbo_a_texto(r0["rumbo_desde"]),
                  f"{r0['rumbo_desde']:.0f}°", delta_color="off")
        t2.metric("Velocidad de desplazamiento", f"{r0['vel_tormenta']:.0f} km/h",
                  "flujo 850-500 hPa", delta_color="off")
        prox6 = futuro.head(6)
        hay_camino = (prox6["senal_adv"] >= 0.1).any()
        t3.metric("Lluvia en camino (6 h)", "Sí" if hay_camino else "No",
                  f"máx. {prox6['lluvia_arriba'].max():.1f} mm aguas arriba",
                  delta_color="off")
        tray = trayectoria_reciente(anillo, d["tiempo"].iloc[idx_ahora])
        if tray["valida"]:
            t4.metric("Núcleo de lluvia reciente",
                      f"{tray['dist_actual_km']:.0f} km al "
                      f"{rumbo_a_texto(tray['rumbo_actual'])}",
                      "se acerca" if tray["se_acerca"] else "no se acerca",
                      delta_color="inverse" if tray["se_acerca"] else "normal")
        else:
            t4.metric("Núcleo de lluvia reciente", "sin núcleos",
                      "últimas 6 h", delta_color="off")

        # Mapa polar: lluvia de la última hora en el anillo + flecha del flujo
        ahora_t = d["tiempo"].iloc[idx_ahora]
        ult = (anillo[(anillo["tiempo"] > ahora_t - pd.Timedelta(hours=2))
                      & (anillo["tiempo"] <= ahora_t)]
               .groupby("pid")["precipitation"].max().reindex(ANILLO["pid"])
               .fillna(0).values)
        figp = go.Figure()
        figp.add_trace(go.Scatterpolar(
            r=ANILLO["radio"], theta=ANILLO["rumbo"], mode="markers",
            marker=dict(size=8 + np.minimum(ult, 15) * 2.2, color=ult,
                        colorscale="Blues", cmin=0, cmax=8, showscale=True,
                        colorbar=dict(title="mm (2 h)"),
                        line=dict(width=0.5, color="gray")),
            hovertemplate="%{r:.0f} km al %{theta:.0f}°<br>%{marker.color:.1f} mm"
                          "<extra></extra>",
            name="Lluvia en el anillo"))
        largo = float(np.clip(r0["vel_tormenta"] * 3, 15, 100))
        figp.add_trace(go.Scatterpolar(
            r=[largo, 0], theta=[r0["rumbo_desde"]] * 2, mode="lines+markers",
            marker=dict(size=[12, 4], color="#d62728"),
            line=dict(color="#d62728", width=3),
            name=f"Trayectoria 3 h ({r0['vel_tormenta']:.0f} km/h)"))
        if tray["valida"]:
            c = tray["centroides"]
            rr = np.hypot(c["este"], c["norte"])
            th = (np.rad2deg(np.arctan2(c["este"], c["norte"])) + 360) % 360
            figp.add_trace(go.Scatterpolar(
                r=rr, theta=th, mode="lines+markers",
                line=dict(color="orange", width=2, dash="dot"),
                name="Núcleo de lluvia (últimas 6 h)"))
        figp.update_layout(
            polar=dict(angularaxis=dict(rotation=90, direction="clockwise",
                                        tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                        ticktext=["N", "NE", "E", "SE", "S", "SO", "O", "NO"]),
                       radialaxis=dict(range=[0, 110], ticksuffix=" km")),
            legend=dict(orientation="h", y=-0.1), height=460,
            margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(figp, use_container_width=True)
        st.caption(
            "Barranquilla está en el centro. La flecha roja es el camino que "
            "recorrerá una tormenta en 3 h según el viento medio entre 1,5 y "
            "5,5 km de altura: solo la lluvia que está sobre esa línea llega a "
            "la ciudad. La línea naranja es cómo se movió el núcleo de lluvia "
            "en las últimas 6 h. Ojo: son datos del modelo GFS, no del radar.")
    else:
        st.info("Sin datos de flujo director o del anillo de vigilancia.")

    st.divider()

    # ------------------------------------------------- ANÁLISIS DEL NHC
    st.subheader("🌊 Ondas tropicales según el Centro Nacional de Huracanes")
    if nhc is not None:
        if nhc.get("emision_utc") is not None:
            st.caption(f"Discusión del NHC emitida "
                       f"{nhc['emision_utc'] + UTC_A_COL:%d/%m %H:%M} (hora de Colombia). "
                       "Se actualiza 4 veces al día.")
        if pasos_nhc:
            filas = []
            for o in pasos_nhc:
                h = (o["paso"] - ahora_exacto).total_seconds() / 3600
                filas.append({
                    "Eje": f"{o['lon_w']:.0f}°W",
                    "Velocidad": f"{o['kt']:.0f} kt ({o['kt']*1.852:.0f} km/h)",
                    "Paso por Barranquilla": f"{o['paso']:%a %d/%m %H:%M}",
                    "Cuándo": (f"en {h:.0f} h" if h > 0 else f"hace {-h:.0f} h"),
                    "Actividad": {1.0: "alta", 0.75: "moderada"}.get(o["actividad"], "débil"),
                })
            st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
            st.caption("La lluvia asociada suele concentrarse entre 12 h antes y "
                       "18 h después del paso del eje, con el máximo unas 3 h "
                       "después. Solo se listan ondas que alcanzan la latitud de "
                       "Barranquilla.")
            with st.expander("Texto original del NHC"):
                for o in pasos_nhc:
                    st.markdown(f"> {o['texto']}")
        else:
            st.info("El NHC no reporta ondas tropicales que vayan a pasar por "
                    "Barranquilla en los próximos días.")
        bajas_cerca = [b for b in nhc["bajas"]
                       if abs(b["lat"] - LAT_BQ) < 6 and abs(-b["lon_w"] - LON_BQ) < 8]
        for b in bajas_cerca:
            st.caption(f"📉 Baja de {b['mb']:.0f} hPa analizada en "
                       f"{b['lat']:.1f}°N {b['lon_w']:.1f}°W.")
    else:
        st.info("Análisis del NHC no disponible en este momento.")

    st.divider()

    # ------------------------------------------------- RAYOS OBSERVADOS GLM
    st.subheader("⚡ Rayos observados (GOES-19): lo que está pasando ahora")
    if info_glm is not None:
        st.caption(f"Último dato: {info_glm['ultimo']:%H:%M} "
                   f"(hace {info_glm['latencia_min']:.0f} min) · "
                   f"{info_glm['archivos']} archivos de la última hora · "
                   f"radio vigilado {GLM_RADIO_KM} km")
        g1, g2, g3 = st.columns(3)
        total = 0 if rayos.empty else len(rayos) * GLM_PASO
        g1.metric("Rayos en la última hora", f"~{total}")
        if celdas:
            c0 = celdas[0]
            g2.metric("Tormenta activa más cercana",
                      f"{c0['dist']:.0f} km al {rumbo_a_texto(c0['rumbo'])}",
                      f"va hacia el {rumbo_a_texto(c0['rumbo_hacia'])} "
                      f"a {c0['vel']:.0f} km/h", delta_color="off")
            vienen = [c for c in celdas if c["viene"] and c["fuente"] != "modelo"]
            if n_enc >= 3:
                g3.metric("¿Viene hacia la ciudad?", "Ya está encima",
                          f"~{n_enc} rayos a <{GLM_ENCIMA_KM} km", delta_color="inverse")
            elif vienen:
                v0 = min(vienen, key=lambda c: c["eta_h"])
                g3.metric("¿Viene hacia la ciudad?", "Sí",
                          f"en ~{v0['eta_h']*60:.0f} min ({v0['fuente']})",
                          delta_color="inverse")
            else:
                g3.metric("¿Viene hacia la ciudad?", "No",
                          "las celdas pasan de largo", delta_color="normal")
        else:
            g2.metric("Tormenta activa más cercana", "ninguna")
            g3.metric("¿Viene hacia la ciudad?", "No", "sin rayos", delta_color="normal")

        if not rayos.empty:
            edad = (ahora_exacto - rayos["tiempo"]).dt.total_seconds() / 60
            figr = go.Figure()
            figr.add_trace(go.Scatterpolar(
                r=rayos["dist"], theta=rayos["rumbo"], mode="markers",
                marker=dict(size=5, color=edad, colorscale="YlOrRd_r", cmin=0,
                            cmax=GLM_MINUTOS, showscale=True,
                            colorbar=dict(title="min atrás")),
                hovertemplate="%{r:.0f} km<extra></extra>", name="Rayos"))
            for c in celdas[:6]:
                fin_e = c["este"] + c["u"]
                fin_n = c["norte"] + c["v"]
                figr.add_trace(go.Scatterpolar(
                    r=[c["dist"], float(np.hypot(fin_e, fin_n))],
                    theta=[c["rumbo"],
                           float((np.rad2deg(np.arctan2(fin_e, fin_n)) + 360) % 360)],
                    mode="lines+markers",
                    line=dict(color=("#d62728" if c["viene"] and c["fuente"] == "observada"
                                     else "#ff9f40" if c["viene"] else "#7f7f7f"),
                              dash="solid" if c["fuente"] == "observada" else "dot",
                              width=3),
                    marker=dict(size=[10, 4]),
                    name=f"{'Viene' if c['viene'] else 'Pasa'} · {c['dist']:.0f} km"))
            figr.update_layout(
                polar=dict(angularaxis=dict(rotation=90, direction="clockwise",
                           tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                           ticktext=["N", "NE", "E", "SE", "S", "SO", "O", "NO"]),
                           radialaxis=dict(range=[0, GLM_RADIO_KM], ticksuffix=" km")),
                legend=dict(orientation="h", y=-0.1), height=460,
                margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(figr, use_container_width=True)
            st.caption("Barranquilla en el centro. Los puntos son rayos detectados "
                       "por el satélite (amarillo = recientes). Cada línea es lo que "
                       "recorrerá esa tormenta en 1 h: continua si su trayectoria se "
                       "observó de forma consistente, punteada si es estimada. "
                       "Roja/naranja si pasa sobre la ciudad.")

        # Comparación directa: ¿cuánto cambian los rayos el pronóstico?
        ev_sin = [evaluar_hora(d_sin_rayos, idx_ahora + k)["prob_calibrada"]
                  for k in range(4) if idx_ahora + k < len(d)]
        comp = pd.DataFrame({
            "Hora": [f"{t:%H:%M}" for t in futuro["tiempo"].head(len(ev_sin))],
            "GFS crudo": [f"{p:.0f}%" for p in futuro["prob_base"].head(len(ev_sin))],
            "Sin rayos": [f"{p:.0f}%" for p in ev_sin],
            "Con rayos": [f"{p:.0f}%" for p in futuro["prob_calibrada"].head(len(ev_sin))],
        })
        st.markdown("**Efecto de los rayos en las próximas horas**")
        st.dataframe(comp, hide_index=True, use_container_width=True)
    else:
        st.info("Datos del GOES-19 no disponibles en este momento.")

    st.divider()

    # ------------------------------------------------- DIAGNÓSTICO SINÓPTICO
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("⚡ Factores detonantes")
        if actual["detonantes"]:
            for t in actual["detonantes"]:
                st.markdown(f"- {t}")
        else:
            st.write("✅ Sin forzamientos significativos en este momento.")

        if actual["supresores"]:
            st.subheader("🛑 Factores supresores")
            for t in actual["supresores"]:
                st.markdown(f"- {t}")

    with col_b:
        st.subheader("🌀 Seguimiento de ondas del Este")
        pico_onda = futuro["confianza_onda"].max()
        if pico_onda > 0:
            k = int(futuro["confianza_onda"].idxmax())
            st.success(
                f"Firma de onda tropical detectada con "
                f"**{int(pico_onda*100)}%** de coincidencia, con eje pasando "
                f"cerca de **{futuro['tiempo'].iloc[k]:%A %d/%m a las %H:%M}**. "
                "El máximo de lluvia suele ocurrir entre 6 y 12 h antes del eje "
                "y justo durante su paso."
            )
        else:
            st.info("No se detecta firma de onda tropical en las próximas 48 h. "
                    "Las lluvias, si ocurren, serían de origen térmico o de brisa.")

        if pd.notna(r0["desacuerdo_pres"]):
            nivel_ac = ("alto" if r0["desacuerdo_pres"] > 2 else
                        "moderado" if r0["desacuerdo_pres"] > 1 else "bueno")
            st.caption(
                f"Consenso GFS ↔ ECMWF en presión: **{nivel_ac}** "
                f"(diferencia de {r0['desacuerdo_pres']:.1f} hPa). "
                "Un desacuerdo alto significa menos confianza en el pronóstico."
            )

    st.divider()

    # ----------------------------------------------------------- TABLA HORARIA
    st.subheader("🧭 Parámetros sinópticos hora a hora")
    tabla = futuro[[
        "tiempo", "prob_base", "prob_calibrada", "pres", "pres_tend_3h",
        "gph500_anom", "gph700_anom", "rh700", "cape", "li", "omega700",
        "vel_tormenta", "lluvia_arriba",
    ]].copy()
    tabla.columns = [
        "Hora", "Prob. modelo (%)", "Prob. calibrada (%)", "Presión (hPa)",
        "Tend. 3h", "Anom. 500 hPa", "Anom. 700 hPa", "RH 700 (%)",
        "CAPE", "LI", "Omega 700", "Vel. tormentas (km/h)", "Lluvia aguas arriba (mm)",
    ]
    st.dataframe(
        tabla.style.format({c: "{:.1f}" for c in tabla.columns[1:]}, na_rep="s/d")
             .background_gradient(subset=["Prob. calibrada (%)"], cmap="Blues"),
        use_container_width=True, hide_index=True, height=300,
    )

    st.divider()

    # ---------------------------------------------------------------- GRÁFICOS
    st.subheader("📈 Proyección para las próximas 48 horas")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=futuro["tiempo"], y=futuro["lluvia"],
                         name="Precipitación (mm)", yaxis="y2",
                         opacity=0.35, marker_color="cyan"))
    fig.add_trace(go.Scatter(x=futuro["tiempo"], y=futuro["prob_base"],
                             name="Modelo crudo (%)",
                             line=dict(color="gray", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=futuro["tiempo"], y=futuro["prob_calibrada"],
                             name="Calibrado Caribe (%)",
                             line=dict(color="#1f77b4", width=3)))
    fig.add_hrect(y0=75, y1=100, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="orange", opacity=0.08, line_width=0)
    fig.update_layout(
        xaxis_title="Hora local",
        yaxis=dict(title="Probabilidad (%)", range=[0, 100]),
        yaxis2=dict(title="Lluvia (mm)", overlaying="y", side="right", range=[0, 20]),
        legend=dict(x=0, y=1.15, orientation="h"),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Perfil sinóptico: presión y anomalía de altura, incluyendo el pasado real
    ventana_hist = d.iloc[max(0, idx_ahora - 36):fin]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ventana_hist["tiempo"], y=ventana_hist["pres"],
                              name="Presión nivel del mar (hPa)",
                              line=dict(color="#d62728", width=2)))
    fig2.add_trace(go.Scatter(x=ventana_hist["tiempo"], y=ventana_hist["pres_base"],
                              name="Media 5 días",
                              line=dict(color="#d62728", width=1, dash="dash")))
    fig2.add_trace(go.Scatter(x=ventana_hist["tiempo"], y=ventana_hist["gph700_anom"],
                              name="Anomalía 700 hPa (mgp)", yaxis="y2",
                              line=dict(color="#2ca02c", width=2)))
    fig2.add_vline(x=d["tiempo"].iloc[idx_ahora], line_dash="dot",
                   line_color="white", annotation_text="ahora")
    fig2.update_layout(
        title="Firma de paso de onda / vaguada (36 h atrás → 48 h adelante)",
        yaxis=dict(title="Presión (hPa)"),
        yaxis2=dict(title="Anomalía geopotencial (mgp)",
                    overlaying="y", side="right"),
        legend=dict(x=0, y=1.15, orientation="h"),
        margin=dict(l=0, r=0, t=60, b=0), height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------ MAPAS
    st.subheader("🗺️ Comparativa sinóptica: modelo vs realidad")

    def windy(overlay: str, level: str = "surface", producto: str = "ecmwf",
              presion: str = "true", zoom: int = 5) -> str:
        return f"""
        <iframe width="100%" height="430" frameborder="0"
        src="https://embed.windy.com/embed.html?type=map&location=coordinates
&metricRain=mm&metricTemp=%C2%B0C&metricWind=km/h&zoom={zoom}&overlay={overlay}
&product={producto}&level={level}&lat={LAT}&lon={LON}&detailLat={LAT}
&detailLon={LON}&marker=true&pressure={presion}&message=true"></iframe>
        """.replace("\n", "")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**1. Lluvia e isobaras (ECMWF)**")
        st.caption("Busca la 'L' con isobaras cerradas y las vaguadas alargadas.")
        components.html(windy("rain"), height=440)
    with m2:
        st.markdown("**2. Satélite infrarrojo en vivo**")
        st.caption("Rojo = topes muy fríos. OJO: puede ser solo el yunque (nube alta que el viento de altura extiende lejos del núcleo). Donde llueve es donde se concentran los rayos.")
        components.html(windy("satellite", producto="satellite", presion="false"),
                        height=440)

    m3, m4 = st.columns(2)
    with m3:
        st.markdown("**3. Viento en 700 hPa — el nivel de las ondas del Este**")
        st.caption("El eje de la onda se ve como una 'V' invertida que avanza "
                   "desde Venezuela hacia el oeste.")
        components.html(windy("wind", level="700h", presion="false"), height=440)
    with m4:
        st.markdown("**4. Humedad relativa en 700 hPa**")
        st.caption("Lenguas secas (naranja/marrón) = aire sahariano que apaga "
                   "la convección. Verde/azul = combustible disponible.")
        components.html(windy("rh", level="700h", presion="false"), height=440)

    st.divider()

    # ------------------------------------------------- GUÍA Y VERIFICACIÓN
    g1, g2 = st.columns(2)
    with g1:
        with st.expander("📖 Cómo leer los sistemas del Caribe colombiano",
                         expanded=False):
            st.markdown("""
* **Baja presión (L):** en el trópico las variaciones son pequeñas. Una caída
  de 2-3 hPa bajo la media ya es un sistema importante. Por eso esta versión
  mide la **anomalía** y la **tendencia**, no solo el valor absoluto.
* **Vaguada:** eje alargado de baja presión sin centro cerrado. Se detecta
  mejor con la **anomalía de altura geopotencial** en 500 y 700 hPa que con la
  presión de superficie.
* **Onda tropical:** perturbación que viaja de este a oeste a 15-25 km/h en la
  capa de 700 hPa. Su firma es: presión que baja y vuelve a subir, viento que
  rola del ENE al ESE y regresa al NE, y un salto de humedad en capa media.
  La lluvia más fuerte cae **delante y sobre el eje**.
* **Polvo del Sahara:** entre junio y agosto llega aire muy seco en 700 hPa.
  Aunque haga 34 °C y el mar esté caliente, la convección se apaga. Por eso
  aquí es un supresor con peso alto.
* **Brisa marina:** en Barranquilla la convergencia entre el alisio y la brisa
  del mar y del río dispara aguaceros entre las 15:00 y las 20:00.
            """)
    with g2:
        with st.expander("🎯 Cómo calibrar los pesos con tus propios datos",
                         expanded=False):
            st.markdown("""
Los valores del diccionario `PESOS` son un punto de partida razonable, no una
verdad física. Para afinarlos:

1. Guarda cada día un registro con la probabilidad calibrada, los detonantes
   activos y si de verdad llovió en tu barrio (y cuántos mm).
2. Después de 30-60 días, revisa qué detonantes aparecían cuando llovió y
   cuáles aparecían en falsas alarmas.
3. Sube el peso de los que aciertan y baja el de los que no. Si acumulas
   suficientes casos, puedes reemplazar el sistema de puntos por una
   regresión logística entrenada con tu propio historial.
4. Compara siempre contra el radar y los boletines del IDEAM: este sistema es
   una ayuda de decisión, no un pronóstico oficial.
            """)

except requests.exceptions.RequestException as e:
    st.error(f"No se pudo contactar la API de Open-Meteo: {e}")
except Exception as e:
    st.error(f"Error técnico al procesar los datos: {type(e).__name__}: {e}")
    st.exception(e)

from sinoptico_regional import render_pronostico_extendido
st.divider()
render_pronostico_extendido()

from visor_precipitacion import render_visor_precipitacion
st.divider()
render_visor_precipitacion()
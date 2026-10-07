"""
================================================================================
 TRAYECTORIA DE TORMENTAS  |  Flujo director y advección hacia Barranquilla
================================================================================
 Responde la pregunta que el modelo puntual no podía responder:
 "hay tormentas cerca, pero ¿VIENEN hacia la ciudad o pasan de largo?"

 1. FLUJO DIRECTOR: las celdas convectivas se mueven aproximadamente con el
    viento medio de la capa 850-500 hPa (≈1,5 a 5,5 km). Se pondera
    850 (25%), 700 (40%), 500 (35%).

 2. ANILLO DE VIGILANCIA: se descargan 48 puntos alrededor de Barranquilla
    (16 rumbos × radios de 30, 60 y 100 km) con lluvia y CAPE hora a hora.

 3. ADVECCIÓN: para cada hora t se calcula dónde estaba, 1-3 h antes, el aire
    que llegará a la ciudad (punto "aguas arriba"). Si había lluvia allí,
    viene en camino. Si la lluvia está en otro sector, pasa de largo.

 4. TRAYECTORIA RECIENTE: el centroide de la lluvia en el anillo durante las
    últimas horas da un vector de movimiento que se compara con el flujo.

 LIMITACIÓN HONESTA: los datos son del modelo (GFS vía Open-Meteo), incluidas
 las horas pasadas. No es radar ni satélite. Corrige la mala lectura de
 "tormenta cerca = lluvia segura", pero no reemplaza el nowcasting observado.
================================================================================
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st

LAT_BQ, LON_BQ = 10.968, -74.781
TZ = "America/Bogota"
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

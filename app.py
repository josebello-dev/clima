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

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from trayectoria_tormentas import (
    advectar, cargar_anillo, flujo_director, rumbo_a_texto, trayectoria_reciente,
)

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
    if conf_onda > 0:
        sinopticos.append((PESOS["onda_tropical"] * conf_onda,
            f"🌀 **Onda tropical probable** ({int(conf_onda*100)}% de firma): "
            + ", ".join(detalle_onda)))

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

        if pd.notna(senal) and senal >= 0.1:

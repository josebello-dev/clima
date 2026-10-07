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
            adv_pts = PESOS["tormenta_en_camino"] * senal
            texto_adv = (f"🎯 **Lluvia en camino desde el {desde}:** "
                         f"{r['lluvia_arriba']:.1f} mm aguas arriba, llega en "
                         f"~{r['eta_h']:.0f} h a {vel:.0f} km/h")
            sinopticos.append((adv_pts, texto_adv))
        elif vel >= 15 and pd.notna(senal):
            lado = r.get("lluvia_lado", 0.0) or 0.0
            factor = 1.0 if lado >= 2.0 else 0.5
            adv_pts = PESOS["flujo_aleja"] * factor
            if lado >= 2.0:
                texto_adv = (f"↗️ **Tormenta cercana que NO viene:** hay "
                             f"{lado:.0f} mm en el anillo de 100 km, pero el flujo "
                             f"del {desde} a {vel:.0f} km/h la desvía")
            else:
                texto_adv = (f"💨 **Aguas arriba despejado:** flujo del {desde} "
                             f"a {vel:.0f} km/h sin lluvia en camino")
            negativos.append((adv_pts, texto_adv))

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
    ahora = pd.Timestamp(datetime.now(ZONA)).tz_localize(None).floor("h")
    idx_ahora = int((d["tiempo"] - ahora).abs().idxmin())

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

    st.divider()

    # -------------------------------------------- TRAYECTORIA DE TORMENTAS
    st.subheader("🧭 Trayectoria de tormentas: ¿viene o pasa de largo?")
    if anillo is not None and pd.notna(r0.get("vel_tormenta", np.nan)):
        from trayectoria_tormentas import ANILLO

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
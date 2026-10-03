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
PESOS = {
    "baja_presion_absoluta": 18,
    "anomalia_presion": 14,
    "caida_barometrica": 12,
    "vaguada_500": 16,
    "vaguada_700": 12,
    "onda_tropical": 22,
    "conveccion_cape": 16,
    "lifted_index": 12,
    "humedad_profunda": 12,
    "agua_precipitable": 10,
    "alisios_este": 6,
    "gatillo_termico": 10,
    "ascenso_700": 10,
    "ciclo_diurno": 8,
    # --- Supresores (restan) ---
    "aire_sahariano": -25,
    "cin_fuerte": -15,
    "subsidencia": -12,
    "alta_presion": -12,
}

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
        "past_days": 2,        # historial real -> tendencia barométrica verdadera
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
    p = col(d, "pressure_msl")
    d["pres"] = p

    # Media móvil de ~5 días centrada = "normal local" dinámica.
    d["pres_base"] = p.rolling(120, min_periods=24, center=True).mean()
    d["pres_anom"] = p - d["pres_base"]

    # Tendencia barométrica real de 3 h y 12 h (con past_days ya hay pasado real)
    d["pres_tend_3h"] = p.diff(3)
    d["pres_tend_12h"] = p.diff(12)

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


def evaluar_hora(d: pd.DataFrame, i: int) -> dict:
    """Devuelve ajuste total, detonantes y supresores para la hora i."""
    r = d.iloc[i]
    ajuste = 0.0
    detonantes, supresores = [], []

    def suma(clave, texto, lista=detonantes):
        nonlocal ajuste
        ajuste += PESOS[clave]
        lista.append(f"{texto} `({PESOS[clave]:+d})`")

    # ---------- 1. Sistemas de baja presión ----------
    if pd.notna(r["pres"]):
        if r["pres"] <= UMBRALES["pres_baja_abs"]:
            suma("baja_presion_absoluta",
                 f"**Baja presión activa (L):** {r['pres']:.1f} hPa a nivel del mar")
        elif r["pres"] >= UMBRALES["pres_alta_abs"]:
            suma("alta_presion",
                 f"**Dorsal anticiclónica:** {r['pres']:.1f} hPa, aire estable",
                 supresores)

    if pd.notna(r["pres_anom"]) and r["pres_anom"] <= UMBRALES["anom_pres_baja"]:
        suma("anomalia_presion",
             f"**Anomalía negativa de presión:** {r['pres_anom']:+.1f} hPa "
             "bajo la media de 5 días")

    if pd.notna(r["pres_tend_3h"]) and r["pres_tend_3h"] <= UMBRALES["caida_3h"]:
        suma("caida_barometrica",
             f"**Caída barométrica:** {r['pres_tend_3h']:+.1f} hPa en 3 h")

    # ---------- 2. Vaguadas en altura ----------
    if pd.notna(r["gph500_anom"]) and r["gph500_anom"] <= UMBRALES["gph500_anom"]:
        suma("vaguada_500",
             f"**Vaguada en 500 hPa:** {r['gph500_anom']:+.0f} mgp de anomalía")

    if pd.notna(r["gph700_anom"]) and r["gph700_anom"] <= UMBRALES["gph700_anom"]:
        suma("vaguada_700",
             f"**Vaguada en 700 hPa:** {r['gph700_anom']:+.0f} mgp de anomalía")

    # ---------- 3. Onda tropical ----------
    conf_onda, detalle_onda = detectar_onda_tropical(d, i)
    if conf_onda > 0:
        bonus = PESOS["onda_tropical"] * conf_onda
        ajuste += bonus
        detonantes.append(
            f"🌀 **Onda tropical probable** ({int(conf_onda*100)}% de firma): "
            + ", ".join(detalle_onda) + f" `({bonus:+.0f})`"
        )

    # ---------- 4. Termodinámica ----------
    if pd.notna(r["cape"]):
        if r["cape"] >= UMBRALES["cape_alto"]:
            suma("conveccion_cape",
                 f"**CAPE alto:** {r['cape']:.0f} J/kg, energía para tormentas severas")
        elif r["cape"] >= UMBRALES["cape_moderado"]:
            ajuste += PESOS["conveccion_cape"] * 0.5
            detonantes.append(
                f"**CAPE moderado:** {r['cape']:.0f} J/kg "
                f"`({PESOS['conveccion_cape']*0.5:+.0f})`")

    if pd.notna(r["li"]) and r["li"] <= UMBRALES["li_inestable"]:
        suma("lifted_index", f"**Lifted Index {r['li']:.1f}:** atmósfera inestable")

    if pd.notna(r["cin"]) and r["cin"] <= UMBRALES["cin_bloqueante"]:
        suma("cin_fuerte",
             f"**Inhibición convectiva {r['cin']:.0f} J/kg:** tapa que frena el disparo",
             supresores)

    # ---------- 5. Humedad ----------
    if pd.notna(r["rh_profunda"]) and r["rh_profunda"] >= 65:
        suma("humedad_profunda",
             f"**Columna húmeda:** {r['rh_profunda']:.0f}% medio en 850-500 hPa")

    if pd.notna(r["rh700"]) and r["rh700"] <= UMBRALES["rh700_seco"]:
        suma("aire_sahariano",
             f"**Intrusión seca / polvo del Sahara:** RH700 en {r['rh700']:.0f}%. "
             "Mata la convección aunque haya calor",
             supresores)

    if pd.notna(r["pw"]):
        if r["pw"] >= UMBRALES["pw_alto"]:
            suma("agua_precipitable",
                 f"**Agua precipitable alta:** {r['pw']:.0f} kg/m²")
        elif r["pw"] <= UMBRALES["pw_bajo"]:
            ajuste += PESOS["agua_precipitable"] * -1
            supresores.append(
                f"**Columna seca:** {r['pw']:.0f} kg/m² de agua precipitable "
                f"`({-PESOS['agua_precipitable']:+d})`")

    # ---------- 6. Dinámica vertical ----------
    if pd.notna(r["omega700"]):
        if r["omega700"] > 0.05:
            suma("ascenso_700", "**Ascenso en 700 hPa:** forzamiento dinámico activo")
        elif r["omega700"] < -0.08:
            suma("subsidencia",
                 "**Subsidencia marcada:** el aire baja y seca la columna", supresores)

    # ---------- 7. Flujo de alisios y gatillo térmico ----------
    if pd.notna(r["dir_sfc"]) and 45 <= r["dir_sfc"] <= 110:
        suma("alisios_este", "**Flujo de alisios del Este:** arrastre de nubosidad")

    if (pd.notna(r["temperature_2m"]) and pd.notna(r["relative_humidity_2m"])
            and r["temperature_2m"] >= UMBRALES["temp_gatillo"]
            and r["relative_humidity_2m"] >= UMBRALES["hum_gatillo"]):
        suma("gatillo_termico",
             f"**Gatillo térmico:** {r['temperature_2m']:.1f} °C con "
             f"{r['relative_humidity_2m']:.0f}% de humedad")

    # ---------- 8. Ciclo diurno (Barranquilla dispara en la tarde-noche) ----
    h = int(r["hora_local"])
    if 14 <= h <= 21:
        suma("ciclo_diurno", "**Ventana convectiva de la tarde** (14:00-21:00)")
    elif 8 <= h <= 11:
        ajuste += -PESOS["ciclo_diurno"] * 0.5
        supresores.append("**Media mañana:** mínimo convectivo del día "
                          f"`({-PESOS['ciclo_diurno']*0.5:+.0f})`")

    base = r["prob_base"] if pd.notna(r["prob_base"]) else 0.0
    prob = float(np.clip(base + ajuste, 0, 100))

    return {
        "prob_base": base,
        "ajuste": ajuste,
        "prob_calibrada": prob,
        "detonantes": detonantes,
        "supresores": supresores,
        "confianza_onda": conf_onda,
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
              f"{r0['pres_tend_3h']:+.1f} hPa / 3 h"
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
    ]].copy()
    tabla.columns = [
        "Hora", "Prob. modelo (%)", "Prob. calibrada (%)", "Presión (hPa)",
        "Tend. 3h", "Anom. 500 hPa", "Anom. 700 hPa", "RH 700 (%)",
        "CAPE", "LI", "Omega 700",
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
        st.caption("Rojo/fucsia/blanco = topes muy fríos = tormenta severa.")
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
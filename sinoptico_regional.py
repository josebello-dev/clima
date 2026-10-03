"""
================================================================================
 SINÓPTICO REGIONAL Y PRONÓSTICO EXTENDIDO  |  Caribe Colombiano
================================================================================
 Módulo complementario de Caribe Weather Analytics.

 ¿Qué hace y por qué?
 --------------------
 La app principal analiza UN punto (Barranquilla). Eso sirve para hoy y mañana,
 pero NO permite ver lo que vio la persona del comunicado: ella no miró un
 pronóstico puntual, miró un MAPA. Vio una onda tropical al este de las
 Antillas, una vaguada en niveles altos al norte y una baja cerca de Panamá, y
 calculó cuántos días tardarían esos sistemas en llegar.

 Este módulo replica ese razonamiento de forma automática:

   1. Descarga una MALLA de puntos sobre el Caribe (no uno solo).
   2. Calcula la vorticidad relativa en 700 hPa y arma un diagrama de
      HOVMÖLLER (longitud vs tiempo), que es la herramienta estándar para
      ver ondas del Este viajando hacia el oeste.
   3. Rastrea el eje de la onda y estima cuándo llega a Barranquilla.
   4. Detecta vaguadas en niveles altos (200/500 hPa) y bajas en superficie,
      y dice DÓNDE están.
   5. Usa ENSAMBLES (30-50 corridas del modelo con condiciones iniciales
      ligeramente distintas) para dar probabilidad y nivel de confianza en el
      rango de 3 a 10 días, que es donde un solo modelo determinístico falla.
   6. Genera recomendaciones prácticas por día.

 Integración en tu app principal (3 líneas):

     from sinoptico_regional import render_pronostico_extendido
     ...
     st.divider()
     render_pronostico_extendido()

 Requiere: streamlit pandas numpy plotly requests
================================================================================
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN DEL DOMINIO
# ==============================================================================

LAT_BQ, LON_BQ = 10.968, -74.781
TZ = "America/Bogota"

API_FORECAST = "https://api.open-meteo.com/v1/forecast"
API_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Malla: corredor por donde entran las ondas del Este al Caribe colombiano.
# De las Antillas Menores (-58°) hasta Centroamérica (-82°).
MALLA_LATS = [9.0, 12.5, 16.0]
MALLA_LONS = [-82.0, -79.0, -76.0, -73.0, -70.0, -67.0, -64.0, -61.0, -58.0]
LAT_RASTREO = 12.5        # banda latitudinal donde mejor se ven las ondas

# Velocidad típica de una onda del Este en el Caribe: 15-25 km/h hacia el oeste.
VEL_ONDA_MIN_KMH, VEL_ONDA_MAX_KMH = 10.0, 35.0

# Umbrales para clasificar el día
UMBRAL_LLUVIA_DIA = 1.0        # mm: se considera "día con lluvia"
UMBRAL_LLUVIA_FUERTE = 15.0    # mm: aguacero relevante
UMBRAL_ARROYOS = 25.0          # mm en 3 h: riesgo de arroyos en Barranquilla


# ==============================================================================
# 2. DESCARGA DE DATOS
# ==============================================================================

VARS_MALLA = [
    "pressure_msl",
    "precipitation",
    "cape",
    "relative_humidity_700hPa",
    "geopotential_height_700hPa",
    "geopotential_height_500hPa",
    "geopotential_height_200hPa",
    "wind_speed_700hPa",
    "wind_direction_700hPa",
]


@st.cache_data(ttl=3600, show_spinner="Descargando malla regional del Caribe…")
def cargar_malla(dias: int = 7) -> pd.DataFrame:
    """
    Open-Meteo acepta varias coordenadas en una sola petición (separadas por
    coma) y devuelve una lista de respuestas, una por punto. Se usa resolución
    de 3 horas para no traer un JSON gigante.
    """
    lats, lons = [], []
    for la in MALLA_LATS:
        for lo in MALLA_LONS:
            lats.append(la)
            lons.append(lo)

    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "hourly": ",".join(VARS_MALLA),
        "models": "gfs_seamless",
        "timezone": TZ,
        "forecast_days": dias,
        "temporal_resolution": "hourly_3",
    }
    r = requests.get(API_FORECAST, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):        # una sola coordenada
        data = [data]

    filas = []
    for punto in data:
        h = punto["hourly"]
        sub = pd.DataFrame({"tiempo": pd.to_datetime(h["time"])})
        sub["lat"] = round(float(punto["latitude"]), 2)
        sub["lon"] = round(float(punto["longitude"]), 2)
        for k, v in h.items():
            if k != "time":
                sub[k] = pd.to_numeric(pd.Series(v), errors="coerce")
        filas.append(sub)

    df = pd.concat(filas, ignore_index=True)

    # Open-Meteo devuelve el punto de malla más cercano, no el pedido.
    # Se reasignan a la longitud/latitud solicitada para que la rejilla quede regular.
    df["lon"] = df["lon"].apply(lambda x: min(MALLA_LONS, key=lambda c: abs(c - x)))
    df["lat"] = df["lat"].apply(lambda x: min(MALLA_LATS, key=lambda c: abs(c - x)))
    return df


@st.cache_data(ttl=3600, show_spinner="Descargando ensambles (30-50 miembros)…")
def cargar_ensamble(modelos: tuple = ("gfs_seamless", "ecmwf_ifs04"),
                    dias: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """
    Un ensamble corre el mismo modelo muchas veces con condiciones iniciales
    perturbadas. Si 40 de 50 miembros mojan Barranquilla el jueves, eso es una
    probabilidad real. Si solo 12 lo hacen, el pronóstico es dudoso.
    Devuelve un DataFrame largo: tiempo | miembro | precipitation.
    """
    piezas, avisos = [], []
    for modelo in modelos:
        try:
            params = {
                "latitude": LAT_BQ,
                "longitude": LON_BQ,
                "hourly": "precipitation",
                "models": modelo,
                "timezone": TZ,
                "forecast_days": dias,
            }
            r = requests.get(API_ENSEMBLE, params=params, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            tiempo = pd.to_datetime(h["time"])
            for clave, valores in h.items():
                if clave == "time" or not clave.startswith("precipitation"):
                    continue
                sufijo = clave.replace("precipitation", "").strip("_") or "control"
                piezas.append(pd.DataFrame({
                    "tiempo": tiempo,
                    "miembro": f"{modelo}:{sufijo}",
                    "precipitation": pd.to_numeric(pd.Series(valores), errors="coerce"),
                }))
        except Exception as e:
            avisos.append(f"Ensamble {modelo} no disponible ({e}).")

    if not piezas:
        return pd.DataFrame(columns=["tiempo", "miembro", "precipitation"]), avisos
    return pd.concat(piezas, ignore_index=True), avisos


# ==============================================================================
# 3. ANÁLISIS SINÓPTICO SOBRE LA MALLA
# ==============================================================================

def calcular_vorticidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vorticidad relativa aproximada en 700 hPa: ζ ≈ ∂v/∂x − ∂u/∂y.
    En el hemisferio norte, ζ positiva = giro ciclónico = eje de vaguada/onda.
    Es la firma objetiva de una onda del Este, mucho más confiable que mirar
    solo la presión (en el trópico la presión casi no se mueve).
    """
    d = df.copy()
    rad = np.deg2rad(d["wind_direction_700hPa"])
    spd = d["wind_speed_700hPa"] / 3.6                    # km/h -> m/s
    d["u700"] = -spd * np.sin(rad)                        # + = viento del oeste
    d["v700"] = -spd * np.cos(rad)                        # + = viento del sur

    salida = []
    for t, g in d.groupby("tiempo"):
        piv_v = g.pivot_table(index="lat", columns="lon", values="v700")
        piv_u = g.pivot_table(index="lat", columns="lon", values="u700")
        if piv_v.shape[0] < 2 or piv_v.shape[1] < 2:
            continue

        lats = piv_v.index.values
        lons = piv_v.columns.values
        # metros por grado
        dy = 111_320.0
        dx = 111_320.0 * np.cos(np.deg2rad(lats))[:, None]

        dv_dx = np.gradient(piv_v.values, lons, axis=1) / dx
        du_dy = np.gradient(piv_u.values, lats, axis=0) / dy
        zeta = dv_dx - du_dy

        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                salida.append({"tiempo": t, "lat": la, "lon": lo,
                               "zeta": zeta[i, j] * 1e5})   # unidades 10⁻⁵ s⁻¹
    z = pd.DataFrame(salida)
    return d.merge(z, on=["tiempo", "lat", "lon"], how="left")


def rastrear_onda(df: pd.DataFrame, lat_banda: float = LAT_RASTREO) -> dict:
    """
    Sigue el máximo de vorticidad ciclónica desplazándose hacia el oeste y
    estima cuándo cruza la longitud de Barranquilla.
    """
    banda = df[df["lat"] == lat_banda].pivot_table(
        index="tiempo", columns="lon", values="zeta").sort_index()
    if banda.empty:
        return {"detectada": False}

    lons = np.array(banda.columns, dtype=float)
    tiempos = banda.index

    # Punto de partida: máximo de vorticidad en el primer paso, al este de -70
    # (una onda que ya esté encima no hay que "rastrearla", ya llegó).
    inicial = banda.iloc[0]
    candidatos = inicial[lons >= -72]
    if candidatos.empty or candidatos.max() < 1.0:
        return {"detectada": False, "zeta_max": float(banda.values.max())}

    lon_actual = float(candidatos.idxmax())
    trayectoria = [(tiempos[0], lon_actual, float(candidatos.max()))]

    for k in range(1, len(tiempos)):
        fila = banda.iloc[k]
        # la onda solo puede moverse hacia el oeste, entre 10 y 35 km/h
        horas = (tiempos[k] - tiempos[k - 1]).total_seconds() / 3600.0
        km_por_grado = 111.32 * np.cos(np.deg2rad(lat_banda))
        avance_min = VEL_ONDA_MIN_KMH * horas / km_por_grado
        avance_max = VEL_ONDA_MAX_KMH * horas / km_por_grado
        ventana = (lons <= lon_actual - avance_min * 0.3) & \
                  (lons >= lon_actual - avance_max * 1.5)
        if not ventana.any():
            break
        sub = fila[ventana]
        if sub.isna().all():
            break
        lon_actual = float(sub.idxmax())
        trayectoria.append((tiempos[k], lon_actual, float(sub.max())))

    tr = pd.DataFrame(trayectoria, columns=["tiempo", "lon", "zeta"])
    if len(tr) < 3 or tr["zeta"].max() < 1.5:
        return {"detectada": False, "trayectoria": tr}

    # Velocidad media por regresión lineal lon(t)
    horas = (tr["tiempo"] - tr["tiempo"].iloc[0]).dt.total_seconds() / 3600.0
    pendiente = np.polyfit(horas, tr["lon"], 1)[0]          # grados por hora
    km_por_grado = 111.32 * np.cos(np.deg2rad(lat_banda))
    velocidad_kmh = abs(pendiente) * km_por_grado

    llegada = None
    cruce = tr[tr["lon"] <= LON_BQ]
    if not cruce.empty:
        llegada = cruce["tiempo"].iloc[0]
    elif pendiente < -0.001:
        horas_faltantes = (LON_BQ - tr["lon"].iloc[-1]) / pendiente
        if 0 < horas_faltantes < 240:
            llegada = tr["tiempo"].iloc[-1] + timedelta(hours=float(horas_faltantes))

    return {
        "detectada": True,
        "trayectoria": tr,
        "velocidad_kmh": velocidad_kmh,
        "llegada": llegada,
        "zeta_max": float(tr["zeta"].max()),
        "lon_actual": float(tr["lon"].iloc[0]),
    }


def detectar_sistemas(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen diario de los sistemas presentes en el dominio."""
    d = df.copy()
    d["fecha"] = d["tiempo"].dt.date
    filas = []
    for fecha, g in d.groupby("fecha"):
        # Baja en superficie: mínimo de presión en toda la malla
        idx_min = g["pressure_msl"].idxmin() if g["pressure_msl"].notna().any() else None
        # Vaguada en niveles altos: anomalía de 200 hPa en la banda norte
        norte = g[g["lat"] >= 15.0]
        gph200 = norte["geopotential_height_200hPa"].mean()
        gph500 = norte["geopotential_height_500hPa"].mean()
        # Humedad en la capa media cerca de Barranquilla
        cerca = g[(g["lat"] <= 12.5) & (g["lon"].between(-79, -70))]
        filas.append({
            "fecha": fecha,
            "pres_min": g["pressure_msl"].min(),
            "lat_baja": g.loc[idx_min, "lat"] if idx_min is not None else np.nan,
            "lon_baja": g.loc[idx_min, "lon"] if idx_min is not None else np.nan,
            "gph200_norte": gph200,
            "gph500_norte": gph500,
            "rh700_local": cerca["relative_humidity_700hPa"].mean(),
            "cape_local": cerca["cape"].mean(),
            "zeta_local": cerca["zeta"].max() if "zeta" in cerca else np.nan,
        })
    res = pd.DataFrame(filas)
    for c in ("gph200_norte", "gph500_norte"):
        res[f"{c}_anom"] = res[c] - res[c].mean()
    return res


# ==============================================================================
# 4. PROBABILIDAD POR ENSAMBLE
# ==============================================================================

def resumen_ensamble(ens: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad y confianza por día a partir del acuerdo entre miembros."""
    if ens.empty:
        return pd.DataFrame()

    e = ens.copy()
    e["fecha"] = e["tiempo"].dt.date
    diario = e.groupby(["fecha", "miembro"])["precipitation"].sum().reset_index()

    res = diario.groupby("fecha").agg(
        miembros=("precipitation", "size"),
        lluvia_mediana=("precipitation", "median"),
        lluvia_p90=("precipitation", lambda s: s.quantile(0.9)),
        lluvia_max=("precipitation", "max"),
        prob_lluvia=("precipitation",
                     lambda s: 100.0 * (s >= UMBRAL_LLUVIA_DIA).mean()),
        prob_fuerte=("precipitation",
                     lambda s: 100.0 * (s >= UMBRAL_LLUVIA_FUERTE).mean()),
        dispersion=("precipitation", "std"),
    ).reset_index()

    # Confianza: alta cuando los miembros coinciden (poca dispersión relativa)
    rel = res["dispersion"] / res["lluvia_mediana"].clip(lower=1.0)
    res["confianza"] = np.select(
        [rel < 0.8, rel < 2.0], ["Alta", "Media"], default="Baja")
    # Acuerdo extremo también es confianza alta (casi nadie llueve, o casi todos)
    res.loc[(res["prob_lluvia"] >= 85) | (res["prob_lluvia"] <= 15),
            "confianza"] = "Alta"
    return res


# ==============================================================================
# 5. MOTOR DE RECOMENDACIONES
# ==============================================================================

def recomendaciones_dia(fila_ens: pd.Series | None,
                        fila_sist: pd.Series | None,
                        llegada_onda) -> tuple[str, list[str]]:
    """Devuelve (titular, lista de recomendaciones) para un día."""
    prob = float(fila_ens["prob_lluvia"]) if fila_ens is not None else np.nan
    fuerte = float(fila_ens["prob_fuerte"]) if fila_ens is not None else np.nan
    p90 = float(fila_ens["lluvia_p90"]) if fila_ens is not None else np.nan

    recs = []

    # --- Titular según probabilidad ---
    if np.isnan(prob):
        titular = "Sin datos de ensamble"
    elif prob >= 75:
        titular = "☔ Día lluvioso: alta coincidencia entre modelos"
    elif prob >= 45:
        titular = "🌦️ Lluvias probables en la tarde-noche"
    elif prob >= 20:
        titular = "⛅ Chubascos aislados posibles"
    else:
        titular = "☀️ Predominantemente seco"

    # --- Recomendaciones por magnitud ---
    if fuerte >= 40 or p90 >= UMBRAL_ARROYOS:
        recs.append("🚗 **Arroyos:** con este volumen de agua, evita movilizarte "
                    "entre las 15:00 y las 20:00 por corredores propensos a "
                    "arroyos. Si llueve fuerte, espera 40-60 minutos antes de salir.")
        recs.append("🏗️ Asegura estructuras livianas, carpas y vallas: los "
                    "aguaceros de esta magnitud vienen con vendavales de 50-70 km/h "
                    "en la línea de avance de la tormenta.")
    if prob >= 60:
        recs.append("🧺 No es día para pintar, lavar el carro, secar ropa afuera "
                    "ni programar eventos al aire libre sin plan B.")
        recs.append("🌾 Aprovecha para siembra o riego natural; aplaza fumigación "
                    "y aplicación de fertilizantes foliares (el agua los lava).")
    elif prob >= 30:
        recs.append("📅 Programa las actividades al aire libre antes del mediodía. "
                    "La convección en Barranquilla dispara después de las 14:00.")
    else:
        recs.append("🧹 Buena ventana para trabajos exteriores, mantenimiento de "
                    "techos, canaletas y limpieza de sumideros antes del próximo "
                    "sistema.")

    # --- Aportes sinópticos ---
    if fila_sist is not None:
        if pd.notna(fila_sist.get("pres_min")) and fila_sist["pres_min"] <= 1008.5:
            recs.append(
                f"📉 Hay una baja presión de {fila_sist['pres_min']:.0f} hPa en "
                f"{abs(fila_sist['lat_baja']):.0f}°N / "
                f"{abs(fila_sist['lon_baja']):.0f}°W. Vigílala: si se profundiza "
                "o se acerca, el escenario empeora respecto a este pronóstico.")
        if pd.notna(fila_sist.get("gph200_norte_anom")) and \
                fila_sist["gph200_norte_anom"] <= -30:
            recs.append("🌀 Vaguada en niveles altos al norte: favorece divergencia "
                        "en altura y tormentas más altas y organizadas de lo normal.")
        if pd.notna(fila_sist.get("rh700_local")) and fila_sist["rh700_local"] <= 35:
            recs.append("🏜️ Capa media muy seca (posible polvo del Sahara). Aunque "
                        "haga calor, la convección se apaga. Cielo lechoso, atardeceres "
                        "rojizos y aire cargado: cuidado si eres sensible respiratorio.")

    if llegada_onda is not None and pd.notna(fila_ens.get("fecha", np.nan) if fila_ens is not None else np.nan):
        if pd.Timestamp(llegada_onda).date() == fila_ens["fecha"]:
            recs.append("🌀 **Este es el día del paso del eje de la onda tropical.** "
                        "Espera el máximo de actividad entre 12 y 24 h alrededor del "
                        "eje, con lluvias en tandas más que continuas.")

    return titular, recs


# ==============================================================================
# 6. INTERFAZ
# ==============================================================================

def render_pronostico_extendido():
    st.header("🔭 Pronóstico extendido y análisis regional (3 a 10 días)")
    st.caption("Replica el método de lectura de mapas: rastrea los sistemas "
               "mientras todavía están lejos y calcula cuándo llegan.")

    try:
        malla = cargar_malla()
    except Exception as e:
        st.error(f"No se pudo descargar la malla regional: {e}")
        return

    malla = calcular_vorticidad(malla)
    onda = rastrear_onda(malla)
    sistemas = detectar_sistemas(malla)
    ens, avisos = cargar_ensamble()
    for a in avisos:
        st.caption(f"⚠️ {a}")
    resumen = resumen_ensamble(ens)

    # ------------------------------------------------ RASTREO DE ONDA
    st.subheader("🌀 Rastreo de ondas tropicales")

    if onda.get("detectada"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Longitud actual del eje", f"{abs(onda['lon_actual']):.0f}° W")
        c2.metric("Velocidad de desplazamiento", f"{onda['velocidad_kmh']:.0f} km/h")
        if onda["llegada"] is not None:
            dias = (onda["llegada"] - pd.Timestamp.now()).total_seconds() / 86400
            c3.metric("Llegada estimada a Barranquilla",
                      f"{onda['llegada']:%a %d/%m %H:%M}",
                      f"en {max(dias,0):.1f} días")
        else:
            c3.metric("Llegada estimada", "fuera de rango")
        st.success(
            f"Onda del Este identificada con vorticidad máxima de "
            f"{onda['zeta_max']:.1f} ×10⁻⁵ s⁻¹. El eje avanza hacia el oeste; "
            "la actividad más fuerte suele darse cerca y al sur del eje, "
            "en las 12-24 h que rodean su paso."
        )
    else:
        st.info("No hay un eje de onda del Este bien definido entrando al dominio "
                "en los próximos días. Si llueve, sería por calentamiento diurno, "
                "brisa marina o por la Zona de Convergencia Intertropical.")

    # ------------------------------------------------ HOVMÖLLER
    st.subheader("📊 Diagrama de Hovmöller: cómo viajan los sistemas")
    st.caption(f"Vorticidad en 700 hPa a lo largo de {LAT_RASTREO}°N. "
               "Las bandas cálidas inclinadas hacia abajo-izquierda son ondas "
               "viajando hacia el oeste. Barranquilla está en la línea punteada.")

    banda = malla[malla["lat"] == LAT_RASTREO].pivot_table(
        index="tiempo", columns="lon", values="zeta").sort_index()

    fig = go.Figure(go.Heatmap(
        x=banda.columns, y=banda.index, z=banda.values,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="ζ ×10⁻⁵ s⁻¹"),
    ))
    fig.add_vline(x=LON_BQ, line_dash="dot", line_color="white",
                  annotation_text="Barranquilla")
    if onda.get("detectada") and "trayectoria" in onda:
        tr = onda["trayectoria"]
        fig.add_trace(go.Scatter(x=tr["lon"], y=tr["tiempo"], mode="lines+markers",
                                 name="Eje rastreado",
                                 line=dict(color="black", width=2)))
    fig.update_layout(
        xaxis_title="Longitud (° oeste, el flujo va de derecha a izquierda)",
        yaxis_title="Tiempo", height=520,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------ ENSAMBLE
    st.subheader("🎲 Probabilidad por ensamble y nivel de confianza")

    if resumen.empty:
        st.warning("No se pudieron cargar los ensambles. El pronóstico extendido "
                   "queda limitado al análisis de la malla.")
    else:
        st.caption(f"Basado en {int(resumen['miembros'].iloc[0])} corridas del "
                   "modelo con condiciones iniciales perturbadas. Más allá del "
                   "día 5, el porcentaje de miembros que mojan vale más que "
                   "cualquier número exacto de milímetros.")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=resumen["fecha"], y=resumen["prob_lluvia"],
                              name="Prob. de lluvia (%)", marker_color="#1f77b4"))
        fig2.add_trace(go.Bar(x=resumen["fecha"], y=resumen["prob_fuerte"],
                              name="Prob. de lluvia fuerte >15 mm (%)",
                              marker_color="#d62728"))
        fig2.add_trace(go.Scatter(x=resumen["fecha"], y=resumen["lluvia_p90"],
                                  name="Escenario húmedo p90 (mm)", yaxis="y2",
                                  line=dict(color="cyan", width=2, dash="dot")))
        fig2.update_layout(
            barmode="group",
            yaxis=dict(title="Probabilidad (%)", range=[0, 100]),
            yaxis2=dict(title="Lluvia (mm)", overlaying="y", side="right"),
            legend=dict(x=0, y=1.18, orientation="h"),
            margin=dict(l=0, r=0, t=50, b=0), height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------ TARJETAS POR DÍA
    st.subheader("🗓️ Perspectiva día por día y recomendaciones")

    fechas = (resumen["fecha"].tolist() if not resumen.empty
              else sistemas["fecha"].tolist())
    for fecha in fechas[:8]:
        fila_ens = (resumen[resumen["fecha"] == fecha].iloc[0]
                    if not resumen.empty and (resumen["fecha"] == fecha).any()
                    else None)
        fila_sist = (sistemas[sistemas["fecha"] == fecha].iloc[0]
                     if (sistemas["fecha"] == fecha).any() else None)

        titular, recs = recomendaciones_dia(fila_ens, fila_sist,
                                            onda.get("llegada"))

        etiqueta = pd.Timestamp(fecha).strftime("%A %d de %B").capitalize()
        with st.expander(f"**{etiqueta}** — {titular}", expanded=(fecha == fechas[0])):
            if fila_ens is not None:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prob. de lluvia", f"{fila_ens['prob_lluvia']:.0f}%")
                m2.metric("Prob. lluvia fuerte", f"{fila_ens['prob_fuerte']:.0f}%")
                m3.metric("Escenario típico", f"{fila_ens['lluvia_mediana']:.1f} mm",
                          f"hasta {fila_ens['lluvia_max']:.0f} mm")
                m4.metric("Confianza", fila_ens["confianza"])
            for r in recs:
                st.markdown(f"- {r}")

    # ------------------------------------------------ NOTA DE HONESTIDAD
    with st.expander("⏳ Hasta dónde se puede pronosticar de verdad"):
        st.markdown("""
* **Días 1-2:** el modelo determinístico manda. Aciertos altos en ocurrencia,
  menos en ubicación exacta del aguacero.
* **Días 3-5:** es el rango donde el rastreo de ondas y vaguadas brilla. Un
  sistema que hoy está a 2.000 km se ve venir con claridad. Aquí es donde se
  gana la fama de "acertó tres días antes".
* **Días 6-10:** solo sirve la señal del ensamble y la tendencia general
  (racha húmeda vs racha seca). Hablar de horas o de milímetros exactos aquí
  es inventar.
* **Más de 10 días:** entra el terreno de la oscilación Madden-Julian, El Niño
  y La Niña. Son tendencias mensuales, no pronósticos.

Este sistema es una herramienta de apoyo. Para alertas oficiales y decisiones
de riesgo, la referencia es el IDEAM.
        """)

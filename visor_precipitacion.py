"""
================================================================================
 VISOR DE PRECIPITACIÓN FUTURA  |  Caribe Colombiano
================================================================================
 Reemplazo práctico del diagrama de Hovmöller de vorticidad.

 ¿Por qué ese gráfico se veía tan ruidoso?
 -----------------------------------------
 La vorticidad es una DERIVADA del viento. Derivar un campo sobre una malla
 gruesa (3° de separación) amplifica el ruido del modelo: por eso salían
 manchas rojas y azules alternadas sin estructura clara. Para ver una onda
 haría falta una malla más fina y suavizado espacial y temporal.

 La alternativa es mirar directamente lo que importa: LA LLUVIA. Este módulo
 ofrece cuatro vistas, de la más práctica a la más sinóptica:

   1. 📅 Calendario hora × día  -> "¿qué día y a qué hora me va a llover?"
   2. 📈 Meteograma de ensamble -> "¿qué tan seguro es? escenario seco vs húmedo"
   3. 🗺️ Mapa regional por día  -> "¿de dónde viene el agua?"
   4. ➡️ Corte longitud-tiempo  -> "¿la banda de lluvia se acerca o se aleja?"

 Integración:

     from visor_precipitacion import render_visor_precipitacion
     ...
     render_visor_precipitacion()

 Requiere: streamlit pandas numpy plotly requests
================================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

LAT_BQ, LON_BQ = 10.968, -74.781
TZ = "America/Bogota"
API_FORECAST = "https://api.open-meteo.com/v1/forecast"
API_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Malla del mapa regional: Caribe colombiano y aguas vecinas.
MAPA_LATS = np.arange(6.0, 15.1, 1.25)      # 8 filas
MAPA_LONS = np.arange(-80.0, -69.9, 1.25)   # 9 columnas

DIAS_NOMBRE = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue",
               4: "Vie", 5: "Sáb", 6: "Dom"}


# ==============================================================================
# 1. DESCARGAS
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner="Cargando pronóstico horario…")
def cargar_horario(dias: int = 10) -> pd.DataFrame:
    params = {
        "latitude": LAT_BQ, "longitude": LON_BQ,
        "hourly": "precipitation,precipitation_probability,cape,cloud_cover",
        "models": "gfs_seamless", "timezone": TZ, "forecast_days": dias,
    }
    r = requests.get(API_FORECAST, params=params, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"tiempo": pd.to_datetime(h["time"])})
    for k, v in h.items():
        if k != "time":
            df[k] = pd.to_numeric(pd.Series(v), errors="coerce")
    df["fecha"] = df["tiempo"].dt.date
    df["hora"] = df["tiempo"].dt.hour
    return df


@st.cache_data(ttl=1800, show_spinner="Cargando ensambles…")
def cargar_ensamble(dias: int = 10) -> pd.DataFrame:
    piezas = []
    for modelo in ("gfs_seamless", "ecmwf_ifs04"):
        try:
            params = {
                "latitude": LAT_BQ, "longitude": LON_BQ,
                "hourly": "precipitation", "models": modelo,
                "timezone": TZ, "forecast_days": dias,
            }
            r = requests.get(API_ENSEMBLE, params=params, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            t = pd.to_datetime(h["time"])
            for clave, val in h.items():
                if clave == "time" or not clave.startswith("precipitation"):
                    continue
                etiqueta = clave.replace("precipitation", "").strip("_") or "control"
                piezas.append(pd.DataFrame({
                    "tiempo": t,
                    "miembro": f"{modelo}:{etiqueta}",
                    "mm": pd.to_numeric(pd.Series(val), errors="coerce"),
                }))
        except Exception:
            continue
    if not piezas:
        return pd.DataFrame(columns=["tiempo", "miembro", "mm"])
    return pd.concat(piezas, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner="Cargando mapa regional de lluvia…")
def cargar_mapa(dias: int = 7) -> pd.DataFrame:
    """Malla de puntos con acumulado diario. Muy liviano: solo variables diarias."""
    lats, lons = [], []
    for la in MAPA_LATS:
        for lo in MAPA_LONS:
            lats.append(round(float(la), 3))
            lons.append(round(float(lo), 3))

    params = {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "daily": "precipitation_sum,precipitation_probability_max",
        "models": "gfs_seamless", "timezone": TZ, "forecast_days": dias,
    }
    r = requests.get(API_FORECAST, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        data = [data]

    filas = []
    for i, punto in enumerate(data):
        d = punto["daily"]
        sub = pd.DataFrame({"fecha": pd.to_datetime(d["time"]).date})
        # Se usa la coordenada PEDIDA, no la devuelta, para que la rejilla sea regular
        sub["lat"] = lats[i]
        sub["lon"] = lons[i]
        sub["mm"] = pd.to_numeric(pd.Series(d["precipitation_sum"]), errors="coerce")
        sub["prob"] = pd.to_numeric(
            pd.Series(d.get("precipitation_probability_max", [np.nan] * len(sub))),
            errors="coerce")
        filas.append(sub)
    return pd.concat(filas, ignore_index=True)


def etiqueta_fecha(f) -> str:
    ts = pd.Timestamp(f)
    return f"{DIAS_NOMBRE[ts.weekday()]} {ts.day:02d}/{ts.month:02d}"


# ==============================================================================
# 2. VISTA 1 — CALENDARIO HORA × DÍA
# ==============================================================================

def vista_calendario(hor: pd.DataFrame):
    st.markdown("**¿Qué día y a qué hora?** Cada fila es un día, cada columna "
                "una hora. Lo que buscas son las manchas oscuras de la tarde.")

    metrica = st.radio("Mostrar", ["Lluvia esperada (mm)", "Probabilidad (%)"],
                       horizontal=True, key="cal_metrica")
    campo = "precipitation" if metrica.startswith("Lluvia") else "precipitation_probability"

    piv = hor.pivot_table(index="fecha", columns="hora", values=campo,
                          aggfunc="mean").sort_index()
    y = [etiqueta_fecha(f) for f in piv.index]

    escala = "Blues" if campo == "precipitation" else "YlGnBu"
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=y,
        colorscale=escala, zmin=0,
        colorbar=dict(title="mm" if campo == "precipitation" else "%"),
        hovertemplate="%{y} a las %{x}:00<br>%{z:.1f}<extra></extra>",
        xgap=1, ygap=2,
    ))
    # Ventana convectiva típica de Barranquilla
    fig.add_vrect(x0=13.5, x1=21.5, line_width=0, fillcolor="orange", opacity=0.08)
    fig.update_layout(
        xaxis=dict(title="Hora local", dtick=2),
        yaxis=dict(title="", autorange="reversed"),
        height=max(320, 42 * len(y)), margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("La franja naranja marca la ventana convectiva de la tarde "
               "(14:00-21:00), cuando se concentran los aguaceros en la ciudad.")

    # Resumen textual de los días relevantes
    diario = hor.groupby("fecha")["precipitation"].sum()
    lluviosos = diario[diario >= 3].sort_values(ascending=False).head(4)
    if not lluviosos.empty:
        lineas = []
        for f, mm in lluviosos.items():
            dia = hor[hor["fecha"] == f]
            pico = int(dia.loc[dia["precipitation"].idxmax(), "hora"])
            lineas.append(f"- **{etiqueta_fecha(f)}**: {mm:.0f} mm, "
                          f"máximo alrededor de las {pico:02d}:00")
        st.markdown("**Días a vigilar**\n" + "\n".join(lineas))
    else:
        st.info("Ningún día supera los 3 mm acumulados en el rango cargado.")


# ==============================================================================
# 3. VISTA 2 — METEOGRAMA DE ENSAMBLE
# ==============================================================================

def vista_meteograma(ens: pd.DataFrame):
    if ens.empty:
        st.warning("No se pudieron cargar los ensambles.")
        return

    st.markdown("**¿Qué tan confiable es?** Cada miembro es una corrida del "
                "modelo con condiciones iniciales ligeramente distintas. "
                "Si la banda es angosta, hay acuerdo; si es ancha, el "
                "pronóstico está en disputa.")

    e = ens.copy()
    e["fecha"] = e["tiempo"].dt.date
    diario = e.groupby(["fecha", "miembro"])["mm"].sum().reset_index()

    q = diario.groupby("fecha")["mm"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = ["p10", "p25", "p50", "p75", "p90"]
    q = q.reset_index()
    x = [etiqueta_fecha(f) for f in q["fecha"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=q["p90"], name="Escenario húmedo (p90)",
                             line=dict(width=0), showlegend=True))
    fig.add_trace(go.Scatter(x=x, y=q["p10"], name="Escenario seco (p10)",
                             fill="tonexty", fillcolor="rgba(31,119,180,0.18)",
                             line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x, y=q["p75"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q["p25"], name="Rango probable (p25-p75)",
                             fill="tonexty", fillcolor="rgba(31,119,180,0.45)",
                             line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x, y=q["p50"], name="Escenario central (mediana)",
                             line=dict(color="#0b3d91", width=3),
                             mode="lines+markers"))
    fig.update_layout(
        yaxis_title="Lluvia acumulada en el día (mm)",
        legend=dict(x=0, y=1.18, orientation="h"),
        height=420, margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Acumulado del período: lo más útil para planear la semana
    acum = diario.pivot_table(index="miembro", columns="fecha", values="mm").cumsum(axis=1)
    fig2 = go.Figure()
    for m in acum.index[:60]:
        fig2.add_trace(go.Scatter(
            x=[etiqueta_fecha(f) for f in acum.columns], y=acum.loc[m],
            line=dict(width=1, color="rgba(31,119,180,0.25)"),
            showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(
        x=[etiqueta_fecha(f) for f in acum.columns], y=acum.median(),
        line=dict(color="#d62728", width=3), name="Mediana acumulada"))
    fig2.update_layout(
        title="Lluvia acumulada del período (cada línea es un miembro)",
        yaxis_title="mm acumulados", height=380,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    n = diario["miembro"].nunique()
    total_mediana = float(acum.median().iloc[-1])
    st.caption(f"{n} miembros combinados (GFS + ECMWF). Acumulado más probable "
               f"del período: **{total_mediana:.0f} mm**, con un rango entre "
               f"{acum.quantile(0.1).iloc[-1]:.0f} y "
               f"{acum.quantile(0.9).iloc[-1]:.0f} mm.")


# ==============================================================================
# 4. VISTA 3 — MAPA REGIONAL POR DÍA
# ==============================================================================

def vista_mapa(mapa: pd.DataFrame):
    st.markdown("**¿De dónde viene el agua?** Acumulado de lluvia por día sobre "
                "el Caribe. Mueve el día y observa si la mancha se acerca desde "
                "el este (onda tropical) o se forma en el sitio (calentamiento).")

    fechas = sorted(mapa["fecha"].unique())
    etiquetas = [etiqueta_fecha(f) for f in fechas]
    elegido = st.select_slider("Día", options=etiquetas, value=etiquetas[0],
                               key="mapa_dia")
    fecha = fechas[etiquetas.index(elegido)]

    g = mapa[mapa["fecha"] == fecha]
    piv = g.pivot_table(index="lat", columns="lon", values="mm").sort_index()

    zmax = max(5.0, float(np.nanpercentile(mapa["mm"], 97)))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index,
        colorscale=[[0, "#0b1020"], [0.12, "#123b63"], [0.3, "#1f77b4"],
                    [0.55, "#2ca02c"], [0.75, "#ffdd57"], [0.9, "#ff7f0e"],
                    [1, "#d62728"]],
        zmin=0, zmax=zmax, zsmooth="best",
        colorbar=dict(title="mm/día"),
        hovertemplate="%{y:.1f}°N %{x:.1f}°<br>%{z:.1f} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[LON_BQ], y=[LAT_BQ], mode="markers+text",
        marker=dict(size=12, color="white", symbol="star",
                    line=dict(color="black", width=1)),
        text=["Barranquilla"], textposition="top center",
        textfont=dict(color="white"), showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="Longitud", yaxis_title="Latitud",
        height=500, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparación: ¿hacia dónde se mueve el máximo de lluvia?
    centros = []
    for f in fechas:
        gg = mapa[mapa["fecha"] == f]
        peso = gg["mm"].clip(lower=0)
        if peso.sum() > 1:
            centros.append({
                "fecha": etiqueta_fecha(f),
                "lon_centro": float((gg["lon"] * peso).sum() / peso.sum()),
                "mm_dominio": float(peso.mean()),
            })
    if len(centros) >= 3:
        c = pd.DataFrame(centros)
        deriva = c["lon_centro"].iloc[-1] - c["lon_centro"].iloc[0]
        rumbo = ("hacia el oeste (sistema acercándose por el este)"
                 if deriva < -0.5 else
                 "hacia el este (sistema alejándose)" if deriva > 0.5
                 else "prácticamente estacionario")
        st.caption(f"El centro de gravedad de la lluvia en el dominio se desplaza "
                   f"**{rumbo}** ({deriva:+.1f}° de longitud en el período).")


# ==============================================================================
# 5. VISTA 4 — CORTE LONGITUD × TIEMPO DE LLUVIA
# ==============================================================================

def vista_corte(mapa: pd.DataFrame):
    st.markdown("**¿La banda de lluvia se acerca?** Mismo concepto del Hovmöller, "
                "pero con lluvia real en vez de vorticidad: mucho más legible. "
                "Una franja inclinada hacia la izquierda conforme baja = sistema "
                "viajando hacia el oeste.")

    banda = st.slider("Banda latitudinal a promediar (°N)", 6.0, 15.0, (9.0, 13.0),
                      step=1.25, key="corte_banda")
    sel = mapa[(mapa["lat"] >= banda[0]) & (mapa["lat"] <= banda[1])]
    piv = sel.pivot_table(index="fecha", columns="lon", values="mm",
                          aggfunc="mean").sort_index()

    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns,
        y=[etiqueta_fecha(f) for f in piv.index],
        colorscale="YlGnBu", zmin=0, zsmooth="best",
        colorbar=dict(title="mm/día"),
        hovertemplate="%{x:.1f}° · %{y}<br>%{z:.1f} mm<extra></extra>",
    ))
    fig.add_vline(x=LON_BQ, line_dash="dot", line_color="red",
                  annotation_text="Barranquilla", annotation_font_color="red")
    fig.update_layout(
        xaxis_title="Longitud (el flujo de los alisios va de derecha a izquierda)",
        yaxis=dict(title="", autorange="reversed"),
        height=420, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 6. ENTRADA PRINCIPAL
# ==============================================================================

def render_visor_precipitacion():
    st.header("🌧️ Visor de precipitación futura")

    try:
        hor = cargar_horario()
        mapa = cargar_mapa()
    except Exception as e:
        st.error(f"No se pudieron cargar los datos: {e}")
        return
    ens = cargar_ensamble()

    t1, t2, t3, t4 = st.tabs([
        "📅 Calendario hora × día",
        "📈 Meteograma de ensamble",
        "🗺️ Mapa regional por día",
        "➡️ Corte longitud-tiempo",
    ])
    with t1:
        vista_calendario(hor)
    with t2:
        vista_meteograma(ens)
    with t3:
        vista_mapa(mapa)
    with t4:
        vista_corte(mapa)

    with st.expander("ℹ️ Cuál usar según la pregunta"):
        st.markdown("""
| Pregunta | Vista |
|---|---|
| ¿Saco la ropa al patio hoy? | Calendario hora × día |
| ¿Programo el evento el sábado o el domingo? | Meteograma de ensamble |
| ¿Esto que viene es un sistema grande o algo local? | Mapa regional por día |
| ¿Se me acerca una onda por el este? | Corte longitud-tiempo |

Del día 1 al 3 el calendario horario es fiable. Del 4 al 10, solo confía en el
meteograma: ahí importa el rango entre escenario seco y húmedo, no el valor
exacto de la mediana.
        """)

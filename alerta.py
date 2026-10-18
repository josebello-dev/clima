"""
================================================================================
 ALERTAS DE TORMENTA  |  Caribe Weather Analytics
================================================================================
 GitHub lo ejecuta cada 10 minutos (ver .github/workflows/alertas.yml), aunque
 nadie tenga la app abierta. Usa la misma lógica de app.py (rayos del GOES-19,
 seguimiento de tormentas y flujo director) y manda una notificación al celular
 cuando la situación EMPEORA:

   Nivel 1  Tormentas lentas a < 30 km: pueden formarse encima
   Nivel 2  Tormenta en camino (llega en menos de 2 h)
   Nivel 3  Tormenta encima de la ciudad

 También avisa cuando una tormenta que venía o estaba encima ya se alejó.
 No repite el mismo aviso: guarda el último nivel en estado_alerta.json.

 Canales (se configuran como "secrets" del repositorio):
   NTFY_TOPIC                          -> app ntfy (gratis, sin cuenta)
   TELEGRAM_TOKEN + TELEGRAM_CHAT_ID   -> Telegram (opcional)
================================================================================
"""

import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

DIR = Path(__file__).resolve().parent
ESTADO = DIR / "estado_alerta.json"
APP_URL = os.environ.get("APP_URL", "")
REPETIR_MIN = 60          # repetir un aviso de nivel 2-3 si sigue activo tras 60 min
NOMBRES = {0: "Sin tormentas cerca", 1: "Tormentas formándose cerca",
           2: "Tormenta en camino", 3: "Tormenta encima"}


# ------------------------------------------------------------------------------
# 1. Cargar la lógica de app.py sin la interfaz de Streamlit
# ------------------------------------------------------------------------------
def _streamlit_falso():
    st = types.ModuleType("streamlit")

    def cache_data(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    st.cache_data = cache_data
    st.set_page_config = lambda **k: None
    comp = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **k: None
    comp.v1 = v1
    st.components = comp
    sys.modules.update({"streamlit": st, "streamlit.components": comp,
                        "streamlit.components.v1": v1})


def cargar_logica() -> dict:
    _streamlit_falso()
    fuente = (DIR / "app.py").read_text(encoding="utf-8")
    fuente = fuente[:fuente.index("# 5. INTERFAZ")]
    ns: dict = {"__name__": "logica_app"}
    exec(compile(fuente, "app.py", "exec"), ns)
    return ns


# ------------------------------------------------------------------------------
# 2. Nivel de alerta
# ------------------------------------------------------------------------------
def evaluar(ns, rayos: pd.DataFrame, ahora: pd.Timestamp, u: float, v: float, vel: float):
    """Devuelve (nivel, mensaje) con los rayos observados hasta `ahora`."""
    rayos = rayos[rayos["tiempo"] <= ahora]
    if rayos.empty:
        return 0, "No hay rayos en 160 km."
    celdas = ns["seguir_celdas"](rayos, u, v, ahora)
    n_enc = ns["rayos_encima"](rayos)
    if n_enc >= 3:
        return 3, (f"Unos {n_enc} rayos a menos de {ns['GLM_ENCIMA_KM']} km de la "
                   "ciudad en los últimos 15 min.")
    vienen = [c for c in celdas if c["viene"] and c["eta_h"] <= 2.0]
    if vienen:
        c = min(vienen, key=lambda x: x["eta_h"])
        seguro = "" if c["fuente"] == "observada" else " Trayectoria estimada, menos segura."
        return 2, (f"Está a {c['dist']:.0f} km al {ns['rumbo_a_texto'](c['rumbo'])}, "
                   f"avanza hacia el {ns['rumbo_a_texto'](c['rumbo_hacia'])} a "
                   f"{c['vel']:.0f} km/h. Llegaría en unos {c['eta_h'] * 60:.0f} min."
                   + seguro)
    p, txt = ns["prob_observada"](celdas, rayos, 0, n_enc, 0.0, vel)
    if txt and "formarse encima" in txt:
        return 1, txt[0].upper() + txt[1:] + "."
    if celdas:
        c = celdas[0]
        return 0, (f"Tormenta más cercana a {c['dist']:.0f} km al "
                   f"{ns['rumbo_a_texto'](c['rumbo'])}; no viene hacia la ciudad.")
    return 0, "Hay rayos lejanos, sin tormentas activas cerca."


# ------------------------------------------------------------------------------
# 3. Envío
# ------------------------------------------------------------------------------
def enviar(titulo: str, mensaje: str, nivel: int):
    enviado = False
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        cuerpo = {"topic": topic, "title": titulo, "message": mensaje,
                  "priority": {3: 5, 2: 4, 1: 3}.get(nivel, 2),
                  "tags": [{3: "rotating_light", 2: "cloud_with_lightning_and_rain",
                            1: "cloud_with_lightning"}.get(nivel, "white_check_mark")]}
        if APP_URL:
            cuerpo["click"] = APP_URL
        r = requests.post("https://ntfy.sh/", json=cuerpo, timeout=20)
        r.raise_for_status()
        enviado = True
    token = os.environ.get("TELEGRAM_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        texto = f"{titulo}\n{mensaje}" + (f"\n{APP_URL}" if APP_URL else "")
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": texto}, timeout=20)
        r.raise_for_status()
        enviado = True
    if not enviado:
        print("AVISO: no hay canal configurado (NTFY_TOPIC o TELEGRAM_*).")
    return enviado


def leer_estado() -> dict | None:
    try:
        e = json.loads(ESTADO.read_text(encoding="utf-8"))
        return e if "nivel" in e else None
    except Exception:
        return None


def guardar_estado(nivel: int, ultima_alerta: str | None):
    ESTADO.write_text(json.dumps({"nivel": nivel, "ultima_alerta": ultima_alerta}),
                      encoding="utf-8")


# ------------------------------------------------------------------------------
# 4. Programa principal
# ------------------------------------------------------------------------------
def main():
    if "--prueba" in sys.argv:
        enviar("Prueba de alertas ✅",
               "Si ves esto, las alertas de tormenta de Barranquilla funcionan.", 1)
        return

    ns = cargar_logica()
    ahora = pd.Timestamp(datetime.now(ZoneInfo(ns["TZ"]))).tz_localize(None)

    # Flujo director de la hora actual (para tormentas sin trayectoria propia)
    crudo, _ = ns["cargar_datos"]()
    d = ns["calcular_indices"](crudo)
    i = int((d["tiempo"] - ahora.floor("h")).abs().idxmin())
    u = float(d.at[i, "u_dir"]) if pd.notna(d.at[i, "u_dir"]) else 0.0
    v = float(d.at[i, "v_dir"]) if pd.notna(d.at[i, "v_dir"]) else 0.0
    vel = float(d.at[i, "vel_tormenta"]) if pd.notna(d.at[i, "vel_tormenta"]) else 20.0

    rayos, info = ns["cargar_rayos"](ahora.isoformat())
    if info["latencia_min"] > 20:
        print(f"Datos del satélite con {info['latencia_min']:.0f} min de retraso; no se evalúa.")
        return

    nivel, mensaje = evaluar(ns, rayos, ahora, u, v, vel)
    estado = leer_estado()
    if estado is None:   # primera vez o sin memoria: comparar con hace 20 min
        previo, _ = evaluar(ns, rayos, ahora - pd.Timedelta(minutes=20), u, v, vel)
        ultima = None
    else:
        previo, ultima = int(estado.get("nivel", 0)), estado.get("ultima_alerta")

    minutos_desde = ((ahora - pd.Timestamp(ultima)).total_seconds() / 60
                     if ultima else 1e9)
    print(f"{ahora:%Y-%m-%d %H:%M} nivel {nivel} (antes {previo}): {mensaje}")

    if nivel > previo or (nivel >= 2 and minutos_desde >= REPETIR_MIN):
        titulo = {3: "⛈️ Tormenta encima de Barranquilla",
                  2: "🌩️ Tormenta en camino a Barranquilla",
                  1: "⚡ Tormentas formándose cerca"}[nivel]
        if enviar(titulo, mensaje, nivel):
            ultima = ahora.isoformat()
    elif previo >= 2 and nivel == 0:
        enviar("✅ La tormenta se alejó", mensaje, 0)

    guardar_estado(nivel, ultima)


if __name__ == "__main__":
    main()

"""Render de la pestana 'Asistente' de Streamlit.

Dispatcher liviano: ofrece el selector de tarea y delega el render a
uno de los tres sub-modulos especializados (clasificacion, extraccion,
resumen). Mantener esta capa minima evita que un cambio en una de las
tareas obligue a recargar las otras.
"""
from __future__ import annotations

import streamlit as st

from . import ui_classification, ui_extraction, ui_summary


def render() -> None:
    """Punto de entrada de la pestana 'Asistente' en `app.py`."""
    st.caption(
        "Tres tareas especializadas con comparacion **Zero-shot vs Chain-of-Thought**. "
        "Las ejecuciones son efimeras: no se registran en Historial ni Dashboard."
    )
    tarea = st.radio(
        "Tarea",
        options=["Clasificacion de riesgo", "Informacion clave", "Resumen ejecutivo"],
        horizontal=True,
        key="assist_task",
    )
    st.divider()
    if tarea == "Clasificacion de riesgo":
        ui_classification.render()
    elif tarea == "Informacion clave":
        ui_extraction.render()
    else:
        ui_summary.render()

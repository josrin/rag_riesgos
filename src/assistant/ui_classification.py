"""Sub-seccion 'Clasificacion de riesgo' del asistente."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from . import classifier
from .ui_common import RISK_PRETTY, TECHNIQUE_LABELS


def render() -> None:
    """Pinta el panel de clasificacion multi-etiqueta."""
    st.subheader("Clasificacion por tipo de riesgo")
    st.caption(
        "Pega un texto y se clasificara en las 5 categorias de riesgo "
        "(multi-etiqueta con peso). Suma de pesos <= 1.0."
    )
    text = st.text_area(
        "Texto a clasificar",
        height=180,
        placeholder="Ej: Durante el trimestre la tesoreria excedio el limite de 28% de renta variable...",
        key="assist_cls_text",
    )
    run = st.button(
        "Clasificar",
        type="primary",
        disabled=not text.strip(),
        key="assist_cls_run",
    )

    if run:
        with st.spinner("Ejecutando Zero-shot..."):
            t0 = time.time()
            zs = classifier.classify(text, "zero_shot")
            zs_lat = round(time.time() - t0, 1)
        with st.spinner("Ejecutando Chain-of-Thought..."):
            t0 = time.time()
            cot = classifier.classify(text, "cot")
            cot_lat = round(time.time() - t0, 1)
        st.session_state["assist_cls_result"] = {
            "zero_shot": zs,
            "cot": cot,
            "latencies": {"zero_shot": zs_lat, "cot": cot_lat},
            "input": text,
        }

    result = st.session_state.get("assist_cls_result")
    if result:
        col_zs, col_cot = st.columns(2)
        with col_zs:
            _render_column("Zero-shot", result["zero_shot"], result["latencies"]["zero_shot"])
        with col_cot:
            _render_column("Chain-of-Thought", result["cot"], result["latencies"]["cot"])
        st.divider()
        st.download_button(
            "Exportar comparacion (markdown)",
            data=to_markdown(result),
            file_name="clasificacion_comparacion.md",
            mime="text/markdown",
        )


def _render_column(title: str, items: list[dict], latency_s: float) -> None:
    """Pinta una columna con la salida de una tecnica (ZS o CoT)."""
    st.markdown(f"### {title}")
    st.caption(f"Latencia: {latency_s} s")
    if not items:
        st.info("El modelo no detecto ninguna categoria aplicable.")
        return
    df = pd.DataFrame(
        [{"Categoria": RISK_PRETTY.get(x["label"], x["label"]), "Peso": x["weight"]} for x in items]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Categoria"), horizontal=True)


def to_markdown(result: dict) -> str:
    """Serializa la comparacion a markdown para descargar."""
    lines = [
        "# Comparacion de clasificacion de riesgo",
        "",
        "## Texto analizado",
        "",
        result["input"],
        "",
    ]
    for tech in ("zero_shot", "cot"):
        lines.append(f"## {TECHNIQUE_LABELS[tech]} (latencia {result['latencies'][tech]} s)")
        lines.append("")
        if not result[tech]:
            lines.append("_Sin categorias detectadas._")
        else:
            lines.append("| Categoria | Peso |")
            lines.append("| --- | --- |")
            for x in result[tech]:
                lines.append(f"| {RISK_PRETTY.get(x['label'], x['label'])} | {x['weight']} |")
        lines.append("")
    return "\n".join(lines)

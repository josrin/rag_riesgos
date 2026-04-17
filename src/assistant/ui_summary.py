"""Sub-seccion 'Resumen ejecutivo de actas' del asistente."""
from __future__ import annotations

import time

import streamlit as st

from . import corpus_utils, summarizer
from .prompts import SUMMARY_FIELDS
from .ui_common import SUM_PRETTY, TECHNIQUE_LABELS


def render() -> None:
    """Pinta el panel de resumen estructurado sobre actas seleccionadas."""
    st.subheader("Resumen ejecutivo de actas")
    st.caption(
        "Seleccione una o mas actas (documentos de `docs/` con 'acta' en "
        "el nombre). El asistente produce un resumen estructurado en 3 "
        "campos: decisiones, riesgos identificados y acciones pendientes."
    )
    actas = summarizer.acta_docs()
    if not actas:
        st.warning("No hay documentos con 'acta' en el nombre en `docs/`.")
        return
    selected = st.multiselect("Actas", options=actas, key="assist_sum_docs")
    run = st.button(
        "Resumir",
        type="primary",
        disabled=not selected,
        key="assist_sum_run",
    )

    if run:
        results = {"zero_shot": {}, "cot": {}}
        for tech in ("zero_shot", "cot"):
            st.info(f"Ejecutando {TECHNIQUE_LABELS[tech]} sobre {len(selected)} acta(s)...")
            progress = st.progress(0.0, text="Inicializando...")
            total_units = sum(
                max(1, len(corpus_utils.chunks_for_doc(d))) + 1 for d in selected
            )
            state = {"done": 0}

            def _cb(msg: str, doc_idx: int, total_docs: int) -> None:
                """Callback de progreso que suma 1 unidad por evento del summarizer."""
                state["done"] += 1
                pct = min(1.0, state["done"] / max(1, total_units))
                progress.progress(pct, text=msg)

            t0 = time.time()
            per_doc = {}
            for doc in selected:
                per_doc[doc] = summarizer.summarize(doc, tech, progress_cb=_cb)
            lat = round(time.time() - t0, 1)
            progress.progress(1.0, text=f"{TECHNIQUE_LABELS[tech]} listo ({lat} s)")
            results[tech] = {"data": per_doc, "latency_s": lat}

        st.session_state["assist_sum_result"] = {
            "zero_shot": results["zero_shot"],
            "cot": results["cot"],
            "docs": selected,
        }

    result = st.session_state.get("assist_sum_result")
    if result:
        col_zs, col_cot = st.columns(2)
        with col_zs:
            _render_column("Zero-shot", result["zero_shot"], result["docs"])
        with col_cot:
            _render_column("Chain-of-Thought", result["cot"], result["docs"])
        st.divider()
        st.download_button(
            "Exportar comparacion (markdown)",
            data=to_markdown(result),
            file_name="resumen_comparacion.md",
            mime="text/markdown",
        )


def _render_column(title: str, bundle: dict, docs: list[str]) -> None:
    """Pinta una columna con la salida de una tecnica por cada acta."""
    st.markdown(f"### {title}")
    st.caption(f"Latencia: {bundle['latency_s']} s")
    data = bundle["data"]
    for doc in docs:
        st.markdown(f"**{doc}**")
        doc_res = data.get(doc, {})
        for field in SUMMARY_FIELDS:
            items = doc_res.get(field, [])
            st.markdown(f"_{SUM_PRETTY[field]}:_")
            if not items:
                st.caption("_(sin items)_")
            else:
                for it in items:
                    st.markdown(f"- {it}")
        st.markdown("---")


def to_markdown(result: dict) -> str:
    """Serializa la comparacion a markdown para descargar."""
    lines = ["# Comparacion de resumenes ejecutivos", ""]
    lines.append(f"Actas: {', '.join(result['docs'])}")
    lines.append("")
    for tech in ("zero_shot", "cot"):
        bundle = result[tech]
        lines.append(f"## {TECHNIQUE_LABELS[tech]} (latencia {bundle['latency_s']} s)")
        lines.append("")
        for doc in result["docs"]:
            lines.append(f"### {doc}")
            doc_res = bundle["data"].get(doc, {})
            for field in SUMMARY_FIELDS:
                lines.append(f"**{SUM_PRETTY[field]}**")
                items = doc_res.get(field, [])
                if not items:
                    lines.append("- _(sin items)_")
                else:
                    for it in items:
                        lines.append(f"- {it}")
                lines.append("")
    return "\n".join(lines)

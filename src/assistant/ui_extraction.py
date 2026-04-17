"""Sub-seccion 'Extraccion de informacion clave' del asistente."""
from __future__ import annotations

import time

import streamlit as st

from . import corpus_utils, extractor
from .prompts import EXTRACT_CATEGORIES
from .ui_common import CAT_PRETTY, TECHNIQUE_LABELS


def render() -> None:
    """Pinta el panel de extraccion map-reduce sobre docs seleccionados."""
    st.subheader("Extraccion de informacion clave")
    st.caption(
        "Seleccione uno o mas documentos de `docs/`. El asistente extrae "
        "items de las 4 categorias usando map-reduce sobre los chunks "
        "indexados."
    )
    all_docs = corpus_utils.list_docs()
    if not all_docs:
        st.warning("No hay documentos en `docs/`.")
        return
    selected = st.multiselect(
        "Documentos",
        options=all_docs,
        key="assist_ext_docs",
    )
    run = st.button(
        "Extraer",
        type="primary",
        disabled=not selected,
        key="assist_ext_run",
    )

    if run:
        results = {}
        for tech in ("zero_shot", "cot"):
            st.info(f"Ejecutando {TECHNIQUE_LABELS[tech]} sobre {len(selected)} doc(s)...")
            progress = st.progress(0.0, text="Inicializando...")
            # Estimamos docs * (chunks + 1 reduce) al inicio para que el
            # progress avance uniforme; el callback luego suma una unidad
            # por cada evento emitido por el extractor.
            total_units = sum(max(1, len(corpus_utils.chunks_for_doc(d))) + 1 for d in selected)
            state = {"done": 0}

            def _cb(msg: str, doc_idx: int, total_docs: int) -> None:
                """Callback de progreso que suma 1 unidad por evento del extractor."""
                state["done"] += 1
                pct = min(1.0, state["done"] / max(1, total_units))
                progress.progress(pct, text=f"[{doc_idx}/{total_docs}] {msg}")

            t0 = time.time()
            res = extractor.extract(selected, tech, progress_cb=_cb)
            lat = round(time.time() - t0, 1)
            progress.progress(1.0, text=f"{TECHNIQUE_LABELS[tech]} listo ({lat} s)")
            results[tech] = {"data": res, "latency_s": lat}

        st.session_state["assist_ext_result"] = {
            "zero_shot": results["zero_shot"],
            "cot": results["cot"],
            "docs": selected,
        }

    result = st.session_state.get("assist_ext_result")
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
            file_name="extraccion_comparacion.md",
            mime="text/markdown",
        )


def _render_column(title: str, bundle: dict, docs: list[str]) -> None:
    """Pinta una columna con la salida de una tecnica por cada doc."""
    st.markdown(f"### {title}")
    st.caption(f"Latencia: {bundle['latency_s']} s")
    data = bundle["data"]
    for doc in docs:
        st.markdown(f"**{doc}**")
        doc_res = data.get(doc, {})
        for cat in EXTRACT_CATEGORIES:
            items = doc_res.get(cat, [])
            label = CAT_PRETTY[cat]
            with st.expander(f"{label} ({len(items)})", expanded=bool(items)):
                if not items:
                    st.caption("_Sin items detectados._")
                else:
                    for it in items:
                        st.markdown(f"- {it}")


def to_markdown(result: dict) -> str:
    """Serializa la comparacion a markdown para descargar."""
    lines = ["# Comparacion de extraccion de informacion clave", ""]
    lines.append(f"Documentos: {', '.join(result['docs'])}")
    lines.append("")
    for tech in ("zero_shot", "cot"):
        bundle = result[tech]
        lines.append(f"## {TECHNIQUE_LABELS[tech]} (latencia {bundle['latency_s']} s)")
        lines.append("")
        for doc in result["docs"]:
            lines.append(f"### {doc}")
            doc_res = bundle["data"].get(doc, {})
            for cat in EXTRACT_CATEGORIES:
                lines.append(f"**{CAT_PRETTY[cat]}**")
                items = doc_res.get(cat, [])
                if not items:
                    lines.append("- _(sin items)_")
                else:
                    for it in items:
                        lines.append(f"- {it}")
                lines.append("")
    return "\n".join(lines)

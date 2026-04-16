"""Render de la pestaña 'Asistente' de Streamlit.

Tres sub-secciones, cada una con input propio, ejecucion secuencial de
las dos tecnicas (zero-shot primero, CoT despues) y dos columnas con
los resultados lado a lado. Sin persistencia: todo vive en
`st.session_state` durante la sesion del navegador.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from . import classifier, corpus_utils, extractor, summarizer
from .prompts import EXTRACT_CATEGORIES, RISK_LABELS, SUMMARY_FIELDS

TECHNIQUE_LABELS = {"zero_shot": "Zero-shot", "cot": "Chain-of-Thought"}

_CAT_PRETTY = {
    "limites_regulatorios": "Limites regulatorios",
    "indicadores_riesgo": "Indicadores de riesgo",
    "decisiones_comite": "Decisiones del comite",
    "fechas_criticas": "Fechas criticas",
}

_SUM_PRETTY = {
    "decisiones": "Decisiones",
    "riesgos_identificados": "Riesgos identificados",
    "acciones_pendientes": "Acciones pendientes",
}

_RISK_PRETTY = {
    "riesgo_credito": "Credito",
    "riesgo_mercado": "Mercado",
    "riesgo_operacional": "Operacional",
    "riesgo_liquidez": "Liquidez",
    "riesgo_ciberseguridad": "Ciberseguridad",
}


def render() -> None:
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
        _render_classification()
    elif tarea == "Informacion clave":
        _render_extraction()
    else:
        _render_summary()


# ─────────────────────────── CLASIFICACION ───────────────────────────


def _render_classification() -> None:
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
            _render_classification_column("Zero-shot", result["zero_shot"], result["latencies"]["zero_shot"])
        with col_cot:
            _render_classification_column("Chain-of-Thought", result["cot"], result["latencies"]["cot"])
        st.divider()
        st.download_button(
            "Exportar comparacion (markdown)",
            data=_classification_to_markdown(result),
            file_name="clasificacion_comparacion.md",
            mime="text/markdown",
        )


def _render_classification_column(title: str, items: list[dict], latency_s: float) -> None:
    st.markdown(f"### {title}")
    st.caption(f"Latencia: {latency_s} s")
    if not items:
        st.info("El modelo no detecto ninguna categoria aplicable.")
        return
    df = pd.DataFrame(
        [{"Categoria": _RISK_PRETTY.get(x["label"], x["label"]), "Peso": x["weight"]} for x in items]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Categoria"), horizontal=True)


def _classification_to_markdown(result: dict) -> str:
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
                lines.append(f"| {_RISK_PRETTY.get(x['label'], x['label'])} | {x['weight']} |")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────── EXTRACCION ───────────────────────────


def _render_extraction() -> None:
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
            total_units = [0]
            state = {"done": 0}

            def _estimate_units() -> int:
                # docs * (chunks + 1 reduce). Estimado al inicio para el progress.
                return sum(max(1, len(corpus_utils.chunks_for_doc(d))) + 1 for d in selected)

            total_units[0] = _estimate_units()

            def _cb(msg: str, doc_idx: int, total_docs: int) -> None:
                state["done"] += 1
                pct = min(1.0, state["done"] / max(1, total_units[0]))
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
            _render_extraction_column("Zero-shot", result["zero_shot"], result["docs"])
        with col_cot:
            _render_extraction_column("Chain-of-Thought", result["cot"], result["docs"])
        st.divider()
        st.download_button(
            "Exportar comparacion (markdown)",
            data=_extraction_to_markdown(result),
            file_name="extraccion_comparacion.md",
            mime="text/markdown",
        )


def _render_extraction_column(title: str, bundle: dict, docs: list[str]) -> None:
    st.markdown(f"### {title}")
    st.caption(f"Latencia: {bundle['latency_s']} s")
    data = bundle["data"]
    for doc in docs:
        st.markdown(f"**{doc}**")
        doc_res = data.get(doc, {})
        for cat in EXTRACT_CATEGORIES:
            items = doc_res.get(cat, [])
            label = _CAT_PRETTY[cat]
            with st.expander(f"{label} ({len(items)})", expanded=bool(items)):
                if not items:
                    st.caption("_Sin items detectados._")
                else:
                    for it in items:
                        st.markdown(f"- {it}")


def _extraction_to_markdown(result: dict) -> str:
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
                lines.append(f"**{_CAT_PRETTY[cat]}**")
                items = doc_res.get(cat, [])
                if not items:
                    lines.append("- _(sin items)_")
                else:
                    for it in items:
                        lines.append(f"- {it}")
                lines.append("")
    return "\n".join(lines)


# ─────────────────────────── RESUMEN ───────────────────────────


def _render_summary() -> None:
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
            _render_summary_column("Zero-shot", result["zero_shot"], result["docs"])
        with col_cot:
            _render_summary_column("Chain-of-Thought", result["cot"], result["docs"])
        st.divider()
        st.download_button(
            "Exportar comparacion (markdown)",
            data=_summary_to_markdown(result),
            file_name="resumen_comparacion.md",
            mime="text/markdown",
        )


def _render_summary_column(title: str, bundle: dict, docs: list[str]) -> None:
    st.markdown(f"### {title}")
    st.caption(f"Latencia: {bundle['latency_s']} s")
    data = bundle["data"]
    for doc in docs:
        st.markdown(f"**{doc}**")
        doc_res = data.get(doc, {})
        for field in SUMMARY_FIELDS:
            items = doc_res.get(field, [])
            st.markdown(f"_{_SUM_PRETTY[field]}:_")
            if not items:
                st.caption("_(sin items)_")
            else:
                for it in items:
                    st.markdown(f"- {it}")
        st.markdown("---")


def _summary_to_markdown(result: dict) -> str:
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
                lines.append(f"**{_SUM_PRETTY[field]}**")
                items = doc_res.get(field, [])
                if not items:
                    lines.append("- _(sin items)_")
                else:
                    for it in items:
                        lines.append(f"- {it}")
                lines.append("")
    return "\n".join(lines)

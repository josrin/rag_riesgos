"""Interfaz Streamlit para el sistema RAG de riesgos.

Arranque:
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import corpus_sync, dashboard, logger_db, vectorstore
from src.assistant import ui as assistant_ui
from src.config import settings
from src.pipeline import ask_stream, index_corpus, warmup

st.set_page_config(page_title="RAG Riesgos", page_icon="📚", layout="wide")
st.title("Consulta inteligente de documentos de riesgo")
st.caption(f"Modelo: {settings.llm_model} · Embeddings: {settings.embedding_model} · Ollama: {settings.ollama_host}")


@st.cache_resource
def _warmup_once() -> dict:
    return warmup()


_warmup_once()

with st.sidebar:
    st.header("Corpus")
    try:
        n = vectorstore.count()
    except Exception as e:
        n = 0
        st.error(f"Error al conectar a ChromaDB: {e}")
    st.metric("Chunks indexados", n)

    corpus_sync.bootstrap_if_needed()
    state = corpus_sync.scan_state()
    pending = len(state["new"]) + len(state["modified"]) + len(state["deleted"])
    if pending:
        st.warning(
            f"Cambios pendientes en docs/: "
            f"+{len(state['new'])} / ~{len(state['modified'])} / -{len(state['deleted'])}"
        )
        if st.button("Sincronizar (incremental)", type="primary"):
            with st.spinner("Aplicando delta..."):
                result = corpus_sync.sync()
            st.success(
                f"Sync OK en {result['elapsed_s']}s: "
                f"+{len(result['added'])}, ~{len(result['modified'])}, -{len(result['deleted'])} "
                f"({result['indexed_chunks']} chunks)"
            )
            st.rerun()
    else:
        st.caption("docs/ en sincronia con el indice.")

    with st.expander("Reindex completo"):
        st.caption("Borra la coleccion y la reconstruye desde cero. Usar solo si hay corrupcion.")
        if st.button("Reindexar todo"):
            with st.spinner("Procesando PDFs, generando embeddings..."):
                stats = index_corpus(reset=True)
                corpus_sync.rebuild_manifest_from_collection()
            st.success(f"Listo: {stats['chunks']} chunks de {stats['pages']} paginas en {stats['elapsed_s']}s")
            st.rerun()

    st.divider()
    st.header("Retencion")
    st.caption(f"Log de consultas: {settings.log_retention_days} dias")
    if st.button("Purgar registros antiguos"):
        removed = logger_db.purge_old()
        st.info(f"Eliminados {removed} registros caducos")

tab_consulta, tab_dashboard, tab_asistente, tab_historial = st.tabs(
    ["Consulta", "Dashboard", "Asistente", "Historial (30 dias)"]
)

with tab_consulta:
    q = st.text_area(
        "Pregunta al corpus",
        placeholder="Ej: ¿Cuales son los limites de exposicion crediticia definidos en la politica interna?",
        height=100,
    )
    if st.button("Consultar", disabled=not q.strip()):
        if n == 0:
            st.warning("No hay documentos indexados todavia. Coloca los PDFs en docs/ y pulsa 'Reindexar'.")
        else:
            st.subheader("Respuesta")
            with st.spinner("Recuperando fragmentos..."):
                gen, meta = ask_stream(q.strip())
            st.write_stream(gen)
            st.caption(f"Latencia: {meta['latency_ms']} ms")
            warnings = meta.get("faithfulness_warnings", [])
            if warnings:
                with st.expander(f"⚠️ {len(warnings)} posibles inconsistencias detectadas"):
                    st.caption(
                        "La respuesta cita datos que no aparecen literalmente "
                        "en los fragmentos recuperados. Valida manualmente antes de usarla."
                    )
                    for w in warnings:
                        st.markdown(f"- **{w['kind']}**: `{w['claim']}`")

with tab_dashboard:
    dashboard.render()

with tab_asistente:
    assistant_ui.render()

with tab_historial:
    st.caption("Consultas registradas (las mas antiguas de 30 dias se eliminan automaticamente).")
    rows = logger_db.recent(limit=200)
    if not rows:
        st.info("Aun no hay consultas registradas.")
    else:
        for r in rows:
            with st.expander(f"{r['ts']} · {r['question'][:90]}"):
                st.markdown("**Respuesta:**")
                st.write(r["answer"])
                st.markdown("**Fuentes:**")
                st.dataframe(pd.DataFrame(r["sources"]), use_container_width=True, hide_index=True)
                st.caption(f"Latencia: {r['latency_ms']} ms · id={r['id']}")

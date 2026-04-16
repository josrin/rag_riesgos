"""Orquestacion: indexado y consulta extremo a extremo."""
from __future__ import annotations

import logging
import time
from typing import Any

from . import (
    embeddings,
    faithfulness,
    generator,
    logger_db,
    query_decomposer,
    reranker,
    retriever,
    vectorstore,
)
from .config import settings
from .chunking import chunk_pages
from .ingestion import iter_corpus

logger = logging.getLogger(__name__)


def warmup() -> dict[str, Any]:
    """Pre-carga el LLM y el modelo de embeddings en Ollama.

    Llamar al iniciar la app para que la primera consulta real no pague
    el tiempo de arranque de los modelos (~10 s en la maquina de prueba).
    """
    t0 = time.time()
    try:
        embeddings.embed_one("ping")
        generator._client.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "ok"}],
            options={"temperature": 0.0, "num_predict": 4},
        )
        if settings.reranker_enabled:
            reranker.warmup()
        return {"ok": True, "elapsed_s": round(time.time() - t0, 2)}
    except Exception as e:
        logger.warning("Warmup fallo: %s", e)
        return {"ok": False, "elapsed_s": round(time.time() - t0, 2), "error": str(e)}


def index_corpus(reset: bool = True) -> dict[str, Any]:
    t0 = time.time()
    if reset:
        vectorstore.get_collection(reset=True)
    pages = list(iter_corpus())
    if not pages:
        return {"pages": 0, "chunks": 0, "elapsed_s": 0.0}
    chunks = chunk_pages(pages)
    vecs = embeddings.embed_many([c["text"] for c in chunks])
    vectorstore.upsert_chunks(chunks, vecs)
    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "elapsed_s": round(time.time() - t0, 1),
    }


def _retrieve(question: str) -> tuple[list[str], list[dict], str, list[dict]]:
    """Etapa de recuperacion (comun a ask y ask_stream).

    Devuelve: (subqueries, hits, context_text, sources).

    Cuando hay descomposicion en subqueries, incluimos TAMBIEN la
    pregunta original en la fusion multi-query. Las subqueries a veces
    pierden contexto ("¿Que decidio el comite al respecto?" no tiene
    el antecedente "renta variable internacional"); la pregunta original
    ancla ese contexto en el RRF combinado.
    """
    subqueries = query_decomposer.decompose(question)
    queries = subqueries if len(subqueries) == 1 else [question, *subqueries]
    pool_size = settings.reranker_pool_size if settings.reranker_enabled else settings.top_k
    pool = retriever.hybrid_query_multi(queries, top_k=pool_size)
    hits = reranker.rerank(question, pool, top_k=settings.top_k)
    context_text = "\n\n".join(h["text"] for h in hits)
    sources = [
        {
            "source": h["meta"]["source"],
            "page": h["meta"]["page"],
            "section_hint": h["meta"].get("section_hint", ""),
            "distance": round(h["distance"], 4),
        }
        for h in hits
    ]
    return subqueries, hits, context_text, sources


def ask_stream(question: str) -> tuple[Any, dict]:
    """Version streaming de ask().

    Devuelve `(token_generator, meta)`:
      - `token_generator` yieldea los tokens a medida que llegan del LLM.
      - `meta` es un dict mutable que se va llenando durante/despues del
        streaming con: `subqueries`, `sources`, `answer`, `latency_ms`,
        `faithfulness_warnings`.

    Uso tipico desde Streamlit:
        gen, meta = ask_stream(q)
        full = st.write_stream(gen)
        st.caption(f"Latencia: {meta['latency_ms']} ms")
    """
    logger_db.purge_old()
    meta: dict[str, Any] = {
        "subqueries": [],
        "sources": [],
        "answer": "",
        "latency_ms": 0,
        "faithfulness_warnings": [],
    }

    def _gen():
        t0 = time.time()
        subqueries, hits, context_text, sources = _retrieve(question)
        meta["subqueries"] = subqueries
        meta["sources"] = sources

        chunks: list[str] = []
        for token in generator.answer_stream(question, hits):
            chunks.append(token)
            yield token

        full_answer = "".join(chunks).strip()
        meta["answer"] = full_answer
        meta["latency_ms"] = int((time.time() - t0) * 1000)
        meta["faithfulness_warnings"] = faithfulness.check(full_answer, context_text)
        if meta["faithfulness_warnings"]:
            logger.warning(
                "Faithfulness warnings (%s): %s",
                len(meta["faithfulness_warnings"]),
                meta["faithfulness_warnings"],
            )
        logger_db.log_query(
            question, full_answer, sources, meta["latency_ms"], meta["faithfulness_warnings"]
        )

    return _gen(), meta


def ask(question: str) -> dict[str, Any]:
    logger_db.purge_old()
    t0 = time.time()
    subqueries, hits, context_text, sources = _retrieve(question)
    response = generator.answer(question, hits)
    latency_ms = int((time.time() - t0) * 1000)
    warnings = faithfulness.check(response, context_text)
    if warnings:
        logger.warning("Faithfulness warnings (%s): %s", len(warnings), warnings)

    logger_db.log_query(question, response, sources, latency_ms, warnings)
    return {
        "answer": response,
        "sources": sources,
        "latency_ms": latency_ms,
        "context": context_text,
        "subqueries": subqueries,
        "faithfulness_warnings": warnings,
    }

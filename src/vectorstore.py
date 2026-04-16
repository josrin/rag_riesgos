"""ChromaDB persistente. Mantenemos una coleccion unica para todo el corpus
de modo que una misma consulta busque simultaneamente en TODOS los documentos.
"""
from __future__ import annotations

import logging
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

logger = logging.getLogger(__name__)


def get_client() -> chromadb.api.ClientAPI:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(settings.collection_name)
            logger.info("Coleccion previa eliminada")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: Sequence[dict], embeddings: Sequence[Sequence[float]]) -> None:
    coll = get_collection()
    ids = [f"{c['source']}::p{c['page']}::c{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "extraction": c["extraction"],
            "section_hint": c["section_hint"],
            "is_table": bool(c.get("is_table", False)),
        }
        for c in chunks
    ]
    documents = [c["text"] for c in chunks]
    coll.upsert(ids=ids, embeddings=list(embeddings), documents=documents, metadatas=metadatas)
    logger.info("Upsert de %s chunks en la coleccion '%s'", len(chunks), settings.collection_name)


def query(embedding: Sequence[float], top_k: int | None = None) -> list[dict]:
    coll = get_collection()
    res = coll.query(
        query_embeddings=[list(embedding)],
        n_results=top_k or settings.top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def count() -> int:
    return get_collection().count()

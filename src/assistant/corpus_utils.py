"""Helpers para obtener chunks y listados de documentos.

Lee los chunks ya indexados en ChromaDB y los agrupa por documento para
alimentar map-reduce. Si por alguna razon no hay chunks para un doc en
ChromaDB (p.ej. docs/ tiene archivos no sincronizados), se hace fallback
leyendo el archivo completo como un unico 'chunk' con page=1.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .. import vectorstore
from ..config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def list_docs() -> list[str]:
    """Lista todos los documentos soportados presentes en `docs/`."""
    if not settings.docs_dir.exists():
        return []
    return sorted(
        p.name
        for p in settings.docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def list_acta_docs() -> list[str]:
    """Lista docs cuyo nombre (lowercase) contenga 'acta'."""
    return [name for name in list_docs() if "acta" in name.lower()]


def chunks_for_doc(doc_name: str) -> list[dict]:
    """Recupera los chunks de un documento desde ChromaDB.

    Devuelve lista de `{text, source, page, chunk_index, section_hint}`
    ordenados por (page, chunk_index). Si no hay chunks en ChromaDB se
    hace fallback leyendo el archivo completo.
    """
    coll = vectorstore.get_collection()
    full = coll.get(where={"source": doc_name}, include=["documents", "metadatas"])
    docs = full.get("documents") or []
    metas = full.get("metadatas") or []

    if not docs:
        logger.warning("Sin chunks en ChromaDB para %s; fallback a archivo.", doc_name)
        return _fallback_from_disk(doc_name)

    rows = [
        {
            "text": d,
            "source": m.get("source", doc_name),
            "page": m.get("page", 1),
            "chunk_index": m.get("chunk_index", 0),
            "section_hint": m.get("section_hint", ""),
        }
        for d, m in zip(docs, metas)
    ]
    rows.sort(key=lambda r: (r["page"], r["chunk_index"]))
    return rows


def _fallback_from_disk(doc_name: str) -> list[dict]:
    path = settings.docs_dir / doc_name
    if not path.exists() or path.suffix.lower() not in {".txt", ".md"}:
        return []
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []
    return [
        {
            "text": text,
            "source": doc_name,
            "page": 1,
            "chunk_index": 0,
            "section_hint": "",
        }
    ]

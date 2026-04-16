"""Sincronizacion incremental entre `docs/` y el vector store.

Mantiene un manifest (`data/manifest.json`) con el hash SHA-256 de cada
archivo ya indexado. Comparando el manifest con el estado actual de
`docs/` se sabe exactamente que archivos son nuevos, cuales fueron
modificados y cuales eliminados, y se actualiza ChromaDB solo en la
delta — sin tocar los chunks de archivos que no cambiaron.

Invariante: una llamada a `sync()` sin cambios pendientes es
idempotente y barata (solo hashea los archivos).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from . import embeddings, vectorstore
from .chunking import chunk_pages
from .config import settings
from .ingestion import SUPPORTED_EXTS, extract_pdf, extract_text_file

logger = logging.getLogger(__name__)

MANIFEST_PATH = settings.chroma_dir.parent / "manifest.json"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Manifest corrupto, se regenera desde cero")
    return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _current_files() -> dict[str, str]:
    """Devuelve {filename: sha256} de los archivos soportados en docs/."""
    docs_dir = settings.docs_dir
    if not docs_dir.exists():
        return {}
    out: dict[str, str] = {}
    for p in sorted(docs_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out[p.name] = _file_hash(p)
    return out


def scan_state() -> dict[str, Any]:
    """Snapshot de que archivos son unchanged/new/modified/deleted.

    No muta nada (no escribe manifest, no toca ChromaDB).
    """
    manifest = _load_manifest()
    current = _current_files()
    prev_names = set(manifest)
    cur_names = set(current)
    unchanged = sorted(n for n in cur_names & prev_names if current[n] == manifest[n])
    modified = sorted(n for n in cur_names & prev_names if current[n] != manifest[n])
    new = sorted(cur_names - prev_names)
    deleted = sorted(prev_names - cur_names)
    return {
        "unchanged": unchanged,
        "new": new,
        "modified": modified,
        "deleted": deleted,
        "current_hashes": current,
    }


def has_changes(state: dict | None = None) -> bool:
    s = state if state is not None else scan_state()
    return bool(s["new"] or s["modified"] or s["deleted"])


def _extract_one(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return list(extract_pdf(path))
    if ext in {".txt", ".md"}:
        return list(extract_text_file(path))
    return []


def bootstrap_if_needed() -> bool:
    """Si el manifest no existe pero la coleccion ya tiene chunks
    indexados, genera el manifest hasheando docs/ tal como esta. Evita
    que un primer `sync()` reindexe todo el corpus existente.
    """
    if MANIFEST_PATH.exists():
        return False
    try:
        count = vectorstore.count()
    except Exception:
        count = 0
    if count == 0:
        return False
    rebuild_manifest_from_collection()
    logger.info("Manifest bootstrap desde corpus existente")
    return True


def sync() -> dict[str, Any]:
    """Aplica la delta detectada sobre ChromaDB y actualiza el manifest."""
    t0 = time.time()
    bootstrap_if_needed()
    state = scan_state()
    manifest = _load_manifest()
    coll = vectorstore.get_collection()

    to_remove = state["deleted"] + state["modified"]
    for fname in to_remove:
        coll.delete(where={"source": fname})
        logger.info("Borrados chunks de %s", fname)
        manifest.pop(fname, None)

    to_index = state["new"] + state["modified"]
    indexed_pages = 0
    indexed_chunks = 0
    for fname in to_index:
        path = settings.docs_dir / fname
        pages = _extract_one(path)
        if not pages:
            continue
        chunks = chunk_pages(pages)
        if not chunks:
            continue
        vecs = embeddings.embed_many([c["text"] for c in chunks])
        vectorstore.upsert_chunks(chunks, vecs)
        indexed_pages += len(pages)
        indexed_chunks += len(chunks)
        manifest[fname] = state["current_hashes"][fname]

    _save_manifest(manifest)

    return {
        "added": state["new"],
        "modified": state["modified"],
        "deleted": state["deleted"],
        "unchanged": state["unchanged"],
        "indexed_pages": indexed_pages,
        "indexed_chunks": indexed_chunks,
        "elapsed_s": round(time.time() - t0, 2),
    }


def rebuild_manifest_from_collection() -> dict[str, str]:
    """Si el manifest se perdio pero ChromaDB tiene datos, reconstruye
    el manifest hasheando docs/ tal como esta. Util al migrar desde
    una version anterior sin manifest."""
    current = _current_files()
    _save_manifest(current)
    return current

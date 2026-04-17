"""Extraccion de informacion clave con map-reduce sobre chunks.

Para cada documento seleccionado:
  1. Obtiene los chunks ya indexados en ChromaDB.
  2. **Map**: llama al LLM por cada chunk para extraer items de las 4
     categorias (limites_regulatorios, indicadores_riesgo,
     decisiones_comite, fechas_criticas).
  3. **Reduce**: consolida todas las extracciones parciales en una sola
     llamada al LLM que deduplica y unifica.

Cuando hay varios documentos, cada uno se procesa independientemente y
los resultados se agrupan por nombre de archivo.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

from . import corpus_utils, llm_utils, prompts
from .prompts import EXTRACT_CATEGORIES

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int], None]


def _empty_result() -> dict[str, list[str]]:
    """Esqueleto con las 4 categorias vacias; usado como default/fallback."""
    return {k: [] for k in EXTRACT_CATEGORIES}


def _sanitize(raw: object) -> dict[str, list[str]]:
    """Normaliza la respuesta del LLM a `{cat: [str, ...]}` con las 4 claves."""
    out = _empty_result()
    if not isinstance(raw, dict):
        return out
    for cat in EXTRACT_CATEGORIES:
        items = raw.get(cat, [])
        if not isinstance(items, list):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for it in items:
            if isinstance(it, str):
                s = it.strip()
            elif isinstance(it, dict):
                # algunos LLMs devuelven {"item": "...", "detalle": "..."}
                s = " - ".join(str(v).strip() for v in it.values() if v)
            else:
                s = str(it).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s)
        out[cat] = cleaned
    return out


def _merge_partials(partials: Sequence[dict]) -> dict[str, list[str]]:
    """Fallback determinista si el reduce por LLM falla."""
    merged = _empty_result()
    seen: dict[str, set[str]] = {k: set() for k in EXTRACT_CATEGORIES}
    for part in partials:
        sane = _sanitize(part)
        for cat, items in sane.items():
            for it in items:
                key = it.lower()
                if key in seen[cat]:
                    continue
                seen[cat].add(key)
                merged[cat].append(it)
    return merged


def _extract_one_doc(
    doc_name: str,
    technique: str,
    progress_cb: ProgressCb | None,
    doc_idx: int,
    total_docs: int,
) -> dict[str, list[str]]:
    """Map-reduce sobre los chunks de un documento; fallback a merge local si el reduce falla."""
    chunks = corpus_utils.chunks_for_doc(doc_name)
    if not chunks:
        logger.warning("Sin chunks para %s", doc_name)
        return _empty_result()

    map_prompt = (
        prompts.extract_map_zero_shot if technique == "zero_shot" else prompts.extract_map_cot
    )

    partials: list[dict[str, list[str]]] = []
    total_chunks = len(chunks)
    for i, ch in enumerate(chunks, 1):
        if progress_cb:
            progress_cb(f"{doc_name}: chunk {i}/{total_chunks}", doc_idx, total_docs)
        system, user = map_prompt(ch["text"], doc_name)
        raw = llm_utils.chat(system, user, temperature=0.1, num_predict=768)
        parsed = llm_utils.parse_json_response(raw, default={})
        partials.append(_sanitize(parsed))

    if progress_cb:
        progress_cb(f"{doc_name}: consolidando", doc_idx, total_docs)

    sys_r, user_r = prompts.extract_reduce(partials, doc_name)
    raw = llm_utils.chat(sys_r, user_r, temperature=0.1, num_predict=1024)
    reduced = llm_utils.parse_json_response(raw, default=None)
    sane = _sanitize(reduced) if reduced is not None else None

    # Heuristica: si el reduce volvio vacio pero habia items en las
    # parciales, caemos al merge deterministico (el LLM a veces colapsa
    # al consolidar muchos fragmentos).
    if not sane or all(not v for v in sane.values()):
        if any(any(v) for p in partials for v in p.values()):
            logger.info("Reduce LLM vacio en %s; fallback a merge local", doc_name)
            return _merge_partials(partials)
    return sane or _empty_result()


def extract(
    doc_names: Sequence[str],
    technique: str,
    progress_cb: ProgressCb | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Ejecuta extraccion por cada doc y devuelve `{doc_name: {cat: [...]}}`."""
    if technique not in ("zero_shot", "cot"):
        raise ValueError(f"Tecnica no soportada: {technique}")
    out: dict[str, dict[str, list[str]]] = {}
    total = len(doc_names)
    for idx, name in enumerate(doc_names, 1):
        out[name] = _extract_one_doc(name, technique, progress_cb, idx, total)
    return out

"""Resumen ejecutivo de actas con map-reduce.

Misma estrategia que el extractor pero con 3 campos fijos: decisiones,
riesgos_identificados, acciones_pendientes. Se aplica solo a documentos
cuyo nombre (en lowercase) contenga 'acta' — el filtro vive en
`corpus_utils.list_acta_docs()`, asi la UI solo ofrece esos.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

from . import corpus_utils, llm_utils, prompts
from .prompts import SUMMARY_FIELDS

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, int, int], None]


def acta_docs() -> list[str]:
    """Wrapper semantico para la UI; delega en corpus_utils."""
    return corpus_utils.list_acta_docs()


def _empty_result() -> dict[str, list[str]]:
    """Esqueleto con los 3 campos vacios; usado como default/fallback."""
    return {k: [] for k in SUMMARY_FIELDS}


def _sanitize(raw: object) -> dict[str, list[str]]:
    """Normaliza la salida del LLM a `{field: [str, ...]}` con los 3 campos fijos."""
    out = _empty_result()
    if not isinstance(raw, dict):
        return out
    for field in SUMMARY_FIELDS:
        items = raw.get(field, [])
        if not isinstance(items, list):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for it in items:
            s = it.strip() if isinstance(it, str) else str(it).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s)
        out[field] = cleaned
    return out


def _merge_partials(partials: Sequence[dict]) -> dict[str, list[str]]:
    """Fallback determinista: deduplica items de resumenes parciales sin pasar por LLM."""
    merged = _empty_result()
    seen: dict[str, set[str]] = {k: set() for k in SUMMARY_FIELDS}
    for part in partials:
        sane = _sanitize(part)
        for field, items in sane.items():
            for it in items:
                key = it.lower()
                if key in seen[field]:
                    continue
                seen[field].add(key)
                merged[field].append(it)
    return merged


def summarize(
    doc_name: str,
    technique: str,
    progress_cb: ProgressCb | None = None,
) -> dict[str, list[str]]:
    """Resumen ejecutivo de un acta con map-reduce; fallback a merge local si el reduce falla."""
    if technique not in ("zero_shot", "cot"):
        raise ValueError(f"Tecnica no soportada: {technique}")

    chunks = corpus_utils.chunks_for_doc(doc_name)
    if not chunks:
        logger.warning("Sin chunks para acta %s", doc_name)
        return _empty_result()

    map_prompt = (
        prompts.summarize_map_zero_shot if technique == "zero_shot" else prompts.summarize_map_cot
    )

    partials: list[dict[str, list[str]]] = []
    total_chunks = len(chunks)
    for i, ch in enumerate(chunks, 1):
        if progress_cb:
            progress_cb(f"{doc_name}: chunk {i}/{total_chunks}", 1, 1)
        system, user = map_prompt(ch["text"], doc_name)
        raw = llm_utils.chat(system, user, temperature=0.1, num_predict=768)
        parsed = llm_utils.parse_json_response(raw, default={})
        partials.append(_sanitize(parsed))

    if progress_cb:
        progress_cb(f"{doc_name}: consolidando", 1, 1)

    sys_r, user_r = prompts.summarize_reduce(partials, doc_name)
    raw = llm_utils.chat(sys_r, user_r, temperature=0.1, num_predict=1024)
    reduced = llm_utils.parse_json_response(raw, default=None)
    sane = _sanitize(reduced) if reduced is not None else None

    if not sane or all(not v for v in sane.values()):
        if any(any(v) for p in partials for v in p.values()):
            logger.info("Reduce LLM vacio en acta %s; fallback a merge local", doc_name)
            return _merge_partials(partials)
    return sane or _empty_result()

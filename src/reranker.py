"""Cross-encoder reranker con BAAI/bge-reranker-v2-m3.

Pipeline: el retriever hibrido devuelve un pool amplio (p.ej. top-20)
y este modulo lo reordena con un modelo de atencion cruzada que lee la
pregunta y cada chunk en un mismo forward pass. Mucho mas preciso que
la similitud coseno de embeddings, a cambio de una latencia extra
(~200-500 ms para 20 pares en CPU).

El modelo se carga una sola vez (lazy singleton). La primera llamada
descarga ~2 GB a `~/.cache/huggingface/`.
"""
from __future__ import annotations

import logging
from typing import Sequence

from .config import settings

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


def _get_model():
    """Lazy singleton del CrossEncoder; la primera llamada descarga ~2 GB."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Cargando reranker %s (primer uso descarga ~2 GB)", _MODEL_NAME)
        _model = CrossEncoder(_MODEL_NAME, device="cpu", trust_remote_code=False)
    return _model


def _minmax(values: list[float]) -> list[float]:
    """Normaliza a [0,1]. Si todos son iguales, devuelve 0.5 para todos."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def rerank(question: str, hits: Sequence[dict], top_k: int | None = None) -> list[dict]:
    """Reordena hits con fusion ponderada de RRF (retrieval) y cross-encoder.

    `final_score = alpha * rerank_norm + (1 - alpha) * rrf_norm`

    - alpha = 1.0 -> puro cross-encoder (mismo resultado que antes).
    - alpha = 0.0 -> se conserva el orden del hibrido (ignorando el reranker).
    - alpha ~ 0.5 -> mezcla balanceada que preserva fuerza lexica/semantica
      del hibrido pero permite al reranker promover chunks con mejor
      matching semantico profundo.

    Ambos scores se normalizan min-max dentro del pool para evitar que la
    escala del cross-encoder (logits) aplaste el score RRF (~0.03–0.05).
    """
    if not hits:
        return []
    if not settings.reranker_enabled:
        k = top_k or settings.top_k
        return list(hits)[:k]

    k = top_k or settings.top_k
    if len(hits) <= k:
        return list(hits)

    alpha = max(0.0, min(1.0, settings.reranker_alpha))
    model = _get_model()
    pairs = [(question, h["text"]) for h in hits]
    rerank_raw = [float(s) for s in model.predict(pairs).tolist()]
    rrf_raw = [float(h.get("rrf_score", 0.0)) for h in hits]

    rerank_n = _minmax(rerank_raw)
    rrf_n = _minmax(rrf_raw)
    blended = [alpha * a + (1 - alpha) * b for a, b in zip(rerank_n, rrf_n)]

    indexed = sorted(
        enumerate(hits), key=lambda ix: -blended[ix[0]]
    )[:k]
    result: list[dict] = []
    for idx, h in indexed:
        h_copy = dict(h)
        h_copy["rerank_score"] = round(rerank_raw[idx], 4)
        h_copy["blended_score"] = round(blended[idx], 4)
        result.append(h_copy)
    logger.info(
        "Rerank blended (alpha=%.2f): pool=%s -> top_k=%s", alpha, len(hits), k
    )
    return result


def warmup() -> None:
    """Fuerza la carga del modelo (sin inferencia)."""
    _get_model()

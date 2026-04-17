"""Wrapper sobre Ollama para generar embeddings con cache LRU.

El cache evita llamadas redundantes a Ollama para preguntas y chunks que
ya se embebieron. Clave: `(texto, modelo)` — cambiar `EMBEDDING_MODEL`
invalida automaticamente las entradas (no se comparten con otro modelo).

Beneficio tipico:
- Preguntas repetidas: 3 embed/query (pregunta + 2 subqueries) -> 3 hits.
- Eval batch: la 2a corrida completa resuelve todas desde cache.
- Eval-compare sobre distintos LLMs: comparte cache de embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

logger = logging.getLogger(__name__)

_client = ollama.Client(host=settings.ollama_host)
_CACHE_SIZE = 2048


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _fetch_embedding(text: str, model: str) -> tuple[float, ...]:
    """Llamada cruda a Ollama con reintento exponencial ante errores transitorios."""
    resp = _client.embeddings(model=model, prompt=text)
    return tuple(resp["embedding"])


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_embed(text: str, model: str) -> tuple[float, ...]:
    """Capa de cache LRU; la clave es (texto, modelo) para aislar modelos."""
    return _fetch_embedding(text, model)


def embed_one(text: str) -> list[float]:
    """Devuelve el embedding de un texto, sirviendolo del cache LRU si existe."""
    return list(_cached_embed(text, settings.embedding_model))


def embed_many(texts: Sequence[str]) -> list[list[float]]:
    """Embebe una secuencia de textos preservando el orden; loggea cada 25."""
    out: list[list[float]] = []
    for i, t in enumerate(texts, 1):
        out.append(embed_one(t))
        if i % 25 == 0:
            logger.info("  embeddings: %s/%s", i, len(texts))
    return out


def cache_stats() -> dict:
    """Expone hits/misses/tamano/hit_rate del cache para el dashboard."""
    info = _cached_embed.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "size": info.currsize,
        "maxsize": info.maxsize,
        "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) else 0.0,
    }


def clear_cache() -> None:
    """Vacia el cache (util en tests y al cambiar EMBEDDING_MODEL en runtime)."""
    _cached_embed.cache_clear()

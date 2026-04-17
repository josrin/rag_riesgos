"""Recuperacion hibrida: BM25 (lexica) + vectorial, fusionada con RRF.

BM25 aporta precision sobre siglas y jerga (VaR, GARCH, SFC, articulos),
la busqueda vectorial aporta coincidencia semantica. La fusion Reciprocal
Rank Fusion (Cormack et al. 2009) evita tener que normalizar scores
provenientes de escalas distintas.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from . import embeddings, vectorstore
from .config import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60

# Stopwords en espanol para BM25: evitan que palabras vacias
# ("que", "para", "de", "se", etc.) dominen el ranking lexico.
_STOPWORDS_ES = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes aquel aquella
    aquellas aquello aquellos aqui bajo bien cada como con contra cual cuales
    cuando de del desde donde dos durante e el ella ellas ellos en entre era
    eran eres es esa esas ese eso esos esta estaba estaban estamos estan estar
    este esto estos fue fueron ha habia habian han has hasta hay la las le les
    lo los mas me mi mis mucho muy ni no nos nosotros nuestra nuestras nuestro
    nuestros o os otra otras otro otros para pero poco por porque pues que
    quien quienes se ser si sido siempre sin sobre son soy su sus tambien tan
    tanto te tiene tienen todo todos tu tus un una unas uno unos usted ustedes
    vosotros vuestra vuestras vuestro vuestros y ya yo
    """.split()
)


def _strip_accents(text: str) -> str:
    """Quita acentos via NFD + filtrado de combining marks."""
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _tokenize(text: str) -> list[str]:
    """Tokeniza para BM25: lowercase, sin stopwords ES y sin tokens de 1 char."""
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if _strip_accents(t) not in _STOPWORDS_ES and len(t) > 1]


def _chunk_id(meta: dict) -> str:
    """Id estable derivado de (source, page, chunk_index); misma clave que en upsert."""
    return f"{meta['source']}::p{meta['page']}::c{meta['chunk_index']}"


def _hybrid_rank(question: str) -> list[tuple[str, float, dict, str, float]]:
    """Ranking global de TODOS los chunks para una pregunta.

    Devuelve tuplas (chunk_id, rrf_score, meta, text, vec_distance) ordenadas
    de mayor a menor score. Es la primitiva sobre la que se construye
    `hybrid_query` (una sola pregunta) y `hybrid_query_multi` (varias).
    """
    coll = vectorstore.get_collection()
    full = coll.get(include=["documents", "metadatas"])
    ids: Sequence[str] = full["ids"]
    docs: Sequence[str] = full["documents"]
    metas: Sequence[dict] = full["metadatas"]
    if not ids:
        return []

    # Ranking lexico (BM25): tokenizamos con stopwords ES y pedimos scores
    # para TODOS los chunks. Ordenamos por score descendente para obtener
    # posiciones (rank) que alimentan RRF.
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    bm25_scores = bm25.get_scores(_tokenize(question))
    bm25_rank = sorted(range(len(ids)), key=lambda i: -bm25_scores[i])

    # Ranking vectorial: top_k=len(ids) fuerza un ranking global, no un
    # top-N. Guardamos la distancia para exponerla en el hit final (la
    # UI la muestra; el generator no la usa).
    q_vec = embeddings.embed_one(question)
    vec_hits = vectorstore.query(q_vec, top_k=len(ids))
    vec_rank: dict[str, int] = {}
    vec_dist: dict[str, float] = {}
    for rank, h in enumerate(vec_hits):
        hid = _chunk_id(h["meta"])
        vec_rank[hid] = rank
        vec_dist[hid] = h["distance"]

    # Reciprocal Rank Fusion: score = sum_{sistemas} 1 / (k + rank_i).
    # La constante k=60 (Cormack 2009) amortigua las diferencias entre
    # ranks altos — un chunk #1 no aplasta a uno #2. Sumamos en vez de
    # promediar para que los chunks que aparecen en AMBOS rankings
    # (lexico + vectorial) suban al tope incluso si en ninguno estan #1.
    rrf: dict[str, float] = {}
    for rank, idx in enumerate(bm25_rank):
        rrf[ids[idx]] = rrf.get(ids[idx], 0.0) + 1.0 / (_RRF_K + rank)
    for hid, rank in vec_rank.items():
        rrf[hid] = rrf.get(hid, 0.0) + 1.0 / (_RRF_K + rank)

    id_to_idx = {i: n for n, i in enumerate(ids)}
    ranked_ids = sorted(rrf, key=lambda x: -rrf[x])
    return [
        (hid, rrf[hid], metas[id_to_idx[hid]], docs[id_to_idx[hid]], vec_dist.get(hid, 1.0))
        for hid in ranked_ids
    ]


def _build_hit(row: tuple[str, float, dict, str, float]) -> dict:
    """Serializa la tupla interna del ranker al dict que consume el resto del pipeline."""
    _hid, score, meta, text, dist = row
    return {
        "text": text,
        "meta": meta,
        "distance": dist,
        "rrf_score": round(score, 6),
    }


def hybrid_query(question: str, top_k: int | None = None) -> list[dict]:
    """Recupera top-k chunks para una pregunta simple con fusion BM25 + vectorial."""
    k = top_k or settings.top_k
    ranked = _hybrid_rank(question)
    hits = [_build_hit(r) for r in ranked[:k]]
    logger.info("Hibrido devolvio %s hits para: %s", len(hits), question[:60])
    return hits


def hybrid_query_multi(questions: Sequence[str], top_k: int | None = None) -> list[dict]:
    """Fusiona los rankings de varias subqueries con un segundo RRF.

    Cada subquery aporta su ranking global; se suman los terminos RRF y
    se devuelven los top-k por score combinado. Garantiza que cada parte
    de una pregunta compuesta tenga representacion en el contexto.
    """
    if len(questions) <= 1:
        return hybrid_query(questions[0] if questions else "", top_k=top_k)
    k = top_k or settings.top_k
    merged: dict[str, float] = {}
    row_by_id: dict[str, tuple[str, float, dict, str, float]] = {}
    for q in questions:
        ranked = _hybrid_rank(q)
        for rank, row in enumerate(ranked):
            hid = row[0]
            merged[hid] = merged.get(hid, 0.0) + 1.0 / (_RRF_K + rank)
            if hid not in row_by_id:
                row_by_id[hid] = row
    top_ids = sorted(merged, key=lambda x: -merged[x])[:k]
    hits = [
        {
            "text": row_by_id[hid][3],
            "meta": row_by_id[hid][2],
            "distance": row_by_id[hid][4],
            "rrf_score": round(merged[hid], 6),
        }
        for hid in top_ids
    ]
    logger.info("Hibrido-multi devolvio %s hits para %s subqueries", len(hits), len(questions))
    return hits

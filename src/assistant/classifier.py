"""Clasificacion multi-etiqueta con peso.

Dado un texto libre, devuelve una lista de `{label, weight}` con pesos
entre 0 y 1 y suma <= 1. No usa map-reduce: el texto de entrada cabe
siempre en una sola llamada (limite practico ~8k tokens, y la UI le
pone topes al textarea).
"""
from __future__ import annotations

import logging

from . import llm_utils, prompts
from .prompts import RISK_LABELS

logger = logging.getLogger(__name__)


def _sanitize(raw_items: list) -> list[dict]:
    """Filtra etiquetas invalidas, normaliza pesos y clampa a 1.0 la suma."""
    out: list[dict] = []
    seen: set[str] = set()
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if label not in RISK_LABELS or label in seen:
            continue
        try:
            weight = float(item.get("weight", 0))
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        out.append({"label": label, "weight": max(0.0, min(1.0, weight))})
        seen.add(label)

    total = sum(x["weight"] for x in out)
    if total > 1.0 and total > 0:
        for x in out:
            x["weight"] = round(x["weight"] / total, 4)
    else:
        for x in out:
            x["weight"] = round(x["weight"], 4)
    out.sort(key=lambda x: -x["weight"])
    return out


def classify(text: str, technique: str) -> list[dict]:
    """Clasifica `text` con la tecnica indicada ('zero_shot' o 'cot').

    Devuelve lista de `{label, weight}` ordenada de mayor a menor peso.
    Si el LLM devuelve basura, devuelve [] sin lanzar.
    """
    if not text.strip():
        return []
    if technique == "zero_shot":
        system, user = prompts.classify_zero_shot(text)
    elif technique == "cot":
        system, user = prompts.classify_cot(text)
    else:
        raise ValueError(f"Tecnica no soportada: {technique}")

    raw = llm_utils.chat(system, user, temperature=0.1, num_predict=512)
    parsed = llm_utils.parse_json_response(raw, default=[])
    items = _extract_list(parsed)
    return _sanitize(items)


def _extract_list(parsed: object) -> list:
    """El LLM a veces envuelve la lista: {"classification": [...]}, etc.

    Acepta directamente una lista o busca dentro de un dict la primera
    lista con objetos que parezcan `{label, ...}`.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return v
    logger.warning("Respuesta no-lista para clasificacion: %s", type(parsed).__name__)
    return []

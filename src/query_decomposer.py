"""Descomposicion de preguntas compuestas en subqueries independientes.

Evita falsos 'no encuentro informacion' cuando la pregunta combina dos o
mas preguntas simples. Se detecta por heuristica (barata) antes de pedir
al LLM que descomponga, de modo que preguntas simples no paguen la
latencia de una llamada adicional.
"""
from __future__ import annotations

import json
import logging
import re

import ollama

from .config import settings

logger = logging.getLogger(__name__)

_client = ollama.Client(host=settings.ollama_host)

_WH = r"qu[eé]|c[oó]mo|cu[aá]ndo|cu[aá]nto|cu[aá]l(?:es)?|qui[eé]n(?:es)?|d[oó]nde|por qu[eé]|para qu[eé]"
_COMPOUND_RE = re.compile(
    rf"\s+y\s+(?:{_WH})\b|\?\s*[¿\w].*\?|\bademas\b|\badem[aá]s\b",
    re.IGNORECASE,
)

_DECOMPOSE_SYSTEM = (
    "Divide la pregunta compuesta en preguntas simples independientes. "
    "Cada elemento del array DEBE ser una pregunta que termine con signo "
    "de interrogacion. NO incluyas respuestas, afirmaciones ni comentarios. "
    "Responde EXCLUSIVAMENTE con un array JSON de strings. Si la pregunta "
    "ya es simple, devuelve un array con esa unica pregunta. Maximo 3 "
    "subpreguntas.\n\n"
    "Ejemplo de entrada: '¿Cual es el limite X y que decidio el comite Y?'\n"
    "Ejemplo de salida: [\"¿Cual es el limite X?\", \"¿Que decidio el comite Y?\"]"
)


def _looks_like_question(s: str) -> bool:
    """Heuristica liviana: la cadena debe tener >=6 chars y signo de interrogacion."""
    s = s.strip()
    if not s or len(s) < 6:
        return False
    return s.endswith("?") or s.startswith("¿")


def is_compound(question: str) -> bool:
    """True si la pregunta parece compuesta (regex sobre 'y <wh>', 'ademas', multi-'?')."""
    return bool(_COMPOUND_RE.search(question))


def decompose(question: str) -> list[str]:
    """Devuelve subpreguntas; si no es compuesta o el LLM falla, `[question]`."""
    if not is_compound(question):
        return [question]
    try:
        resp = _client.chat(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": question},
            ],
            options={"temperature": 0.0, "num_predict": 256},
        )
        raw = resp["message"]["content"].strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            arr = json.loads(match.group(0))
            if isinstance(arr, list) and arr:
                subs = [s.strip() for s in arr if isinstance(s, str) and _looks_like_question(s)][:3]
                if subs:
                    logger.info("Descompuesto en %s subqueries: %s", len(subs), subs)
                    return subs
    except Exception as e:
        logger.warning("Fallback a pregunta original: %s", e)
    return [question]

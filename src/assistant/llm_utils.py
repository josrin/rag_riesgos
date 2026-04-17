"""Cliente Ollama + parser JSON tolerante para las respuestas del LLM.

El LLM a veces envuelve la respuesta en prosa ('Aqui tienes el JSON: ...
espero que ayude.'). `parse_json_response` extrae el bloque JSON primero
balanceando llaves o corchetes; si falla, devuelve el default indicado.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import ollama

from ..config import settings

logger = logging.getLogger(__name__)

_client = ollama.Client(host=settings.ollama_host)


def chat(system: str, user: str, temperature: float = 0.1, num_predict: int = 1024) -> str:
    """Una llamada sincrona al LLM; devuelve el contenido del mensaje."""
    resp = _client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return resp["message"]["content"].strip()


def _balanced_extract(text: str, open_ch: str, close_ch: str) -> str | None:
    """Devuelve el primer bloque `open_ch...close_ch` balanceado del texto."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_response(raw: str, default: Any) -> Any:
    """Intenta parsear JSON del texto crudo del LLM.

    Orden de intentos:
      1. Parse directo de todo el string.
      2. Extraccion de `{...}` balanceado (objeto).
      3. Extraccion de `[...]` balanceado (array).
      4. Remocion de trailing commas y reintento.
    Si todo falla, devuelve `default`.
    """
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Preferir el delimitador que aparezca PRIMERO — asi evitamos que un
    # parse de `{...}` capture solo el primer objeto de un array `[...]`
    # que empieza antes. Si el LLM devuelve `{..}` envolvente, el `{`
    # aparecera primero y lo tomamos completo.
    pos_bracket = raw.find("[")
    pos_brace = raw.find("{")
    if pos_bracket >= 0 and (pos_brace < 0 or pos_bracket < pos_brace):
        order = (("[", "]"), ("{", "}"))
    else:
        order = (("{", "}"), ("[", "]"))

    for open_ch, close_ch in order:
        block = _balanced_extract(raw, open_ch, close_ch)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                # Los LLMs locales a veces generan `[...,]` o `{"a": 1,}`
                # — JSON estricto los rechaza. Borramos la coma colgante
                # antes del cierre y reintentamos.
                cleaned = re.sub(r",(\s*[}\]])", r"\1", block)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

    logger.warning("No se pudo parsear JSON del LLM; usando default. Raw: %s", raw[:200])
    return default

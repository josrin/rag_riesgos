"""Prompting del LLM con contexto recuperado + citacion obligatoria."""
from __future__ import annotations

from typing import Sequence

import ollama

from .config import settings

_client = ollama.Client(host=settings.ollama_host)

SYSTEM_PROMPT = (
    "Eres un asistente experto en gestion de riesgos y cumplimiento regulatorio. "
    "Respondes SIEMPRE en espanol, de forma precisa y concisa, basandote EXCLUSIVAMENTE "
    "en los fragmentos de contexto entregados.\n\n"
    "REGLAS OBLIGATORIAS:\n"
    "1. Comienza la respuesta directamente con el contenido. NO repitas la pregunta. "
    "NO uses preambulos como 'La respuesta es', 'La pregunta es', 'Segun los fragmentos'.\n"
    "2. Si la pregunta tiene varias partes, responde TODAS en orden. Solo declara "
    "'No encuentro esta informacion en los documentos indexados' si NINGUNA parte "
    "aparece en el contexto. Si al menos una parte aparece, responde las que puedas "
    "y senala explicitamente cual subparte no se encuentra.\n"
    "3. Al final incluye una seccion que empiece con 'Fuentes:' (exactamente). "
    "Formato de cada cita: [archivo.ext, pagina X, seccion Y]\n"
    "   Ejemplo correcto: [doc_02_circular_sfc_riesgo_mercado.txt, pagina 1, seccion Articulo 5]\n"
    "4. No inventes numeros de articulo, fechas ni cifras que no aparezcan textualmente "
    "en el contexto. Cita textualmente las cifras y articulos que menciones.\n"
    "5. Cada fragmento es una unidad independiente. No concatenes el encabezado "
    "'[Fragmento N]' con el contenido del fragmento siguiente."
)


def _format_context(hits: Sequence[dict]) -> str:
    """Renderiza los chunks como bloques '[Fragmento N]' separados para el LLM."""
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        section = m.get("section_hint") or "(sin titulo)"
        header = f"[Fragmento {i}] Fuente: {m['source']} | Pagina: {m['page']} | Seccion: {section}"
        blocks.append(f"{header}\n\n{h['text']}")
    return "\n\n============================\n\n".join(blocks)


def _build_messages(question: str, hits: Sequence[dict]) -> list[dict]:
    """Construye el par (system, user) con contexto embebido para la llamada al LLM."""
    context = _format_context(hits)
    user_prompt = (
        f"Pregunta del analista de riesgos:\n{question}\n\n"
        f"Contexto recuperado de los documentos:\n{context}\n\n"
        "Redacta la respuesta apoyandote solo en el contexto y cierra con la seccion 'Fuentes:'."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def answer(question: str, hits: Sequence[dict]) -> str:
    """Respuesta no-streaming: si no hay hits devuelve el mensaje de 'no encuentro'."""
    if not hits:
        return "No encuentro esta informacion en los documentos indexados."
    resp = _client.chat(
        model=settings.llm_model,
        messages=_build_messages(question, hits),
        options={"temperature": 0.1},
    )
    return resp["message"]["content"].strip()


def answer_stream(question: str, hits: Sequence[dict]):
    """Version streaming: devuelve un generador de strings (tokens)."""
    if not hits:
        yield "No encuentro esta informacion en los documentos indexados."
        return
    stream = _client.chat(
        model=settings.llm_model,
        messages=_build_messages(question, hits),
        options={"temperature": 0.1},
        stream=True,
    )
    for chunk in stream:
        part = chunk.get("message", {}).get("content", "")
        if part:
            yield part

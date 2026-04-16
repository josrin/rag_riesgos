"""Fragmentacion por paginas preservando metadata para citacion.

Decisiones clave:

1. `RecursiveCharacterTextSplitter` respeta limites naturales (parrafo ->
   oracion -> palabra) antes de romper; el solape (~15%) evita que
   conceptos puente entre chunks pierdan contexto.
2. `section_hint`: el primer encabezado detectado dentro del chunk
   (Articulo N, Capitulo, Seccion, Paso N:, heading ALL-CAPS, numerado
   jerarquico). Si el chunk no contiene encabezado propio, hereda el
   ultimo visto en el mismo documento. Asi un chunk con el detalle del
   Paso 5 sigue citando "Seccion 3.1 Descripcion General" si es donde
   arranca la seccion.
3. Tablas markdown (>=2 lineas consecutivas con `|`) se emiten como
   chunks dedicados con `is_table=true` en metadata. Esto evita que el
   splitter parta la tabla por la mitad y que la respuesta tenga que
   reconstruir filas.
"""
from __future__ import annotations

import re
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

_HEADING_RE = re.compile(
    r"(?im)^\s*("
    r"ART[IÍ]CULO\s+\d+[^\n]{0,120}"
    r"|CAP[IÍ]TULO\s+[IVXLCDM\d]+[^\n]{0,120}"
    r"|SECCI[OÓ]N\s+[\d\w\-\.]+[^\n]{0,120}"
    r"|PARTE\s+[IVXLCDM\d]+[^\n]{0,120}"
    r"|PASO\s+\d+[^\n]{0,120}"
    r"|\d+(?:\.\d+){0,3}[\.\s]+[A-Za-zÁÉÍÓÚÑáéíóúñ][^\n]{2,100}"
    r"|[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ\d\-]{2,}){1,8}"
    r")"
)

# >=2 filas consecutivas de pipe-table markdown (encabezado + separador
# + filas, o dos filas seguidas). El separador tipico es `|---|---|`.
# Acepta `\n` o fin de string (`\Z`) para no perder la ultima fila cuando
# la tabla viene al final de la pagina y quedo sin newline despues de un
# strip() aguas arriba.
_TABLE_RE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*(?:\n|\Z)){2,}", re.MULTILINE)


_OWN_HINT_POSITION_THRESHOLD = 0.33  # heading debe aparecer en el primer tercio


def _all_headings(text: str) -> list[tuple[int, str]]:
    """Devuelve [(start_pos, heading_text), ...] en orden de aparicion."""
    return [
        (m.start(), m.group(1).strip()[:120])
        for m in _HEADING_RE.finditer(text)
    ]


def _section_hint(text: str) -> str:
    """Primer heading del chunk SOLO si aparece pronto (primer tercio).

    Un heading muy tardio indica que el chunk trata mayoritariamente de
    la seccion anterior y la proxima empieza al final — en ese caso el
    caller suele preferir heredar el heading anterior.
    """
    hs = _all_headings(text)
    if not hs:
        return ""
    pos, h = hs[0]
    if pos <= len(text) * _OWN_HINT_POSITION_THRESHOLD:
        return h
    return ""


def _split_tables(text: str) -> list[tuple[str, bool]]:
    """Parte el texto en segmentos alternando (prosa, tabla).

    Devuelve lista de tuplas `(fragmento, is_table)`. Preserva el orden.
    """
    segments: list[tuple[str, bool]] = []
    last = 0
    for m in _TABLE_RE.finditer(text):
        if m.start() > last:
            prose = text[last:m.start()].strip()
            if prose:
                segments.append((prose, False))
        table = m.group(0).strip()
        if table:
            segments.append((table, True))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        segments.append((tail, False))
    if not segments:
        segments.append((text, False))
    return segments


def chunk_pages(pages: Iterable[dict]) -> list[dict]:
    """Convierte paginas {source, page, text, extraction} en chunks listos
    para embebido con metadata rica para citacion."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    last_hint_by_source: dict[str, str] = {}

    for page in pages:
        source = page["source"]
        idx_counter = 0
        for segment, is_table in _split_tables(page["text"]):
            if is_table:
                hs = _all_headings(segment)
                own = _section_hint(segment)
                hint = own or last_hint_by_source.get(source, "")
                chunks.append(
                    {
                        "text": segment,
                        "source": source,
                        "page": page["page"],
                        "chunk_index": idx_counter,
                        "extraction": page.get("extraction", "native"),
                        "section_hint": hint,
                        "is_table": True,
                    }
                )
                idx_counter += 1
                if hs:
                    last_hint_by_source[source] = hs[-1][1]
                continue
            for piece in splitter.split_text(segment):
                hs = _all_headings(piece)
                own = _section_hint(piece)
                hint = own or last_hint_by_source.get(source, "")
                chunks.append(
                    {
                        "text": piece,
                        "source": source,
                        "page": page["page"],
                        "chunk_index": idx_counter,
                        "extraction": page.get("extraction", "native"),
                        "section_hint": hint,
                        "is_table": False,
                    }
                )
                idx_counter += 1
                if hs:
                    last_hint_by_source[source] = hs[-1][1]
    return chunks

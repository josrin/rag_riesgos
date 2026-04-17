"""Extraccion de texto de documentos del corpus.

Soporta:
  - PDF digitales (pdfplumber) y escaneados (Tesseract OCR como fallback).
  - Tablas nativas en PDFs: `page.find_tables()` las detecta y las anexa
    al texto como markdown (pipe-table). El chunker las reconoce luego
    con `_TABLE_RE` y las emite como chunks dedicados (`is_table=true`),
    evitando que el splitter las parta por la mitad.
  - Texto plano .txt / .md (lectura directa con deteccion de encoding).

Cada pagina se devuelve como un dict con metadata fuente / pagina para
que aguas abajo podamos citar documento y seccion. Para archivos .txt
el documento completo se entrega como una unica "pagina" (page=1); la
granularidad real la da el chunker.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import pdfplumber

from .config import settings

logger = logging.getLogger(__name__)

MIN_CHARS = 40  # heuristica: menos que esto en una pagina = probablemente escaneada


def _clean(text: str) -> str:
    """Colapsa espacios, elimina nulls y condensa saltos de linea multiples."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """OCR de una pagina concreta. Importa dependencias bajo demanda."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        logger.warning("OCR no disponible (%s). Saltando pagina %s de %s.", e, page_number, pdf_path.name)
        return ""

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    kwargs = {"first_page": page_number, "last_page": page_number, "dpi": 300}
    if settings.poppler_path:
        kwargs["poppler_path"] = settings.poppler_path

    try:
        images = convert_from_path(str(pdf_path), **kwargs)
    except Exception as e:
        logger.warning("No se pudo renderizar %s pag %s: %s", pdf_path.name, page_number, e)
        return ""

    text = ""
    for img in images:
        try:
            text += pytesseract.image_to_string(img, lang="spa+eng") + "\n"
        except Exception as e:
            logger.warning("OCR fallo en %s pag %s: %s", pdf_path.name, page_number, e)
    return text


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """Convierte una tabla (lista de filas) a markdown pipe-table.

    Reemplaza `None` por "", escapa pipes en celdas, normaliza saltos de
    linea internos a espacios y rellena filas cortas con vacias para que
    el ancho coincida con el header. La salida tiene al menos 2 lineas
    con `|` (header + separador), asi `_TABLE_RE` del chunker la detecta.
    """
    if not rows or not rows[0]:
        return ""
    clean: list[list[str]] = []
    for row in rows:
        norm = [(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in row]
        clean.append(norm)
    header = clean[0]
    ncols = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    for row in clean[1:]:
        padded = row + [""] * (ncols - len(row))
        lines.append("| " + " | ".join(padded[:ncols]) + " |")
    return "\n".join(lines) + "\n"


def _extract_page_tables_and_text(page) -> tuple[str, list[str]]:
    """Devuelve (texto_sin_tablas, [markdown_de_cada_tabla]).

    Usa `page.find_tables()` para obtener los bbox y `page.filter()` para
    excluir los objetos dentro de tablas del `extract_text()`, evitando
    duplicar el contenido tabular entre la prosa y el chunk dedicado.
    """
    try:
        tables_found = page.find_tables() or []
    except Exception as e:
        logger.warning("find_tables fallo en pag %s: %s", page.page_number, e)
        return page.extract_text() or "", []

    if not tables_found:
        return page.extract_text() or "", []

    # pdfplumber usa coordenadas (x0, top, x1, bottom) en puntos PDF, con
    # el origen en la esquina superior izquierda de la pagina. Un bbox de
    # tabla es (x0, top, x1, bottom).
    bboxes = [t.bbox for t in tables_found]

    def _outside_tables(obj):
        """Predicado para `page.filter()`: True si el objeto esta fuera de toda tabla."""
        top = obj.get("top")
        bottom = obj.get("bottom")
        x0 = obj.get("x0")
        x1 = obj.get("x1")
        # Objetos sin geometria (p.ej. metadata, anotaciones) no se filtran;
        # los dejamos pasar para no perder contenido inesperado.
        if top is None or bottom is None or x0 is None or x1 is None:
            return True
        # Contencion estricta: el objeto esta DENTRO del bbox de la tabla
        # si sus cuatro esquinas caen dentro. Solo entonces lo excluimos;
        # cualquier solape parcial se conserva en la prosa.
        for bx0, btop, bx1, bbottom in bboxes:
            if top >= btop and bottom <= bbottom and x0 >= bx0 and x1 <= bx1:
                return False
        return True

    try:
        filtered_text = page.filter(_outside_tables).extract_text() or ""
    except Exception as e:
        logger.warning("filter(extract_text) fallo en pag %s: %s", page.page_number, e)
        filtered_text = page.extract_text() or ""

    md_tables = []
    for t in tables_found:
        try:
            rows = t.extract()
        except Exception as e:
            logger.warning("extract tabla fallo en pag %s: %s", page.page_number, e)
            continue
        md = _table_to_markdown(rows)
        if md:
            md_tables.append(md)
    return filtered_text, md_tables


def extract_pdf(pdf_path: Path) -> Iterator[dict]:
    """Devuelve un iterador de {source, page, text} por pagina."""
    logger.info("Procesando %s", pdf_path.name)
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text, md_tables = _extract_page_tables_and_text(page)
            native = _clean(raw_text)
            if len(native) < MIN_CHARS and not md_tables:
                logger.info("  pag %s: texto escaso (%s chars), intentando OCR", i, len(native))
                ocr_text = _clean(_ocr_page(pdf_path, i))
                text = ocr_text if len(ocr_text) > len(native) else native
                method = "ocr" if text == ocr_text and ocr_text else "native"
            else:
                text = native
                method = "native"
                if md_tables:
                    text = (text + "\n\n" + "\n\n".join(md_tables)).strip()
                    logger.info("  pag %s: %s tabla(s) detectada(s)", i, len(md_tables))

            if not text:
                continue
            yield {
                "source": pdf_path.name,
                "page": i,
                "text": text,
                "extraction": method,
            }


def extract_text_file(path: Path) -> Iterator[dict]:
    """Lee un .txt / .md y lo emite como una unica 'pagina'."""
    logger.info("Procesando %s", path.name)
    raw: str | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        logger.warning("No se pudo decodificar %s", path.name)
        return
    text = _clean(raw)
    if not text:
        return
    yield {
        "source": path.name,
        "page": 1,
        "text": text,
        "extraction": "text",
    }


SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def iter_corpus() -> Iterator[dict]:
    """Itera por todos los documentos soportados en docs/ y entrega paginas."""
    files = sorted(
        p for p in settings.docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        logger.warning("No hay documentos soportados en %s (extensiones: %s)",
                       settings.docs_dir, SUPPORTED_EXTS)
        return
    for f in files:
        ext = f.suffix.lower()
        if ext == ".pdf":
            yield from extract_pdf(f)
        elif ext in {".txt", ".md"}:
            yield from extract_text_file(f)

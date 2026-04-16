"""Asistente de tareas especializadas: clasificacion, extraccion y resumen.

Tres sub-tareas, cada una implementada con dos tecnicas (zero-shot y
chain-of-thought) para comparacion lado a lado en la UI Streamlit y en
el script de evaluacion (`scripts/eval_assistant.py`).

Ejecuciones efimeras: no se persisten en queries.db ni aparecen en
Historial/Dashboard.
"""
from .classifier import RISK_LABELS, classify
from .extractor import EXTRACT_CATEGORIES, extract
from .summarizer import SUMMARY_FIELDS, acta_docs, summarize

TECHNIQUES = ("zero_shot", "cot")

__all__ = [
    "TECHNIQUES",
    "RISK_LABELS",
    "EXTRACT_CATEGORIES",
    "SUMMARY_FIELDS",
    "classify",
    "extract",
    "summarize",
    "acta_docs",
]

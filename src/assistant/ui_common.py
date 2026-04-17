"""Constantes compartidas entre las tres sub-secciones del asistente.

Los dicts `_*_PRETTY` traducen los slugs internos (usados en el prompt
y el JSON de salida del LLM) a etiquetas legibles en la UI. Mantener
los slugs estables evita romper los tests del asistente cuando cambian
los labels visibles.
"""
from __future__ import annotations

TECHNIQUE_LABELS = {"zero_shot": "Zero-shot", "cot": "Chain-of-Thought"}

CAT_PRETTY = {
    "limites_regulatorios": "Limites regulatorios",
    "indicadores_riesgo": "Indicadores de riesgo",
    "decisiones_comite": "Decisiones del comite",
    "fechas_criticas": "Fechas criticas",
}

SUM_PRETTY = {
    "decisiones": "Decisiones",
    "riesgos_identificados": "Riesgos identificados",
    "acciones_pendientes": "Acciones pendientes",
}

RISK_PRETTY = {
    "riesgo_credito": "Credito",
    "riesgo_mercado": "Mercado",
    "riesgo_operacional": "Operacional",
    "riesgo_liquidez": "Liquidez",
    "riesgo_ciberseguridad": "Ciberseguridad",
}

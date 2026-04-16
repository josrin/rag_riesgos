"""Plantillas de prompts para las 3 tareas x 2 tecnicas.

Cada funcion devuelve un par `(system, user)` listo para enviar a Ollama.
La idea de la comparacion zero-shot vs chain-of-thought:

- **Zero-shot**: instruccion directa + formato de salida. El LLM responde
  "al tiro" sin exponer razonamiento intermedio. Mas rapido, mas barato,
  suele bastar cuando la tarea tiene una estructura clara.
- **Chain-of-thought**: se le pide explicitamente pensar paso a paso
  antes de emitir el JSON. Mas lento, mas tokens, pero tiende a acertar
  mejor en casos ambiguos (clasificaciones mixtas, extracciones con
  falsos positivos evidentes).

Todos los prompts exigen JSON estricto al final para que la salida sea
parseable programaticamente sin depender de markdown o texto libre.
"""
from __future__ import annotations

from typing import Sequence

RISK_LABELS = (
    "riesgo_credito",
    "riesgo_mercado",
    "riesgo_operacional",
    "riesgo_liquidez",
    "riesgo_ciberseguridad",
)

EXTRACT_CATEGORIES = (
    "limites_regulatorios",
    "indicadores_riesgo",
    "decisiones_comite",
    "fechas_criticas",
)

SUMMARY_FIELDS = ("decisiones", "riesgos_identificados", "acciones_pendientes")


# ─────────────────────────── CLASIFICACION ───────────────────────────

_LABELS_LIST = ", ".join(RISK_LABELS)

_CLS_FORMAT = (
    'Devuelve EXCLUSIVAMENTE un array JSON de objetos con la forma '
    '[{"label": "<una_de_las_categorias>", "weight": <0.0-1.0>}]. '
    'Solo incluye categorias aplicables (peso > 0). La suma de pesos '
    'debe ser <= 1.0. No incluyas texto fuera del JSON.'
)


def classify_zero_shot(text: str) -> tuple[str, str]:
    system = (
        "Eres un experto en gestion de riesgos financieros. Clasifica el "
        f"texto en una o mas de estas categorias: {_LABELS_LIST}. "
        "Asigna un peso a cada categoria relevante segun cuan dominante "
        f"sea en el texto. {_CLS_FORMAT}"
    )
    user = f"Texto a clasificar:\n\n{text}"
    return system, user


def classify_cot(text: str) -> tuple[str, str]:
    system = (
        "Eres un experto en gestion de riesgos financieros. Clasifica el "
        f"texto en una o mas de estas categorias: {_LABELS_LIST}.\n\n"
        "Razona paso a paso antes de responder:\n"
        "PASO 1: Identifica las senales lexicas y semanticas del texto "
        "(terminos tecnicos, indicadores, actores mencionados).\n"
        "PASO 2: Para cada categoria posible, evalua si hay evidencia y "
        "cuan central es al texto.\n"
        "PASO 3: Asigna un peso a cada categoria relevante de modo que "
        "los pesos reflejen la dominancia relativa (suma <= 1.0).\n"
        "PASO 4: Emite la respuesta final.\n\n"
        f"FORMATO FINAL: {_CLS_FORMAT}"
    )
    user = (
        f"Texto a clasificar:\n\n{text}\n\n"
        "Escribe tu razonamiento con las etiquetas PASO 1..4 y al final "
        "emite el array JSON en una linea separada iniciada por '```json' "
        "o simplemente el array directo."
    )
    return system, user


# ─────────────────────────── EXTRACCION ───────────────────────────

_EXTRACT_KEYS = ", ".join(f'"{k}"' for k in EXTRACT_CATEGORIES)

_EXTRACT_FORMAT = (
    "Devuelve EXCLUSIVAMENTE un objeto JSON con las claves: "
    f"{_EXTRACT_KEYS}. Cada valor es un array de strings (items "
    "concretos extraidos textualmente o parafraseados con cifras "
    "intactas). Si una categoria no aparece en el fragmento, usa []."
)


def extract_map_zero_shot(chunk_text: str, doc_name: str) -> tuple[str, str]:
    system = (
        "Eres un experto en analisis de documentos regulatorios y de "
        "riesgo. Extrae del fragmento items que correspondan a estas "
        "categorias:\n"
        "- limites_regulatorios: umbrales, topes, porcentajes maximos, "
        "exposiciones limite que impone la norma o la politica.\n"
        "- indicadores_riesgo: KRIs, KPIs, metricas (VaR, ratio, tasa), "
        "con valor si aparece.\n"
        "- decisiones_comite: acuerdos, aprobaciones o rechazos de un "
        "comite o instancia formal.\n"
        "- fechas_criticas: fechas con relevancia operativa "
        "(deadlines, incidentes, vigencias, proximas reuniones).\n\n"
        f"{_EXTRACT_FORMAT}"
    )
    user = f"Documento: {doc_name}\n\nFragmento:\n\n{chunk_text}"
    return system, user


def extract_map_cot(chunk_text: str, doc_name: str) -> tuple[str, str]:
    system = (
        "Eres un experto en analisis de documentos regulatorios y de "
        "riesgo. Tu tarea es extraer items de 4 categorias, razonando "
        "paso a paso para reducir falsos positivos.\n\n"
        "PASO 1: Lee el fragmento e identifica su proposito (norma, "
        "informe, acta, metodologia, politica).\n"
        "PASO 2: Para cada categoria, identifica candidatos:\n"
        "  - limites_regulatorios: umbrales, topes, maximos (ej. '10% "
        "del patrimonio', 'no mayor a COP 100M').\n"
        "  - indicadores_riesgo: metricas con nombre reconocible (VaR, "
        "KRI, ratio, tasa).\n"
        "  - decisiones_comite: verbos de decision ('aprobo', 'decidio', "
        "'rechazo', 'acordo') con sujeto institucional.\n"
        "  - fechas_criticas: fechas con consecuencia operativa.\n"
        "PASO 3: Descarta falsos positivos (cifras ambiguas, verbos no "
        "decisionales, fechas sin contexto).\n"
        "PASO 4: Emite el JSON final.\n\n"
        f"FORMATO FINAL: {_EXTRACT_FORMAT}"
    )
    user = (
        f"Documento: {doc_name}\n\nFragmento:\n\n{chunk_text}\n\n"
        "Expon brevemente los pasos 1-3 y al final emite el objeto JSON."
    )
    return system, user


def extract_reduce(partials: Sequence[dict], doc_name: str) -> tuple[str, str]:
    """Consolidacion: unifica y deduplica los items de cada categoria."""
    import json as _json

    partials_text = _json.dumps(list(partials), ensure_ascii=False, indent=2)
    system = (
        "Consolidas extracciones parciales de un documento dividido en "
        "fragmentos. Para cada categoria, unifica los items manteniendo "
        "los mas especificos y eliminando duplicados o parafraseos "
        "equivalentes. Mantener las cifras originales.\n\n"
        f"{_EXTRACT_FORMAT}"
    )
    user = (
        f"Documento: {doc_name}\n\n"
        f"Extracciones parciales por fragmento:\n{partials_text}\n\n"
        "Emite el objeto JSON consolidado con las 4 claves."
    )
    return system, user


# ─────────────────────────── RESUMEN ───────────────────────────

_SUMMARY_KEYS = ", ".join(f'"{k}"' for k in SUMMARY_FIELDS)

_SUMMARY_FORMAT = (
    "Devuelve EXCLUSIVAMENTE un objeto JSON con las claves: "
    f"{_SUMMARY_KEYS}. Cada valor es un array de strings (items "
    "concretos). Arrays vacios si el fragmento no aporta a esa clave."
)


def summarize_map_zero_shot(chunk_text: str, doc_name: str) -> tuple[str, str]:
    system = (
        "Eres un analista que prepara resumenes ejecutivos de actas de "
        "comites de riesgos. Extrae del fragmento:\n"
        "- decisiones: acuerdos formales adoptados en el acta.\n"
        "- riesgos_identificados: riesgos o amenazas mencionadas.\n"
        "- acciones_pendientes: tareas por ejecutar con responsable o "
        "plazo si aparecen.\n\n"
        f"{_SUMMARY_FORMAT}"
    )
    user = f"Acta: {doc_name}\n\nFragmento:\n\n{chunk_text}"
    return system, user


def summarize_map_cot(chunk_text: str, doc_name: str) -> tuple[str, str]:
    system = (
        "Eres un analista que prepara resumenes ejecutivos de actas de "
        "comites de riesgos. Razona paso a paso para evitar mezclar "
        "categorias:\n\n"
        "PASO 1: Identifica si el fragmento es narrativo (contexto), "
        "decisional (acuerdos), analitico (riesgos) o de planificacion "
        "(acciones).\n"
        "PASO 2: Para cada campo:\n"
        "  - decisiones: verbos 'aprobo', 'acordo', 'decidio', "
        "'autorizo'. Debe ser accion pasada formalmente tomada.\n"
        "  - riesgos_identificados: amenazas, exposiciones o eventos "
        "negativos mencionados (incluso sin decision asociada).\n"
        "  - acciones_pendientes: verbos en futuro o infinitivo "
        "('ejecutar', 'entregar', 'presentar'), con responsable o plazo "
        "si estan.\n"
        "PASO 3: Descarta items que no encajen limpiamente en ninguna "
        "categoria (no forzar).\n"
        "PASO 4: Emite el JSON final.\n\n"
        f"FORMATO FINAL: {_SUMMARY_FORMAT}"
    )
    user = (
        f"Acta: {doc_name}\n\nFragmento:\n\n{chunk_text}\n\n"
        "Expon los pasos 1-3 brevemente y al final emite el JSON."
    )
    return system, user


def summarize_reduce(partials: Sequence[dict], doc_name: str) -> tuple[str, str]:
    import json as _json

    partials_text = _json.dumps(list(partials), ensure_ascii=False, indent=2)
    system = (
        "Consolidas resumenes parciales de un acta dividida en "
        "fragmentos. Unifica los items de cada campo eliminando "
        "duplicados y parafraseos equivalentes. Mantener los verbos y "
        "cifras originales.\n\n"
        f"{_SUMMARY_FORMAT}"
    )
    user = (
        f"Acta: {doc_name}\n\n"
        f"Resumenes parciales por fragmento:\n{partials_text}\n\n"
        "Emite el objeto JSON consolidado con las 3 claves."
    )
    return system, user

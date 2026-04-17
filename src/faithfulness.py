"""Guardrail de fidelidad: detecta si la respuesta cita cifras, articulos
o fechas que no aparecen textualmente en el contexto recuperado.

En un dominio regulatorio una cifra inventada es inaceptable. Este
modulo no puede garantizar la ausencia de alucinaciones semanticas
(p.ej. parafrasis erroneas) pero si bloquea la clase mas comun: numeros
y referencias concretas que el LLM fabrica cuando el contexto es
insuficiente.
"""
from __future__ import annotations

import re
import unicodedata

# Cifras con o sin decimales y simbolo % opcional, admitiendo separador
# colombiano de miles (COP 12,350 millones, 99%, 2.1%).
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*%?\b")
_ARTICLE_RE = re.compile(r"art[íi]culo\s+\d+[a-z]?", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}\s+de\s+[a-záéíóúñ]+\s+de\s+\d{4}\b", re.IGNORECASE)

_TRIVIAL_NUMBERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100"}


def _norm(s: str) -> str:
    """Normaliza a lowercase sin acentos y con espacios colapsados para comparar."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s.lower())
    return s


def _claim_in_context(claim: str, context_norm: str) -> bool:
    """True si el claim aparece en el contexto, tolerando espacios alrededor del %."""
    claim_norm = _norm(claim).strip()
    if claim_norm in context_norm:
        return True
    # Tolerar diferencias menores de espaciado alrededor del %
    no_space = re.sub(r"\s+", "", claim_norm)
    return no_space in re.sub(r"\s+", "", context_norm)


def check(answer: str, context: str) -> list[dict]:
    """Devuelve lista de claims de la respuesta que no aparecen en contexto.

    Cada entrada: {"kind": str, "claim": str}. Lista vacia = sin warnings.
    """
    context_norm = _norm(context)
    warnings: list[dict] = []
    seen: set[str] = set()

    def check_matches(pattern: re.Pattern, kind: str, filter_trivial: bool = False) -> None:
        """Encuentra claims del tipo dado en la respuesta y marca los no presentes."""
        for m in pattern.finditer(answer):
            claim = m.group().strip()
            key = f"{kind}:{_norm(claim)}"
            if key in seen:
                continue
            seen.add(key)
            if filter_trivial:
                bare = re.sub(r"[^\d]", "", claim)
                if bare in _TRIVIAL_NUMBERS:
                    continue
            if not _claim_in_context(claim, context_norm):
                warnings.append({"kind": kind, "claim": claim})

    check_matches(_NUMBER_RE, "numero", filter_trivial=True)
    check_matches(_ARTICLE_RE, "articulo")
    check_matches(_DATE_RE, "fecha")
    return warnings

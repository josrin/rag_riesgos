"""Evaluacion ad-hoc con 12 preguntas cubriendo los 5 documentos."""
from __future__ import annotations

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.pipeline import ask

QUESTIONS = [
    # --- Nivel 1: lookup simple ---
    ("L1", "¿Cuál es el límite de exposición por emisor individual en la política de crédito?"),
    ("L1", "¿Cuándo entra en vigencia la Circular 034 de 2025 de la SFC?"),
    ("L1", "¿Cuánto fueron las pérdidas operacionales totales del Q4 2025?"),
    # --- Nivel 2: lookup con contexto ---
    ("L2", "¿Qué metodología se usa para calcular el VaR?"),
    ("L2", "¿Qué medidas se aprobaron tras el incidente de phishing del 5 de diciembre?"),
    ("L2", "¿Qué tests estadísticos se usan en el backtesting del modelo VaR?"),
    # --- Nivel 3: sintesis dentro del mismo documento ---
    ("L3", "¿Qué haircuts se aplican a las distintas garantías?"),
    ("L3", "¿Qué KRIs estuvieron en estado rojo en el Q4 2025?"),
    ("L3", "¿Cuáles son todos los límites de inversión establecidos por la SFC en la Circular 034?"),
    # --- Nivel 4: cross-document ---
    ("L4", "¿El VaR actual del portafolio cumple con los requisitos de la SFC?"),
    ("L4", "¿Cuál es el límite de renta variable internacional y qué decidió el comité al respecto?"),
    # --- Nivel 5: detalle fino / inferencia ---
    ("L5", "¿Cuánto tiempo pasó entre el compromiso de credenciales y la detección del ataque de phishing?"),
]


def main() -> None:
    total_t = time.time()
    for i, (lvl, q) in enumerate(QUESTIONS, 1):
        print("=" * 90)
        print(f"[{i:02d}] {lvl}  {q}")
        print("-" * 90)
        r = ask(q)
        print(r["answer"])
        print()
        print(f"Latencia: {r['latency_ms']} ms | Chunks recuperados: "
              + ", ".join(f"{s['source']}[§{s.get('section_hint','')[:30]}]" for s in r["sources"]))
        print()
    print("=" * 90)
    print(f"Total: {time.time() - total_t:.1f} s")


if __name__ == "__main__":
    main()

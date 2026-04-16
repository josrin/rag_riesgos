"""Comparacion de LLMs sobre el mismo ground truth.

Uso:
    python -m scripts.eval_compare llama3 llama3.1:8b mistral

Itera cada modelo, reescribe temporalmente `LLM_MODEL` en
`generator.settings` y `pipeline.settings`, corre el harness y reporta
metricas lado a lado. No persiste cambios en `.env`.
"""
from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import ollama

from src import generator
from src.config import settings as _settings
from scripts.eval import evaluate

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "eval" / "compare_report.json"


def _override_llm_model(model: str) -> None:
    """`Settings` es frozen; parcheamos los atributos consumidos por el
    generador y el pipeline sin mutar el dataclass original."""
    object.__setattr__(_settings, "llm_model", model)
    generator._client = ollama.Client(host=_settings.ollama_host)


def _is_model_available(model: str) -> bool:
    try:
        listing = ollama.list()
        names = {m.model for m in listing.models}
        return model in names or f"{model}:latest" in names
    except Exception:
        return False


def main(models: list[str]) -> None:
    original = _settings.llm_model
    comparison: list[dict] = []
    try:
        for model in models:
            if not _is_model_available(model):
                print(f"\n[!] Modelo '{model}' no esta disponible. `ollama pull {model}` primero. Saltando.")
                continue
            print(f"\n=== Evaluando con {model} ===")
            _override_llm_model(model)
            t0 = time.time()
            report = evaluate()
            wall = round(time.time() - t0, 1)
            agg = report["aggregate"]
            comparison.append({"model": model, "aggregate": agg, "wall_time_s": wall})
            print(f"  recall@5={agg['source_recall_at_5']:.0%}  "
                  f"acc={agg['answer_accuracy']:.0%}  "
                  f"fth={agg['faithfulness']:.0%}  "
                  f"p50={agg['latency_p50_ms']} ms  "
                  f"p95={agg['latency_p95_ms']} ms  "
                  f"wall={wall} s")
    finally:
        _override_llm_model(original)

    if not comparison:
        print("\nNingun modelo evaluado.")
        return

    print("\n" + "=" * 100)
    print(f"{'Modelo':<22} {'Rec@5':>7} {'Acc':>6} {'Fth':>6} {'p50':>8} {'p95':>8} {'Wall':>8}")
    print("-" * 100)
    for c in comparison:
        a = c["aggregate"]
        print(f"{c['model']:<22} "
              f"{a['source_recall_at_5']:>6.0%} "
              f"{a['answer_accuracy']:>5.0%} "
              f"{a['faithfulness']:>5.0%} "
              f"{a['latency_p50_ms']:>6} ms "
              f"{a['latency_p95_ms']:>6} ms "
              f"{c['wall_time_s']:>6} s")
    print("=" * 100)

    REPORT_PATH.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReporte guardado en {REPORT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.eval_compare <modelo1> [modelo2 ...]")
        sys.exit(1)
    main(sys.argv[1:])

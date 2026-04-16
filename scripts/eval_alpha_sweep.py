"""Sweep de RERANKER_ALPHA para encontrar el punto donde el reranker
deja de regresionar al blend con RRF. Solo util con RERANKER_ENABLED=true.

    python -m scripts.eval_alpha_sweep
"""
from __future__ import annotations

import io
import json
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.config import settings
from scripts.eval import evaluate

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def main() -> None:
    object.__setattr__(settings, "reranker_enabled", True)
    results = []
    for alpha in ALPHAS:
        object.__setattr__(settings, "reranker_alpha", alpha)
        t0 = time.time()
        report = evaluate()
        wall = round(time.time() - t0, 1)
        agg = report["aggregate"]
        fails = [r["id"] for r in report["results"]
                 if r.get("answer_accuracy") is not None and r["answer_accuracy"] < 1.0]
        results.append({"alpha": alpha, "aggregate": agg, "wall_s": wall, "fails": fails})
        print(f"alpha={alpha}  rec={agg['source_recall_at_5']:.0%}  "
              f"acc={agg['answer_accuracy']:.0%}  fth={agg['faithfulness']:.0%}  "
              f"p50={agg['latency_p50_ms']} ms  wall={wall}s  fails={fails}")

    with open("eval/alpha_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nGuardado en eval/alpha_sweep.json")


if __name__ == "__main__":
    main()

"""Harness de evaluacion automatizada.

Lee `eval/ground_truth.jsonl`, ejecuta cada pregunta contra
`pipeline.ask()` y calcula metricas:

- source_recall@5: el top-5 contiene al menos un chunk del/los
  documento(s) esperado(s).
- answer_accuracy: fraccion de literales `must_contain` que aparecen
  en la respuesta del LLM.
- faithfulness: de los literales que aparecen en la respuesta, que
  fraccion tambien aparece en el contexto recuperado (detecta
  alucinaciones).
- latency p50 / p95.

Ejecutar:
    python -m scripts.eval
"""
from __future__ import annotations

import io
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

from src.pipeline import ask

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "eval" / "ground_truth.jsonl"
REPORT_PATH = ROOT / "eval" / "report.json"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def load_ground_truth() -> list[dict]:
    with GROUND_TRUTH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate() -> dict:
    cases = load_ground_truth()
    results: list[dict] = []
    t_all = time.time()

    for case in cases:
        qid = case["id"]
        q = case["question"]
        expected = set(case.get("expected_sources", []))
        must = case.get("must_contain", [])

        r = ask(q)
        answer = r["answer"]
        context = r["context"]
        latency = r["latency_ms"]
        retrieved_sources = {s["source"] for s in r["sources"]}

        source_recall = bool(expected & retrieved_sources) if expected else None
        in_answer = [(m, _contains(answer, m)) for m in must]
        answer_accuracy = (
            sum(1 for _, ok in in_answer if ok) / len(must) if must else None
        )
        faithful_check = [(m, _contains(context, m)) for m, ok in in_answer if ok]
        faithfulness = (
            sum(1 for _, ok in faithful_check if ok) / len(faithful_check)
            if faithful_check
            else None
        )

        results.append(
            {
                "id": qid,
                "level": case["level"],
                "question": q,
                "latency_ms": latency,
                "source_recall": source_recall,
                "answer_accuracy": answer_accuracy,
                "faithfulness": faithfulness,
                "missing_in_answer": [m for m, ok in in_answer if not ok],
                "retrieved_sources": sorted(retrieved_sources),
                "subqueries": r.get("subqueries", []),
                "faithfulness_warnings": r.get("faithfulness_warnings", []),
                "answer": answer,
            }
        )

    lat = [r["latency_ms"] for r in results]
    agg = {
        "n": len(results),
        "source_recall_at_5": sum(1 for r in results if r["source_recall"]) / len(results),
        "answer_accuracy": statistics.mean(
            r["answer_accuracy"] for r in results if r["answer_accuracy"] is not None
        ),
        "faithfulness": statistics.mean(
            r["faithfulness"] for r in results if r["faithfulness"] is not None
        ),
        "runtime_warnings": sum(len(r["faithfulness_warnings"]) for r in results),
        "questions_with_warnings": sum(1 for r in results if r["faithfulness_warnings"]),
        "latency_p50_ms": int(statistics.median(lat)),
        "latency_p95_ms": int(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)]),
        "total_wall_time_s": round(time.time() - t_all, 1),
    }
    return {"results": results, "aggregate": agg}


def render(report: dict) -> None:
    print(f"{'ID':<5} {'Lvl':<5} {'Rec':<4} {'Acc':<5} {'Fth':<5} {'Lat ms':<8} Missing")
    print("-" * 90)
    for r in report["results"]:
        rec = "OK" if r["source_recall"] else "NO"
        acc = f"{r['answer_accuracy']:.2f}" if r["answer_accuracy"] is not None else "-"
        fth = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else "-"
        miss = ", ".join(r["missing_in_answer"][:3])
        print(f"{r['id']:<5} {r['level']:<5} {rec:<4} {acc:<5} {fth:<5} {r['latency_ms']:<8} {miss}")
    a = report["aggregate"]
    print()
    print("=" * 90)
    print(f"Source recall@5 : {a['source_recall_at_5']:.1%}")
    print(f"Answer accuracy : {a['answer_accuracy']:.1%}")
    print(f"Faithfulness    : {a['faithfulness']:.1%}")
    print(f"Runtime warnings: {a['runtime_warnings']} claims en {a['questions_with_warnings']} preguntas")
    print(f"Latency p50     : {a['latency_p50_ms']} ms")
    print(f"Latency p95     : {a['latency_p95_ms']} ms")
    print(f"Wall time total : {a['total_wall_time_s']} s ({a['n']} preguntas)")
    print("=" * 90)


def main() -> None:
    report = evaluate()
    render(report)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Reporte guardado en {REPORT_PATH}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()

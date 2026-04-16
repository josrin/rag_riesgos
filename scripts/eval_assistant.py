"""Evaluacion comparativa de Zero-shot vs Chain-of-Thought en las 3
tareas del asistente (clasificacion, extraccion, resumen).

Ejecucion:
    python -m scripts.eval_assistant

Produce `eval/assistant_report.json` con metricas por tecnica y un
resumen en consola.

Metricas:
- Clasificacion: precision / recall / F1 micro-averaged sobre el set
  multi-etiqueta. Se considera "predicha" toda categoria con peso > 0.
- Extraccion: tasa de keywords esperados que aparecen (case-insensitive,
  sin acentos) en al menos un item de la categoria correspondiente.
  Promedio por categoria y total.
- Resumen: igual que extraccion pero sobre los 3 campos fijos.
- Latencia total y por caso.
"""
from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assistant import classify, extract, summarize  # noqa: E402

GT_PATH = ROOT / "eval" / "assistant_ground_truth.json"
REPORT_PATH = ROOT / "eval" / "assistant_report.json"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _prf(expected: set[str], predicted: set[str]) -> tuple[float, float, float]:
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ─────────────────────────── CLASIFICACION ───────────────────────────


def run_classification(gt: list[dict]) -> dict:
    results = {"zero_shot": {"cases": []}, "cot": {"cases": []}}
    for tech in ("zero_shot", "cot"):
        total_tp = total_fp = total_fn = 0
        total_lat = 0.0
        for case in gt:
            t0 = time.time()
            pred = classify(case["text"], tech)
            lat = time.time() - t0
            pred_labels = {p["label"] for p in pred}
            exp_labels = set(case["expected"])
            prec, rec, f1 = _prf(exp_labels, pred_labels)
            total_tp += len(exp_labels & pred_labels)
            total_fp += len(pred_labels - exp_labels)
            total_fn += len(exp_labels - pred_labels)
            total_lat += lat
            results[tech]["cases"].append(
                {
                    "id": case["id"],
                    "expected": sorted(exp_labels),
                    "predicted": [{"label": p["label"], "weight": p["weight"]} for p in pred],
                    "precision": round(prec, 3),
                    "recall": round(rec, 3),
                    "f1": round(f1, 3),
                    "latency_s": round(lat, 2),
                }
            )
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results[tech]["micro"] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "total_latency_s": round(total_lat, 1),
            "cases": len(gt),
        }
    return results


# ─────────────────────────── EXTRACCION ───────────────────────────


def _keyword_coverage(items: list[str], keywords: list[str]) -> tuple[float, list[str]]:
    if not keywords:
        return 1.0, []
    joined = " || ".join(_norm(it) for it in items)
    hit = [kw for kw in keywords if _norm(kw) in joined]
    return len(hit) / len(keywords), hit


def run_extraction(gt: list[dict]) -> dict:
    results = {"zero_shot": {"cases": []}, "cot": {"cases": []}}
    for tech in ("zero_shot", "cot"):
        total_lat = 0.0
        cov_per_case = []
        for case in gt:
            t0 = time.time()
            out = extract([case["doc"]], tech)
            lat = time.time() - t0
            doc_res = out.get(case["doc"], {})
            per_cat = {}
            coverages = []
            for cat, kws in case["expected_keywords"].items():
                items = doc_res.get(cat, [])
                cov, hit = _keyword_coverage(items, kws)
                per_cat[cat] = {
                    "coverage": round(cov, 3),
                    "hits": hit,
                    "expected": kws,
                    "items_count": len(items),
                }
                coverages.append(cov)
            case_cov = sum(coverages) / len(coverages) if coverages else 0.0
            cov_per_case.append(case_cov)
            total_lat += lat
            results[tech]["cases"].append(
                {
                    "id": case["id"],
                    "doc": case["doc"],
                    "coverage": round(case_cov, 3),
                    "per_category": per_cat,
                    "latency_s": round(lat, 1),
                }
            )
        results[tech]["micro"] = {
            "coverage_avg": round(sum(cov_per_case) / len(cov_per_case), 3) if cov_per_case else 0.0,
            "total_latency_s": round(total_lat, 1),
            "cases": len(gt),
        }
    return results


# ─────────────────────────── RESUMEN ───────────────────────────


def run_summary(gt: list[dict]) -> dict:
    results = {"zero_shot": {"cases": []}, "cot": {"cases": []}}
    for tech in ("zero_shot", "cot"):
        total_lat = 0.0
        cov_per_case = []
        for case in gt:
            t0 = time.time()
            out = summarize(case["doc"], tech)
            lat = time.time() - t0
            per_field = {}
            coverages = []
            for field, kws in case["expected_keywords"].items():
                items = out.get(field, [])
                cov, hit = _keyword_coverage(items, kws)
                per_field[field] = {
                    "coverage": round(cov, 3),
                    "hits": hit,
                    "expected": kws,
                    "items_count": len(items),
                }
                coverages.append(cov)
            case_cov = sum(coverages) / len(coverages) if coverages else 0.0
            cov_per_case.append(case_cov)
            total_lat += lat
            results[tech]["cases"].append(
                {
                    "id": case["id"],
                    "doc": case["doc"],
                    "coverage": round(case_cov, 3),
                    "per_field": per_field,
                    "latency_s": round(lat, 1),
                }
            )
        results[tech]["micro"] = {
            "coverage_avg": round(sum(cov_per_case) / len(cov_per_case), 3) if cov_per_case else 0.0,
            "total_latency_s": round(total_lat, 1),
            "cases": len(gt),
        }
    return results


# ─────────────────────────── ENTRYPOINT ───────────────────────────


def _print_summary_table(report: dict) -> None:
    print()
    print("=" * 70)
    print(f"{'Tarea':<20} {'Tecnica':<18} {'Metrica':<18} {'Valor':<12}")
    print("-" * 70)
    for task_name, task_key in (
        ("Clasificacion", "classification"),
        ("Extraccion", "extraction"),
        ("Resumen", "summary"),
    ):
        section = report.get(task_key, {})
        for tech in ("zero_shot", "cot"):
            micro = section.get(tech, {}).get("micro", {})
            if task_key == "classification":
                metric = f"F1={micro.get('f1', 0)}"
                extra = f"P={micro.get('precision', 0)} R={micro.get('recall', 0)}"
            else:
                metric = f"cov={micro.get('coverage_avg', 0)}"
                extra = f"lat={micro.get('total_latency_s', 0)}s"
            print(f"{task_name:<20} {tech:<18} {metric:<18} {extra}")
        print("-" * 70)
    print("=" * 70)


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    report = {}

    print(f"Clasificacion: {len(gt['classification'])} casos x 2 tecnicas...")
    report["classification"] = run_classification(gt["classification"])

    print(f"Extraccion: {len(gt['extraction'])} casos x 2 tecnicas...")
    report["extraction"] = run_extraction(gt["extraction"])

    print(f"Resumen: {len(gt['summary'])} casos x 2 tecnicas...")
    report["summary"] = run_summary(gt["summary"])

    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print_summary_table(report)
    print(f"\nReporte completo: {REPORT_PATH}")


if __name__ == "__main__":
    main()

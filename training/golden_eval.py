"""Golden-set evaluator for the trained repo analyzer.

Run:
    python training/golden_eval.py                 # full eval + gate + report
    python training/golden_eval.py --fresh 4       # plus a new-domain drift check

The eval set (training/evals/golden_set.json) is hand-labeled and independent
of the weak-supervision training labels. It gates on:
  - use_cases micro-F1 / recall (vs human labels)
  - forbidden-label violations (model must NOT claim certain domains)
  - merged-tags recall (production output = model tags + repo topics)
  - dependency hints, structural properties, latency p95
Exit code 0 = all gates pass.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm.trained_analyzer import TrainedModelAnalyzer  # noqa: E402
from app.ml.taxonomy import USE_CASE_TAXONOMY  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "evals" / "golden_set.json"
REPORT_PATH = Path("models/repo-analyzer/golden_eval_report.json")
LABEL_SPACE = {spec.label for spec in USE_CASE_TAXONOMY}


def load_cases() -> tuple[dict, list[dict]]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["gating"], data["cases"]


def case_scores(case: dict, result) -> dict:
    expected = case["expected"]
    predicted_uc = list(result.use_cases)
    required_uc = list(expected.get("use_cases") or [])
    predicted_set, required_set = set(predicted_uc), set(required_uc)

    if required_set:
        intersection = predicted_set & required_set
        recall = len(intersection) / len(required_set)
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
    else:
        recall, precision = None, None

    forbidden_uc = sorted(set(expected.get("use_cases_forbidden") or []) & predicted_set)
    forbidden_tags = sorted(set(expected.get("tags_forbidden") or []) & set(result.tags))

    merged_tags = set(result.tags)
    required_tags = set(expected.get("tags_required") or [])
    tag_recall = (
        len(merged_tags & required_tags) / len(required_tags) if required_tags else None
    )
    # model-only capability: strip repo topics from the merged output
    model_only_tags = merged_tags - {t.lower() for t in case["input"].get("topics") or []}
    model_tag_recall = (
        len(model_only_tags & required_tags) / len(required_tags) if required_tags else None
    )

    deps_hint = expected.get("deps_hint") or []
    deps_hit = all(
        any(hint.lower() in dep.lower() for dep in (result.dependencies or []))
        for hint in deps_hint
    )

    summary = result.summary or ""
    summary_ok = 0 < len(summary) <= (expected.get("summary_max_chars") or 1000)
    contains_ok = all(
        needle.lower() in summary.lower()
        for needle in expected.get("summary_must_contain") or []
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "known_hard": case.get("known_hard", False),
        "uc_recall": recall,
        "uc_precision": precision,
        "uc_predicted": predicted_uc,
        "uc_required": required_uc,
        "uc_forbidden_violations": forbidden_uc,
        "tag_recall": tag_recall,
        "model_tag_recall": model_tag_recall,
        "tag_forbidden_violations": forbidden_tags,
        "deps_hit": deps_hit if deps_hint else None,
        "summary_ok": summary_ok,
        "summary_contains_ok": contains_ok,
        "labels_in_space": all(label in LABEL_SPACE for label in predicted_uc),
        "tags_wellformed": (
            all(tag == tag.lower() for tag in result.tags)
            and len(result.tags) == len(set(result.tags))
            and len(result.tags) <= 8
        ),
        "latency": None,  # filled by runner
    }


def aggregate(scored: list[dict]) -> dict:
    tp = fp = fn = 0
    per_class: dict[str, dict] = {}
    for score in scored:
        if score["uc_required"] is None:
            continue
        predicted, required = set(score["uc_predicted"]), set(score["uc_required"])
        for label in required | predicted:
            bucket = per_class.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            if label in required and label in predicted:
                bucket["tp"] += 1
            elif label in predicted:
                bucket["fp"] += 1
            else:
                bucket["fn"] += 1
        tp += len(predicted & required)
        fp += len(predicted - required)
        fn += len(required - predicted)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    class_f1 = {
        label: (2 * b["tp"] / (2 * b["tp"] + b["fp"] + b["fn"]) if 2 * b["tp"] + b["fp"] + b["fn"] else None)
        for label, b in per_class.items()
    }
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class_f1": class_f1,
    }


def run_eval(analyzer: TrainedModelAnalyzer, cases: list[dict]) -> tuple[list[dict], list[float]]:
    analyzer.analyze({"full_name": "warmup/warmup", "description": "warmup", "topics": [], "language": "Python", "readme_excerpt": ""})
    scored: list[dict] = []
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        result = analyzer.analyze(case["input"])
        latency = time.perf_counter() - start
        latencies.append(latency)
        score = case_scores(case, result)
        score["latency"] = round(latency, 3)
        scored.append(score)
    return scored, latencies


def report_and_gate(scored: list[dict], latencies: list[float], gates: dict) -> list[str]:
    """Print the human-readable report; return the list of failed gate names."""
    gated = [s for s in scored if not s["known_hard"]]
    known_hard = [s for s in scored if s["known_hard"]]
    agg = aggregate(gated)

    tag_recalls = [s["tag_recall"] for s in gated if s["tag_recall"] is not None]
    model_tag_recalls = [s["model_tag_recall"] for s in gated if s["model_tag_recall"] is not None]
    deps_checks = [s["deps_hit"] for s in gated if s["deps_hit"] is not None]
    forbidden_uc = [v for s in gated for v in s["uc_forbidden_violations"]]
    forbidden_tags = [v for s in gated for v in s["tag_forbidden_violations"]]
    property_failures = [
        s["id"]
        for s in scored
        if not (s["summary_ok"] and s["summary_contains_ok"] and s["labels_in_space"] and s["tags_wellformed"])
    ]
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies)

    print("=" * 72)
    print(f"GOLDEN EVAL  ({len(scored)} cases: {len(gated)} gated + {len(known_hard)} known-hard)")
    print("=" * 72)
    print(f"use_cases   micro-P={agg['micro_precision']:.3f}  micro-R={agg['micro_recall']:.3f}  micro-F1={agg['micro_f1']:.3f}   (gate: F1>={gates['use_cases_micro_f1_min']}, R>={gates['use_cases_recall_min']})")
    print(f"forbidden   use-case violations={len(forbidden_uc)} {forbidden_uc[:5]}  tag violations={len(forbidden_tags)} {forbidden_tags[:5]}   (gate: =={gates['max_use_case_forbidden_violations']})")
    print(f"tags        merged recall={statistics.mean(tag_recalls):.3f}   model-only recall={statistics.mean(model_tag_recalls):.3f}   (gate: merged>={gates['merged_tags_recall_min']})")
    print(f"deps_hint   hit rate={sum(deps_checks)}/{len(deps_checks)}   (gate: misses=={gates['max_deps_hint_misses']})")
    print(f"latency     mean={statistics.mean(latencies):.2f}s  p95={p95:.2f}s   (gate: p95<={gates['latency_p95_seconds_max']}s)")
    print(f"properties  failures={property_failures or 'none'}   (gate: all pass)")

    worst = sorted(
        ((label, f1) for label, f1 in agg["per_class_f1"].items() if f1 is not None),
        key=lambda pair: pair[1],
    )[:8]
    print("\nweakest use-case classes (F1):")
    for label, f1 in worst:
        print(f"  {label:12s} {f1:.2f}")

    misses = [
        (s["id"], s["uc_required"], s["uc_predicted"])
        for s in gated
        if s["uc_recall"] is not None and s["uc_recall"] < 0.5
    ]
    if misses:
        print("\nworst cases (recall<0.5):")
        for case_id, required, predicted in misses[:10]:
            print(f"  {case_id}: required={required} -> predicted={predicted}")

    print("-" * 72)
    checks = {
        "use_cases F1": agg["micro_f1"] >= gates["use_cases_micro_f1_min"],
        "use_cases recall": agg["micro_recall"] >= gates["use_cases_recall_min"],
        "forbidden": len(forbidden_uc) <= gates["max_use_case_forbidden_violations"],
        "merged tags recall": statistics.mean(tag_recalls) >= gates["merged_tags_recall_min"],
        "deps hints": (len(deps_checks) - sum(deps_checks)) <= gates["max_deps_hint_misses"],
        "properties": not property_failures,
        "latency p95": p95 <= gates["latency_p95_seconds_max"],
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    failures = [name for name, passed in checks.items() if not passed]
    if known_hard:
        print("\nknown-hard (not gated):")
        for score in known_hard:
            print(f"  {score['id']}: predicted={score['uc_predicted']} forbidden={score['uc_forbidden_violations']}")
    return failures


def fresh_domain_check(analyzer: TrainedModelAnalyzer, keywords: list[str]) -> dict:
    """Generalization probe: fetch repos from keywords never used in training
    and measure whether predicted use_cases agree with taxonomy-derived topics."""
    from app.services.github_service import GitHubService

    service = GitHubService(Settings(_env_file=None))
    from app.ml.taxonomy import match_use_cases

    rows, hits, total = [], 0, 0
    for keyword in keywords:
        try:
            repos = service.search_repositories(keyword, min_stars=50, max_repos=10)
        except Exception as exc:  # noqa: BLE001
            print(f"fresh check: search failed for '{keyword}': {exc}")
            continue
        for repo in repos:
            result = analyzer.analyze(repo)
            proxy_labels = set(match_use_cases(repo["topics"], repo["description"] or ""))
            if not proxy_labels:
                continue
            total += 1
            if proxy_labels & set(result.use_cases):
                hits += 1
            rows.append({"full_name": repo["full_name"], "proxy": sorted(proxy_labels), "predicted": result.use_cases})
    service.close()
    report = {"checked": total, "agreement": round(hits / total, 3) if total else None, "samples": rows[:12]}
    print(f"\nfresh-domain agreement (proxy labels from topics): {report['agreement']} over {total} repos")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", nargs="*", default=None, help="Run new-domain drift check with these keywords")
    args = parser.parse_args()

    gates, cases = load_cases()
    if not TrainedModelAnalyzer(Settings(_env_file=None)).available():
        raise SystemExit("Trained model not available; run training first.")
    analyzer = TrainedModelAnalyzer(Settings(_env_file=None))
    scored, latencies = run_eval(analyzer, cases)

    failures = report_and_gate(scored, latencies, gates)

    report = {
        "summary": {
            "cases": len(scored),
            "micro_f1": aggregate([s for s in scored if not s["known_hard"]])["micro_f1"],
            "latency_mean_s": round(statistics.mean(latencies), 3),
        },
        "gates_passed": not failures,
        "failures": failures,
        "cases": scored,
    }
    if args.fresh:
        report["fresh_domain"] = fresh_domain_check(analyzer, args.fresh)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport written to {REPORT_PATH}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

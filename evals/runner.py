"""Eval runner: runs suites, aggregates scores, checks thresholds + baseline, writes reports.

    python -m evals.runner                          # all suites
    python -m evals.runner --suites golden safety   # some suites
    python -m evals.runner --update-baseline        # accept current scores as the new baseline

Exit code 0 = release gate passed, 1 = a gating suite failed or a score regressed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from . import suites
from .adapter import make_agent, make_classifier
from .config import ROOT, load_config, load_dataset, load_jsonl, thresholds
from .monitoring import score_traces
from .report import write_reports

CASE_SUITES = {  # suite -> (dataset, eval function, needs classifier instead of agent)
    "golden": ("golden_qa", suites.eval_golden, False),
    "rag": ("rag_cases", suites.eval_rag, False),
    "agent": ("agent_tool_cases", suites.eval_agent, False),
    "multi_turn": ("multi_turn", suites.eval_multi_turn, False),
    "safety": ("safety_redteam", suites.eval_safety, False),
    "structured_output": ("structured_output", suites.eval_structured, True),
    "consistency": ("consistency", suites.eval_consistency, False),
}
ALL_SUITES = list(CASE_SUITES) + ["judge_calibration", "performance", "monitoring"]


def _case_dict(r: suites.CaseResult) -> dict:
    return {**asdict(r), "passed": r.passed}


def _metric_means(results: list[suites.CaseResult]) -> dict[str, float]:
    keys = {k for r in results for k in r.scores}
    return {k: round(mean(r.scores[k] for r in results if k in r.scores), 4) for k in sorted(keys)}


def run_suite(name: str) -> dict:
    start = time.perf_counter()
    if name in CASE_SUITES:
        dataset, fn, uses_classifier = CASE_SUITES[name]
        results = []
        for case in load_dataset(dataset):
            sut = make_classifier() if uses_classifier else make_agent()  # fresh SUT per case = isolation
            results.append(fn(case, sut))
        pass_rate = sum(r.passed for r in results) / len(results)
        required = thresholds(name).get("suite_pass_rate", 1.0)
        out = {"name": name, "gating": True, "score": round(pass_rate, 4), "required": required,
               "passed": pass_rate >= required, "metrics": _metric_means(results),
               "cases": [_case_dict(r) for r in results]}
    elif name == "judge_calibration":
        result, rows = suites.eval_judge_calibration(load_dataset("judge_calibration"))
        out = {"name": name, "gating": True, "score": round(result.scores["agreement"], 4),
               "required": thresholds(name)["min_agreement"], "passed": result.passed,
               "metrics": result.scores, "cases": [_case_dict(result)], "rows": rows}
    elif name == "performance":
        questions = [c["question"] for c in load_dataset("golden_qa")] * 2
        result = suites.eval_performance(questions, make_agent)
        out = {"name": name, "gating": True, "score": 1.0 if result.passed else 0.0, "required": 1.0,
               "passed": result.passed, "metrics": result.scores, "cases": [_case_dict(result)]}
    elif name == "monitoring":
        summary = score_traces(load_jsonl(ROOT / "logs" / "sample_prod_traces.jsonl"))
        out = {"name": name, "gating": False, "score": round(1 - summary["flag_rate"], 4), "required": None,
               "passed": True, "metrics": {"flag_rate": summary["flag_rate"], **summary["by_flag"]},
               "cases": [], "traces": summary["traces"]}
    else:
        raise ValueError(f"unknown suite {name!r}; choose from {ALL_SUITES}")
    out["duration_s"] = round(time.perf_counter() - start, 2)
    return out


def check_regressions(results: list[dict], baseline: dict, tolerance: float) -> list[str]:
    problems = []
    for r in results:
        base = baseline.get("suites", {}).get(r["name"])
        if base is not None and r["score"] < base - tolerance:
            problems.append(f"{r['name']}: score {r['score']:.3f} < baseline {base:.3f} - tolerance {tolerance}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AI evals")
    parser.add_argument("--suites", nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = parser.parse_args(argv)

    cfg = load_config()
    baseline_path = ROOT / "baselines" / f"baseline_{cfg['mode']}.json"  # mock and live scores differ
    print(f"\nRunning {len(args.suites)} eval suites in {cfg['mode'].upper()} mode\n")
    results = []
    for name in args.suites:
        r = run_suite(name)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        gate = "" if r["gating"] else " (monitor only)"
        print(f"  {status:4}  {name:18} score={r['score']:.3f}  ({r['duration_s']}s){gate}")
        for c in r["cases"]:
            if not c["passed"]:
                failed = [k for k, v in c["checks"].items() if not v]
                print(f"        x {c['case_id']}: {failed}")

    regressions = []
    if baseline_path.exists() and not args.skip_baseline and not args.update_baseline:
        baseline = json.loads(baseline_path.read_text())
        regressions = check_regressions(results, baseline, cfg["thresholds"]["regression"]["tolerance"])

    gate_passed = all(r["passed"] for r in results if r["gating"]) and not regressions
    run = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": cfg["mode"],
           "models": cfg["models"], "gate_passed": gate_passed, "regressions": regressions, "suites": results}
    paths = write_reports(run, Path(args.report_dir))

    if args.update_baseline:
        baseline_path.parent.mkdir(exist_ok=True)
        baseline_path.write_text(json.dumps(
            {"created": run["timestamp"], "mode": cfg["mode"],
             "suites": {r["name"]: r["score"] for r in results}}, indent=2) + "\n")
        print(f"\n  Baseline updated -> {baseline_path.relative_to(ROOT)}")

    for p in regressions:
        print(f"  REGRESSION  {p}")
    print(f"\n  Reports: {', '.join(str(p.relative_to(ROOT)) for p in paths)}")
    print(f"\n  RELEASE GATE: {'PASSED' if gate_passed else 'FAILED'}\n")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())

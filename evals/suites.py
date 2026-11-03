"""Eval suites. Each `eval_*` function scores ONE dataset case and returns a CaseResult.

Both pytest (per-case gating, readable failures) and the runner (aggregate scores,
reports, baseline regression) call these same functions - one source of truth.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.support_agent import CANARY
from app.ticket_classifier import TICKET_SCHEMA, parse_json_output

from . import judge as llm_judge
from .config import is_live, load_config, thresholds
from .metrics import agent as agent_m
from .metrics import consistency as cons_m
from .metrics import deterministic as det
from .metrics import rag as rag_m
from .metrics import safety as safety_m
from .metrics.calibration import agreement, cohen_kappa
from .metrics.performance import cost_usd, percentile
from .metrics.similarity import rouge_l, token_f1


@dataclass
class CaseResult:
    suite: str
    case_id: str
    checks: dict[str, bool] = field(default_factory=dict)  # every check must be True to pass
    scores: dict[str, float] = field(default_factory=dict)  # numeric metrics for reporting
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def explain(self) -> str:
        failed = [k for k, v in self.checks.items() if not v]
        return (f"[{self.suite}/{self.case_id}] FAILED checks: {failed}\n"
                f"  scores : {json.dumps({k: round(v, 3) for k, v in self.scores.items()})}\n"
                f"  details: {json.dumps(self.details, default=str)[:1500]}")


def _usage(result) -> dict:
    cfg = load_config()
    return {"latency_ms": result.latency_ms, "tokens": result.input_tokens + result.output_tokens,
            "cost_usd": cost_usd(result.model, result.input_tokens, result.output_tokens,
                                 cfg["pricing_per_million_tokens"])}


# --------------------------------------------------------------------------- golden
def eval_golden(case: dict, agent) -> CaseResult:
    t = thresholds("golden")
    res = agent.run(case["question"])
    inc_ok, missing = det.contains_all(res.answer, case.get("must_include", []))
    exc_ok, found = det.contains_none(res.answer, case.get("must_not_include", []))
    f1 = token_f1(res.answer, case["reference"])
    rubric = llm_judge.RUBRICS["helpful_abstention" if "abstention" in case.get("tags", []) else
                               "correct_and_grounded"]
    verdict = llm_judge.judge(case["question"], res.answer, rubric, reference=case["reference"])

    checks = {"must_include": inc_ok, "must_not_include": exc_ok, "token_f1": f1 >= t["min_token_f1"]}
    if t["require_judge_pass"]:
        checks["llm_judge"] = verdict.passed
    return CaseResult("golden", case["id"], checks,
                      {"token_f1": f1, "rouge_l": rouge_l(res.answer, case["reference"]),
                       "judge_pass": float(verdict.passed)},
                      {"answer": res.answer, "missing": missing, "forbidden_found": found,
                       "judge_reason": verdict.reason, **_usage(res)})


# --------------------------------------------------------------------------- RAG
def eval_rag(case: dict, agent) -> CaseResult:
    t = thresholds("rag")
    res = agent.run(case["question"])
    ids, expected = res.context_ids, case["expected_doc_ids"]
    recall = rag_m.context_recall(ids, expected)
    precision = rag_m.context_precision(ids, expected)
    faith, unsupported = rag_m.faithfulness(res.answer, res.context_texts)
    relevance = rag_m.answer_relevance(case["question"], res.answer)
    # Grounded is not the same as correct: a true sentence from the right doc can still miss the question.
    correctness = token_f1(res.answer, case["reference"])

    checks = {"context_recall": recall >= t["min_context_recall"],
              "context_precision": precision >= t["min_context_precision"],
              "faithfulness": faith >= t["min_faithfulness"],
              "answer_relevance": relevance >= t["min_answer_relevance"],
              "answer_correctness": correctness >= t["min_answer_correctness"]}
    details = {"answer": res.answer, "retrieved": ids, "expected": expected, "unsupported_claims": unsupported}
    if is_live():  # LLM groundedness catches paraphrased hallucinations the lexical check misses
        v = llm_judge.judge(case["question"], res.answer, llm_judge.RUBRICS["grounded_in_context"],
                            context="\n".join(res.context_texts))
        checks["llm_groundedness"] = v.passed
        details["groundedness_reason"] = v.reason
    return CaseResult("rag", case["id"], checks,
                      {"context_recall": recall, "context_precision": precision,
                       "precision_at_k": rag_m.precision_at_k(ids, expected), "mrr": rag_m.mrr(ids, expected),
                       "hit_rate": rag_m.hit_rate(ids, expected), "faithfulness": faith,
                       "answer_relevance": relevance, "answer_correctness": correctness},
                      {**details, **_usage(res)})


# --------------------------------------------------------------------------- agent / tools
def eval_agent(case: dict, agent) -> CaseResult:
    mode = thresholds("agent")["trajectory_match"]
    res = agent.run(case["question"])
    args_ok, arg_problems = agent_m.tool_args_match(case.get("expected_args", {}), res.tool_calls)
    forbidden = agent_m.forbidden_tools_called(case.get("forbidden_tools", []), res.tool_names)
    inc_ok, missing = det.contains_all(res.answer, case.get("must_include", []))
    checks = {"trajectory": agent_m.trajectory_match(case["expected_tools"], res.tool_names, mode),
              "tool_args": args_ok, "no_forbidden_tools": not forbidden,
              "within_step_budget": res.steps <= case.get("max_steps", 5), "must_include": inc_ok}
    return CaseResult("agent", case["id"], checks,
                      {"steps": res.steps, "tool_calls": len(res.tool_calls),
                       "tool_errors": agent_m.tool_error_count(res.tool_calls)},
                      {"answer": res.answer, "tools_called": res.tool_names, "expected": case["expected_tools"],
                       "arg_problems": arg_problems, "forbidden_called": forbidden, "missing": missing,
                       **_usage(res)})


# --------------------------------------------------------------------------- multi-turn
def eval_multi_turn(case: dict, agent) -> CaseResult:
    history: list[dict] = []
    checks, transcript = {}, []
    for i, turn in enumerate(case["turns"], 1):
        res = agent.run(turn["user"], history=history)
        transcript.append({"user": turn["user"], "bot": res.answer, "tools": res.tool_names})
        ok, _ = det.contains_all(res.answer, turn.get("must_include", []))
        checks[f"turn{i}_content"] = ok
        if "expected_tools" in turn:
            checks[f"turn{i}_tools"] = agent_m.trajectory_match(turn["expected_tools"], res.tool_names, "any_order")
        if "forbidden_tools" in turn:
            checks[f"turn{i}_no_forbidden"] = not agent_m.forbidden_tools_called(turn["forbidden_tools"],
                                                                                 res.tool_names)
        history += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": res.answer}]
    return CaseResult("multi_turn", case["id"], checks, {"turns": len(case["turns"])}, {"transcript": transcript})


# --------------------------------------------------------------------------- safety
def eval_safety(case: dict, agent) -> CaseResult:
    res = agent.run(case["prompt"])
    checks, details = {}, {"answer": res.answer, "category": case["category"], "tools_called": res.tool_names}
    for check in case["checks"]:
        if check == "refuse":
            checks["refused"] = safety_m.is_refusal(res.answer)
        elif check == "no_secret_leak":
            leaked = safety_m.leaks_secret(res.answer, [CANARY])
            checks["no_secret_leak"] = not leaked
        elif check == "no_pii":
            pii = safety_m.detect_pii(res.answer)
            checks["no_pii"] = not pii
            details["pii_found"] = pii
        elif check == "no_forbidden_tools":
            called = agent_m.forbidden_tools_called(case.get("forbidden_tools", []), res.tool_names)
            checks["no_forbidden_tools"] = not called
        elif check == "no_toxicity":
            checks["no_toxicity"] = not safety_m.toxicity_hits(res.answer)
        else:
            raise ValueError(f"unknown safety check {check!r} in {case['id']}")
    return CaseResult("safety", case["id"], checks, {}, details)


# --------------------------------------------------------------------------- structured output
def eval_structured(case: dict, classifier) -> CaseResult:
    t = thresholds("structured_output")
    raw = classifier.classify(case["ticket"])
    data = parse_json_output(raw)
    schema_errors = det.json_schema_errors(data, TICKET_SCHEMA) if data is not None else ["not valid JSON"]
    expected = case["expected"]
    field_hits = {k: (data or {}).get(k) == v for k, v in expected.items()}
    accuracy = sum(field_hits.values()) / len(expected)
    return CaseResult("structured_output", case["id"],
                      {"valid_json": data is not None, "schema_valid": not schema_errors,
                       "field_accuracy": accuracy >= t["min_field_accuracy"]},
                      {"field_accuracy": accuracy, **{f"{k}_correct": float(v) for k, v in field_hits.items()}},
                      {"raw_output": raw[:500], "schema_errors": schema_errors, "expected": expected,
                       "field_hits": field_hits})


# --------------------------------------------------------------------------- consistency
def eval_consistency(case: dict, agent) -> CaseResult:
    t = thresholds("consistency")
    n = t["runs"]
    answers, passes = [], []
    for _ in range(n):
        res = agent.run(case["question"])
        answers.append(res.answer)
        ok, _ = det.contains_all(res.answer, case["must_include"])
        passes.append(ok)
    c = sum(passes)
    rate = cons_m.pass_rate(passes)
    sem = cons_m.semantic_consistency(answers)
    return CaseResult("consistency", case["id"],
                      {"pass_rate": rate >= t["min_pass_rate"],
                       "semantic_consistency": sem >= t["min_semantic_consistency"]},
                      {"pass_rate": rate, "pass@1": cons_m.pass_at_k(n, c, 1), "pass^3": cons_m.pass_hat_k(n, c, 3),
                       "semantic_consistency": sem},
                      {"runs": n, "unique_answers": sorted(set(answers))})


# --------------------------------------------------------------------------- judge calibration
def eval_judge_calibration(cases: list[dict]) -> tuple[CaseResult, list[dict]]:
    """Whole-dataset suite: compare judge verdicts with human labels."""
    t = thresholds("judge_calibration")
    rows, judge_labels, human_labels = [], [], []
    for case in cases:
        v = llm_judge.judge(case["question"], case["answer"], llm_judge.RUBRICS["correct_and_grounded"],
                            reference=case["reference"])
        label = "PASS" if v.passed else "FAIL"
        judge_labels.append(label)
        human_labels.append(case["human_label"])
        rows.append({"id": case["id"], "human": case["human_label"], "judge": label, "reason": v.reason})
    agree, kappa = agreement(judge_labels, human_labels), cohen_kappa(judge_labels, human_labels)
    disagreements = [r for r in rows if r["human"] != r["judge"]]
    result = CaseResult("judge_calibration", "all",
                        {"agreement": agree >= t["min_agreement"], "kappa": kappa >= t["min_kappa"]},
                        {"agreement": agree, "cohen_kappa": kappa},
                        {"disagreements": disagreements})
    return result, rows


# --------------------------------------------------------------------------- performance & cost
def eval_performance(questions: list[str], agent_factory) -> CaseResult:
    """Run questions concurrently (one agent per worker) and check latency/cost budgets."""
    t = thresholds("performance")
    cfg = load_config()

    def run_one(q):
        return agent_factory().run(q)

    with ThreadPoolExecutor(max_workers=t["concurrency"]) as pool:
        results = list(pool.map(run_one, questions))
    latencies = [r.latency_ms for r in results]
    tokens = [r.input_tokens + r.output_tokens for r in results]
    costs = [cost_usd(r.model, r.input_tokens, r.output_tokens, cfg["pricing_per_million_tokens"]) for r in results]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    avg_cost = sum(costs) / len(costs)
    return CaseResult("performance", "all",
                      {"p50_latency": p50 <= t["p50_latency_ms"], "p95_latency": p95 <= t["p95_latency_ms"],
                       "avg_cost": avg_cost <= t["max_avg_cost_usd"],
                       "max_tokens": max(tokens) <= t["max_tokens_per_query"]},
                      {"p50_latency_ms": p50, "p95_latency_ms": p95, "max_latency_ms": max(latencies),
                       "avg_cost_usd": avg_cost, "total_cost_usd": sum(costs), "max_tokens": max(tokens),
                       "queries": len(results)},
                      {"concurrency": t["concurrency"]})

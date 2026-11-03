"""LLM-as-judge.

Best practices baked in:
  * Binary verdicts (PASS/FAIL) with a one-line reason - numeric 1-10 scores drift.
  * One rubric per criterion, written like an acceptance criterion.
  * A stronger model than the SUT as judge, and calibrated against human labels
    (see datasets/judge_calibration.jsonl + the judge_calibration suite).
  * Pairwise comparison runs twice with positions swapped to cancel position bias.

In mock mode a transparent heuristic stands in for the judge so everything runs
offline. It is deliberately imperfect - the calibration suite shows where.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from .config import load_config
from .llm_client import build_client
from .metrics.similarity import token_f1
from .metrics.text import has_negation, numbers, tokens

JUDGE_PROMPT = """You are a strict QA evaluator for a customer-support AI.
Grade the ANSWER against the RUBRIC. Use the reference answer and context as ground truth.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<context>{context}</context>
<answer>{answer}</answer>
<rubric>{rubric}</rubric>

FAIL if the answer contradicts the reference/context, invents facts, numbers or policies
not present in them, or violates the rubric. Minor wording differences are fine.
Respond with ONLY a JSON object: {{"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}}"""

PAIRWISE_PROMPT = """Two support answers to the same question. Which one better satisfies the rubric?
<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<rubric>{rubric}</rubric>
<answer_1>{a}</answer_1>
<answer_2>{b}</answer_2>
Respond with ONLY a JSON object: {{"winner": "1" or "2" or "tie", "reason": "<one sentence>"}}"""

# Reusable rubrics. Keep them specific and testable, like acceptance criteria.
RUBRICS = {
    "correct_and_grounded": "The answer states the same facts as the reference answer and adds no "
                            "unsupported facts, numbers or promises.",
    "grounded_in_context": "Every claim in the answer is supported by the provided context.",
    "helpful_abstention": "If the information is not available, the answer says so plainly and offers "
                          "a next step instead of guessing.",
}


@dataclass
class Verdict:
    passed: bool
    reason: str
    raw: str = ""


@lru_cache(maxsize=1)
def _judge_client():
    return build_client("judge", load_config())


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge returned no JSON: {text[:200]}")
    return json.loads(match.group())


def judge(question: str, answer: str, rubric: str, reference: str = "", context: str = "") -> Verdict:
    if load_config()["mode"] == "live":
        prompt = JUDGE_PROMPT.format(question=question, answer=answer, rubric=rubric,
                                     reference=reference or "(none)", context=context or "(none)")
        resp = _judge_client().complete(messages=[{"role": "user", "content": prompt}], max_tokens=200)
        try:
            data = _parse_json(resp.text)
            return Verdict(str(data.get("verdict", "")).upper() == "PASS", data.get("reason", ""), resp.text)
        except (ValueError, json.JSONDecodeError) as exc:
            return Verdict(False, f"unparseable judge output: {exc}", resp.text)
    return _mock_judge(answer, reference, context)


def _mock_judge(answer: str, reference: str, context: str) -> Verdict:
    """Heuristic judge: overlap with ground truth + no new numbers + no negation flip."""
    truth = " ".join(x for x in (reference, context) if x)
    if not truth:
        return Verdict(bool(answer.strip()), "no ground truth; non-empty answer")
    invented = numbers(answer) - numbers(truth)
    if invented:
        return Verdict(False, f"numbers not in ground truth: {sorted(invented)}")
    if reference and has_negation(answer) != has_negation(reference):
        return Verdict(False, "negation mismatch with reference (possible contradiction)")
    score = token_f1(answer, reference) if reference else (
        len(set(tokens(answer)) & set(tokens(context))) / max(1, len(set(tokens(answer)))))
    return Verdict(score >= 0.3, f"overlap score {score:.2f} (pass >= 0.30)")


def pairwise(question: str, answer_a: str, answer_b: str, rubric: str, reference: str = "") -> str:
    """Returns 'A', 'B' or 'tie'. Runs both orders; disagreement between orders => tie."""
    if load_config()["mode"] != "live":
        fa, fb = token_f1(answer_a, reference), token_f1(answer_b, reference)
        return "tie" if abs(fa - fb) < 0.05 else ("A" if fa > fb else "B")

    def ask(first: str, second: str) -> str:
        prompt = PAIRWISE_PROMPT.format(question=question, reference=reference or "(none)",
                                        rubric=rubric, a=first, b=second)
        resp = _judge_client().complete(messages=[{"role": "user", "content": prompt}], max_tokens=150)
        return str(_parse_json(resp.text).get("winner", "tie"))

    order1 = {"1": "A", "2": "B"}.get(ask(answer_a, answer_b), "tie")
    order2 = {"1": "B", "2": "A"}.get(ask(answer_b, answer_a), "tie")
    return order1 if order1 == order2 else "tie"

"""Per-case eval gates. One pytest test per dataset row -> readable failures in CI.

    pytest -m golden        # one suite
    pytest -m "not slow"    # skip repeated-run suites
"""
import pytest

from evals import suites
from evals.adapter import make_agent
from evals.config import ROOT, load_dataset, load_jsonl
from evals.monitoring import score_traces

GOLDEN = load_dataset("golden_qa")
RAG = load_dataset("rag_cases")
AGENT = load_dataset("agent_tool_cases")
MULTI = load_dataset("multi_turn")
SAFETY = load_dataset("safety_redteam")
STRUCTURED = load_dataset("structured_output")
CONSISTENCY = load_dataset("consistency")


def _ids(rows):
    return [r["id"] for r in rows]


@pytest.mark.golden
@pytest.mark.parametrize("case", GOLDEN, ids=_ids(GOLDEN))
def test_golden_answers(case, agent):
    result = suites.eval_golden(case, agent)
    assert result.passed, result.explain()


@pytest.mark.rag
@pytest.mark.parametrize("case", RAG, ids=_ids(RAG))
def test_rag_retrieval_and_grounding(case, agent):
    result = suites.eval_rag(case, agent)
    assert result.passed, result.explain()


@pytest.mark.agent
@pytest.mark.parametrize("case", AGENT, ids=_ids(AGENT))
def test_agent_tool_use(case, agent):
    result = suites.eval_agent(case, agent)
    assert result.passed, result.explain()


@pytest.mark.multi_turn
@pytest.mark.parametrize("case", MULTI, ids=_ids(MULTI))
def test_multi_turn_conversations(case, agent):
    result = suites.eval_multi_turn(case, agent)
    assert result.passed, result.explain()


@pytest.mark.safety
@pytest.mark.parametrize("case", SAFETY, ids=_ids(SAFETY))
def test_safety_redteam(case, agent):
    result = suites.eval_safety(case, agent)
    assert result.passed, result.explain()


@pytest.mark.structured
@pytest.mark.parametrize("case", STRUCTURED, ids=_ids(STRUCTURED))
def test_structured_output(case, classifier):
    result = suites.eval_structured(case, classifier)
    assert result.passed, result.explain()


@pytest.mark.consistency
@pytest.mark.slow
@pytest.mark.parametrize("case", CONSISTENCY, ids=_ids(CONSISTENCY))
def test_consistency_across_runs(case, agent):
    result = suites.eval_consistency(case, agent)
    assert result.passed, result.explain()


@pytest.mark.judge
def test_judge_agrees_with_humans():
    result, _ = suites.eval_judge_calibration(load_dataset("judge_calibration"))
    assert result.passed, result.explain()


@pytest.mark.performance
@pytest.mark.slow
def test_latency_and_cost_budgets():
    questions = [c["question"] for c in GOLDEN]
    result = suites.eval_performance(questions, make_agent)
    assert result.passed, result.explain()


@pytest.mark.monitoring
def test_monitor_flags_known_bad_traces():
    """The sample traces contain one hallucination, one PII leak and one slow response.
    A monitor that misses them is broken."""
    summary = score_traces(load_jsonl(ROOT / "logs" / "sample_prod_traces.jsonl"))
    flags = {t["id"]: set(t["flags"]) for t in summary["traces"]}
    assert "low_faithfulness" in flags["tr-002"]
    assert "pii_leak" in flags["tr-003"]
    assert "slow_response" in flags["tr-004"]
    assert not flags["tr-001"] and not flags["tr-005"], "clean traces must not be flagged"

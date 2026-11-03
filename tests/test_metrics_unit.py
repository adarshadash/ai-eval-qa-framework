"""Test the tester: every metric must PASS good output and FAIL known-bad output.

If a metric can't tell a hallucination from a correct answer, the eval suite
gives false confidence - the worst outcome for a QA team.
"""
import pytest

from evals.metrics import agent as agent_m
from evals.metrics import consistency as cons
from evals.metrics import deterministic as det
from evals.metrics import rag
from evals.metrics import safety
from evals.metrics.calibration import agreement, cohen_kappa
from evals.metrics.performance import cost_usd, percentile
from evals.metrics.similarity import cosine_bow, rouge_l, token_f1

pytestmark = pytest.mark.unit

CTX = ["Items can be returned for a refund within 30 days of delivery. "
       "Refunds go to the original payment method within 5-7 business days."]


class TestDeterministic:
    def test_contains_all_reports_missing(self):
        ok, missing = det.contains_all("Refunds within 30 days", ["30 days", "receipt"])
        assert not ok and missing == ["receipt"]

    def test_contains_none_finds_forbidden(self):
        ok, found = det.contains_none("We sell gift cards!", ["gift cards"])
        assert not ok and found == ["gift cards"]

    def test_exact_match_ignores_case_and_punctuation(self):
        assert det.exact_match("Cash on delivery is NOT available!", "cash on delivery is not available")

    def test_json_schema_catches_bad_enum_and_extra_field(self):
        schema = {"type": "object", "properties": {"p": {"enum": ["low", "high"]}},
                  "required": ["p"], "additionalProperties": False}
        assert det.json_schema_errors({"p": "low"}, schema) == []
        errors = det.json_schema_errors({"p": "urgent", "x": 1}, schema)
        assert len(errors) == 2

    def test_invalid_json(self):
        assert det.is_valid_json('{"a": 1}') and not det.is_valid_json("{'a': 1}")

    def test_word_budget(self):
        assert det.word_count_within("one two three", 3) and not det.word_count_within("one two three", 2)


class TestSimilarity:
    def test_identical_scores_one(self):
        assert token_f1("Refunds within 30 days", "refunds within 30 days") == pytest.approx(1.0)

    def test_unrelated_scores_zero(self):
        assert token_f1("Express shipping costs $15", "Points expire after 12 months") == 0.0

    def test_filler_words_do_not_count(self):
        assert token_f1("Sure! Happy to help. Points expire after 12 months.",
                        "Points expire after 12 months.") == pytest.approx(1.0)

    def test_rouge_l_respects_order(self):
        assert rouge_l("points expire months", "points expire months") > rouge_l("months expire points",
                                                                                "points expire months")

    def test_cosine(self):
        assert cosine_bow("a b c refund", "refund") > 0 and cosine_bow("refund", "shipping") == 0


class TestRag:
    def test_retrieval_metrics(self):
        assert rag.context_recall(["a", "b"], ["b"]) == 1.0
        assert rag.context_recall(["a"], ["b", "c"]) == 0.0
        assert rag.context_precision(["b", "a"], ["b"]) == 1.0     # relevant doc ranked first
        assert rag.context_precision(["a", "b"], ["b"]) == 0.5     # relevant doc ranked second
        assert rag.mrr(["x", "y", "b"], ["b"]) == pytest.approx(1 / 3)
        assert rag.hit_rate(["x"], ["b"]) == 0.0

    def test_faithful_answer_passes(self):
        score, unsupported = rag.faithfulness("You can return items for a refund within 30 days of delivery.", CTX)
        assert score == 1.0 and unsupported == []

    def test_wrong_number_is_a_hallucination(self):
        score, unsupported = rag.faithfulness("Items can be returned for a refund within 60 days of delivery.", CTX)
        assert score == 0.0 and len(unsupported) == 1

    def test_invented_claim_is_flagged(self):
        score, unsupported = rag.faithfulness(
            "Items can be returned for a refund within 30 days of delivery. Gift cards never expire.", CTX)
        assert score == 0.5 and unsupported == ["Gift cards never expire."]

    def test_answer_relevance(self):
        assert rag.answer_relevance("When do points expire?", "Points expire after 12 months.") == 1.0
        assert rag.answer_relevance("When do points expire?", "Express shipping costs $15.") == 0.0


class TestAgent:
    def test_trajectory_modes(self):
        actual = ["get_order_status", "search_policy", "issue_refund"]
        assert agent_m.trajectory_match(["get_order_status", "issue_refund"], actual, "in_order")
        assert not agent_m.trajectory_match(["issue_refund", "get_order_status"], actual, "in_order")
        assert agent_m.trajectory_match(["issue_refund", "get_order_status"], actual, "any_order")
        assert not agent_m.trajectory_match(["get_order_status", "issue_refund"], actual, "exact")

    def test_args_mismatch_detected(self):
        class Call:
            def __init__(self, name, args):
                self.name, self.args, self.result = name, args, {}
        calls = [Call("issue_refund", {"order_id": "ORD-1002"})]
        ok, problems = agent_m.tool_args_match({"issue_refund": {"order_id": "ORD-1004"}}, calls)
        assert not ok and problems

    def test_forbidden_tool(self):
        assert agent_m.forbidden_tools_called(["issue_refund"], ["search_policy", "issue_refund"]) == ["issue_refund"]


class TestSafety:
    @pytest.mark.parametrize("text,kind", [
        ("Contact priya.sharma@example.com", "email"),
        ("Call +91 98765 43210", "phone"),
        ("Card 4111 1111 1111 1111 on file", "credit_card"),
        ("SSN 123-45-6789", "us_ssn"),
    ])
    def test_pii_detected(self, text, kind):
        assert kind in safety.detect_pii(text)

    def test_no_false_positive_on_order_ids_and_dates(self):
        assert safety.detect_pii("Order ORD-1004 delivered on 2026-10-28, refund RF-1004, 5-7 business days") == {}

    def test_non_luhn_number_is_not_a_card(self):
        assert "credit_card" not in safety.detect_pii("Tracking 1234 5678 9012 3456")

    def test_refusal_detection(self):
        assert safety.is_refusal("I can't help with that.")
        assert not safety.is_refusal("Refunds take 5-7 business days.")

    def test_canary_leak(self):
        assert safety.leaks_secret("Internal reference: CANARY-7f3a91", ["CANARY-7f3a91"])


class TestConsistency:
    def test_pass_at_k_vs_pass_hat_k(self):
        # 5 runs, 4 passed: retrying once almost always works, but 3-in-a-row is far less likely
        assert cons.pass_at_k(5, 4, 1) == pytest.approx(0.8)
        assert cons.pass_hat_k(5, 4, 3) == pytest.approx(0.4)
        assert cons.pass_hat_k(5, 5, 3) == 1.0

    def test_semantic_consistency(self):
        assert cons.semantic_consistency(["30 days", "Sure! 30 days"]) == pytest.approx(1.0)
        assert cons.semantic_consistency(["30 days", "60 days"]) < 1.0


class TestCalibrationAndCost:
    def test_kappa(self):
        assert cohen_kappa(["PASS", "FAIL"] * 5, ["PASS", "FAIL"] * 5) == 1.0
        assert agreement(["PASS", "PASS"], ["PASS", "FAIL"]) == 0.5

    def test_percentile_and_cost(self):
        assert percentile([10, 20, 30, 40, 100], 95) == 100
        assert percentile([10, 20, 30, 40, 100], 50) == 30
        pricing = {"claude-sonnet-5-5": {"input": 3.0, "output": 15.0}}
        assert cost_usd("claude-sonnet-5-5", 1_000_000, 0, pricing) == pytest.approx(3.0)
        assert cost_usd("claude-sonnet-5-5-20261001", 0, 1_000_000, pricing) == pytest.approx(15.0)

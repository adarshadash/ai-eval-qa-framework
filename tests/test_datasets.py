"""Eval data quality checks. Bad test data = silent false passes."""
import pytest

from app.knowledge_base import DOCS
from app.tools import TOOL_SCHEMAS
from evals.config import load_dataset

pytestmark = pytest.mark.unit

DOC_IDS = {d["id"] for d in DOCS}
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}

REQUIRED_FIELDS = {
    "golden_qa": {"id", "question", "reference", "must_include", "must_not_include", "tags"},
    "rag_cases": {"id", "question", "expected_doc_ids", "reference"},
    "agent_tool_cases": {"id", "question", "expected_tools", "expected_args", "forbidden_tools"},
    "multi_turn": {"id", "turns"},
    "safety_redteam": {"id", "category", "prompt", "checks"},
    "structured_output": {"id", "ticket", "expected"},
    "judge_calibration": {"id", "question", "reference", "answer", "human_label"},
    "consistency": {"id", "question", "must_include", "reference"},
}


@pytest.mark.parametrize("name,fields", REQUIRED_FIELDS.items())
def test_required_fields_and_unique_ids(name, fields):
    rows = load_dataset(name)  # also fails on duplicate ids / bad JSON
    assert rows, f"{name} is empty"
    for row in rows:
        missing = fields - row.keys()
        assert not missing, f"{name}/{row.get('id')} missing {missing}"


def test_rag_expected_docs_exist():
    for row in load_dataset("rag_cases"):
        assert set(row["expected_doc_ids"]) <= DOC_IDS, row["id"]


def test_agent_cases_reference_real_tools():
    for row in load_dataset("agent_tool_cases"):
        used = set(row["expected_tools"]) | set(row["forbidden_tools"]) | set(row["expected_args"])
        assert used <= TOOL_NAMES, f"{row['id']}: unknown tools {used - TOOL_NAMES}"
        assert not set(row["expected_tools"]) & set(row["forbidden_tools"]), f"{row['id']}: contradictory"


def test_calibration_set_is_balanced():
    labels = [r["human_label"] for r in load_dataset("judge_calibration")]
    assert set(labels) == {"PASS", "FAIL"}
    assert 0.3 <= labels.count("PASS") / len(labels) <= 0.7, "unbalanced labels make agreement meaningless"


def test_golden_set_covers_abstention():
    tags = {t for r in load_dataset("golden_qa") for t in r["tags"]}
    assert "abstention" in tags, "golden set must include questions the bot should NOT answer"

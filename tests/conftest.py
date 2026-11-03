import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.adapter import make_agent, make_classifier  # noqa: E402
from evals.config import load_config  # noqa: E402


def pytest_report_header(config):
    cfg = load_config()
    return f"AI evals mode: {cfg['mode'].upper()} | SUT: {cfg['models']['sut']} | judge: {cfg['models']['judge']}"


@pytest.fixture
def agent():
    """Fresh agent per test: tool side effects (refunds) never leak between cases."""
    return make_agent()


@pytest.fixture
def classifier():
    return make_classifier()


def ids(cases, key="id"):
    return [c[key] for c in cases]

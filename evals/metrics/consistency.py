"""Non-determinism metrics. LLMs are stochastic, so measure rates, not single runs.

pass@k : chance at least ONE of k attempts passes   (fine for "try again" UX)
pass^k : chance ALL k attempts pass                  (what production reliability needs)
"""
from __future__ import annotations

from itertools import combinations
from math import comb

from .similarity import token_f1


def pass_rate(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al., 2021). n runs, c passed."""
    if n - c < k:
        return 1.0
    return 1 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability that k independent runs ALL pass (tau-bench style)."""
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def semantic_consistency(outputs: list[str]) -> float:
    """Mean pairwise similarity between outputs of the same input (1.0 = identical meaning)."""
    if len(outputs) < 2:
        return 1.0
    pairs = list(combinations(outputs, 2))
    return sum(token_f1(a, b) for a, b in pairs) / len(pairs)

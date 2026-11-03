"""Judge calibration: before trusting an LLM judge, measure it against human labels."""
from __future__ import annotations


def agreement(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 0.0


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement corrected for chance. >0.6 substantial, >0.8 almost perfect."""
    n = len(a)
    if n == 0:
        return 0.0
    labels = set(a) | set(b)
    observed = agreement(a, b)
    expected = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)

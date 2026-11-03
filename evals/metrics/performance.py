"""Latency, token and cost metrics."""
from __future__ import annotations

import math


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, pct in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def cost_usd(model: str, input_tokens: int, output_tokens: int, pricing: dict) -> float:
    price = pricing.get(model)
    if price is None:  # live model ids can carry date suffixes; match by prefix
        price = next((p for m, p in pricing.items() if model.startswith(m)), {"input": 0, "output": 0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

"""Reference-based similarity metrics (no model needed).

For semantic similarity in production you'd add embedding cosine similarity
(e.g. sentence-transformers or an embeddings API); the interface is the same.
"""
from __future__ import annotations

import math
from collections import Counter

from .text import tokens


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 (stopwords removed)."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure based on the longest common subsequence."""
    a, b = tokens(prediction), tokens(reference)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def cosine_bow(a: str, b: str) -> float:
    """Cosine similarity of bag-of-words vectors."""
    va, vb = Counter(tokens(a)), Counter(tokens(b))
    dot = sum(va[t] * vb[t] for t in va)
    norm = math.sqrt(sum(v * v for v in va.values())) * math.sqrt(sum(v * v for v in vb.values()))
    return dot / norm if norm else 0.0

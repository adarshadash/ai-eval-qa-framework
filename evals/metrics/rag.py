"""RAG metrics: judge retrieval and generation separately.

Retrieval : did we fetch the right documents?   (recall, precision, MRR, hit@k)
Generation: is the answer grounded in them?     (faithfulness, answer relevance)

faithfulness() here is a fast lexical heuristic. In live mode the suites also run
an LLM-as-judge groundedness check (evals/judge.py) - use both: the heuristic is
free and deterministic, the judge catches paraphrased hallucinations.
"""
from __future__ import annotations

from .text import numbers, sentences, tokens


def context_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not expected_ids:
        return 1.0
    return len(set(retrieved_ids) & set(expected_ids)) / len(set(expected_ids))


def context_precision(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Rank-aware precision (average precision): relevant docs should be ranked first."""
    if not retrieved_ids:
        return 0.0 if expected_ids else 1.0
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in expected_ids:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(d in expected_ids for d in retrieved_ids) / len(retrieved_ids)


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in expected_ids:
            return 1 / rank
    return 0.0


def hit_rate(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    return float(any(d in expected_ids for d in retrieved_ids))


def faithfulness(answer: str, contexts: list[str], min_overlap: float = 0.6) -> tuple[float, list[str]]:
    """Share of answer claims supported by the context.

    A claim (sentence) is supported when >= min_overlap of its content words appear
    in one context AND every number in it appears in that context. Number checking
    catches the most common hallucination: right sentence, wrong figure.
    Returns (score, unsupported_claims).
    """
    claims = [s for s in sentences(answer) if len(tokens(s)) >= 2]
    if not claims:
        return 1.0, []
    unsupported = []
    for claim in claims:
        claim_tokens = set(tokens(claim))
        claim_numbers = numbers(claim)
        supported = False
        for ctx in contexts:
            ctx_tokens = set(tokens(ctx))
            coverage = len(claim_tokens & ctx_tokens) / len(claim_tokens)
            if coverage >= min_overlap and claim_numbers <= numbers(ctx):
                supported = True
                break
        if not supported:
            unsupported.append(claim)
    return 1 - len(unsupported) / len(claims), unsupported


def answer_relevance(question: str, answer: str) -> float:
    """Share of the question's content words addressed in the answer."""
    q = set(tokens(question))
    if not q:
        return 1.0
    return len(q & set(tokens(answer))) / len(q)

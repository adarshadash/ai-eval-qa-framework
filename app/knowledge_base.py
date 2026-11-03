"""Policy knowledge base + a lightweight keyword retriever (IDF-weighted).

In production this would be a vector store (pgvector, Qdrant, Chroma...).
The eval framework only cares about the *contract*: given a query,
return ranked documents with stable IDs, so retrieval can be scored.
"""
from .text_utils import idf_table, split_sentences, tokenize

DOCS = [
    {
        "id": "refund-policy",
        "title": "Refund policy",
        "text": "Items can be returned for a refund within 30 days of delivery. "
        "Damaged items receive a full refund including original shipping. "
        "Refunds go to the original payment method within 5-7 business days.",
    },
    {
        "id": "return-shipping",
        "title": "Return shipping",
        "text": "Return shipping is free for damaged or incorrect items. "
        "For change-of-mind returns, a $5 fee is deducted from the refund.",
    },
    {
        "id": "shipping-times",
        "title": "Shipping times",
        "text": "Standard shipping takes 3-5 business days. "
        "Express shipping takes 1-2 business days and costs $15.",
    },
    {
        "id": "international-shipping",
        "title": "International shipping",
        "text": "We ship internationally to 40 countries. "
        "International orders take 7-14 business days. "
        "Customs duties are paid by the customer.",
    },
    {
        "id": "warranty",
        "title": "Warranty",
        "text": "Electronics carry a 1-year limited warranty covering manufacturing defects. "
        "Accidental damage is not covered by the warranty.",
    },
    {
        "id": "order-cancellation",
        "title": "Order cancellation",
        "text": "Orders can be cancelled free of charge within 2 hours of placing them. "
        "After an order has shipped it cannot be cancelled and must be returned instead.",
    },
    {
        "id": "loyalty-program",
        "title": "Loyalty program",
        "text": "Loyalty members earn 1 point per dollar spent. "
        "100 points can be redeemed for a $5 discount. "
        "Loyalty points expire after 12 months.",
    },
    {
        "id": "payment-methods",
        "title": "Payment methods",
        "text": "We accept Visa, Mastercard, UPI and PayPal. "
        "Cash on delivery is not available.",
    },
]

_DOC_TOKENS = [set(tokenize(d["title"] + " " + d["text"])) for d in DOCS]
_IDF = idf_table(_DOC_TOKENS)


def _score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    return sum(_IDF.get(t, 0.0) for t in query_tokens & doc_tokens)


def search(query: str, top_k: int = 2, min_score: float = 1.0) -> list[dict]:
    """Return up to top_k docs scoring >= min_score, best first."""
    q = set(tokenize(query))
    scored = [(_score(q, toks), doc) for toks, doc in zip(_DOC_TOKENS, DOCS)]
    scored = [(s, d) for s, d in scored if s >= min_score]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**d, "score": round(s, 3)} for s, d in scored[:top_k]]


def best_sentence(query: str, doc: dict) -> str:
    """Extractive answer: the sentence of `doc` that best matches the query."""
    q = set(tokenize(query))
    sentences = split_sentences(doc["text"])
    return max(sentences, key=lambda s: _score(q, set(tokenize(s))))

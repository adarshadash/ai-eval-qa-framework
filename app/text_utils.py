"""Tiny text helpers used by the sample app's retriever (no external deps)."""
import math
import re

STOPWORDS = set(
    """a an the is are was were be been do does did i you your yours my me we our us to for of in on at by
    with and or can could would should what when where which who whom how long many much it its this that
    these those have has had will if any please about from as so there their they them get got want need
    tell know""".split()
)


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def idf_table(docs_tokens: list[set[str]]) -> dict[str, float]:
    n = len(docs_tokens)
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for t in toks:
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1 + n / c) for t, c in df.items()}

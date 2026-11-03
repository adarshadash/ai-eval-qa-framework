"""Text normalisation shared by all metrics."""
import re
import string

# Stopwords + conversational filler ("Sure!", "Happy to help") that carry no facts.
# Negations are deliberately NOT stopwords: "covered" vs "not covered" must differ.
STOPWORDS = set(
    """a an the is are was were be been being do does did i you your yours my me we our us to for of in on
    at by with and or can could would should what when where which who whom how long many much it its this
    that these those have has had will if any please about from as so there their they them get got want
    need tell know sure happy help glad hi hello thanks thank great just let""".split()
)

_NUMBER_RE = re.compile(r"\$?\d+(?:[.,:-]\d+)*%?")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "").replace("$", "")))
    return re.sub(r"\s+", " ", text)


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    words = re.findall(r"[a-z0-9$]+(?:-[a-z0-9]+)*", text.lower())
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [stem(w) for w in words]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def numbers(text: str) -> set[str]:
    """Numbers/amounts/ranges in text: '30', '$15', '1-2', '5-7'."""
    return {n.lstrip("$").rstrip(".,") for n in _NUMBER_RE.findall(text)}


NEGATIONS = {"not", "no", "never", "cannot", "can't", "isn't", "aren't", "won't", "don't", "doesn't"}


def has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & NEGATIONS)

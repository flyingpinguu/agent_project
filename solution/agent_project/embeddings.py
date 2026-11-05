"""Small deterministic text vectors for no-key local evaluation."""

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text for the deterministic retrieval baseline."""

    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def vectorize(text: str) -> Counter[str]:
    """Represent text as a sparse term-frequency vector."""

    return Counter(tokenize(text))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    """Compute cosine similarity between sparse vectors."""

    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(token, 0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    return dot / (left_norm * right_norm)

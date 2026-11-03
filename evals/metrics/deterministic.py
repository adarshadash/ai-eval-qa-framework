"""Deterministic assertions - cheap, fast, never flaky. Always run these FIRST."""
from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator

from .text import normalize


def exact_match(output: str, expected: str) -> bool:
    return normalize(output) == normalize(expected)


def contains_all(output: str, required: list[str]) -> tuple[bool, list[str]]:
    """Case-insensitive. Returns (ok, missing)."""
    low = output.lower()
    missing = [r for r in required if r.lower() not in low]
    return not missing, missing


def contains_none(output: str, forbidden: list[str]) -> tuple[bool, list[str]]:
    low = output.lower()
    found = [f for f in forbidden if f.lower() in low]
    return not found, found


def regex_match(output: str, pattern: str) -> bool:
    return re.search(pattern, output, flags=re.IGNORECASE) is not None


def word_count_within(output: str, max_words: int) -> bool:
    return len(output.split()) <= max_words


def is_valid_json(output: str) -> bool:
    try:
        json.loads(output)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def json_schema_errors(data, schema: dict) -> list[str]:
    """Empty list = valid. Each error names the failing field path."""
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in validator.iter_errors(data)]

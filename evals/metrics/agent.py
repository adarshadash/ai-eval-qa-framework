"""Agent metrics: test the PATH, not just the final answer."""
from __future__ import annotations


def trajectory_match(expected: list[str], actual: list[str], mode: str = "in_order") -> bool:
    """
    exact     : same tools, same order, nothing extra
    in_order  : expected tools appear in this order (extra calls allowed in between)
    any_order : all expected tools were called, in any order
    """
    if mode == "exact":
        return actual == expected
    if mode == "any_order":
        return set(expected) <= set(actual)
    if mode == "in_order":
        it = iter(actual)
        return all(tool in it for tool in expected)
    raise ValueError(f"unknown trajectory mode {mode!r}")


def tool_args_match(expected_args: dict[str, dict], tool_calls) -> tuple[bool, list[str]]:
    """For each tool, at least one call must contain the expected key/values (case-insensitive)."""
    problems = []
    for tool, want in expected_args.items():
        calls = [c for c in tool_calls if c.name == tool]
        if not calls:
            problems.append(f"{tool} never called")
            continue
        if not any(all(str(c.args.get(k, "")).lower() == str(v).lower() for k, v in want.items())
                   for c in calls):
            problems.append(f"{tool} args {[c.args for c in calls]} != expected {want}")
    return not problems, problems


def forbidden_tools_called(forbidden: list[str], actual: list[str]) -> list[str]:
    return [t for t in forbidden if t in actual]


def tool_error_count(tool_calls) -> int:
    """Tool calls that returned an error (bad args, invalid order...)."""
    return sum(1 for c in tool_calls if isinstance(c.result, dict) and "error" in c.result)

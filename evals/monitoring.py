"""Online evals: score sampled PRODUCTION traces (no reference answers available).

Offline evals gate releases; online evals catch what the golden set never imagined.
Feed this from your tracing tool export (Langfuse, LangSmith, OpenTelemetry, logs).
Flagged traces should be triaged and turned into new golden/regression cases.
"""
from __future__ import annotations

from .config import load_config
from .metrics.rag import faithfulness
from .metrics.safety import detect_pii, toxicity_hits


def score_trace(trace: dict) -> dict:
    cfg = load_config()["monitoring"]
    flags = []
    faith, unsupported = faithfulness(trace["answer"], trace.get("contexts", []))
    if trace.get("contexts") and faith < cfg["min_faithfulness"]:
        flags.append("low_faithfulness")
    pii = detect_pii(trace["answer"])
    if pii:
        flags.append("pii_leak")
    if toxicity_hits(trace["answer"]):
        flags.append("toxicity")
    if trace.get("latency_ms", 0) > cfg["max_latency_ms"]:
        flags.append("slow_response")
    if trace.get("input_tokens", 0) + trace.get("output_tokens", 0) > cfg["max_tokens"]:
        flags.append("token_budget")
    return {"id": trace["id"], "flags": flags, "faithfulness": round(faith, 3),
            "unsupported_claims": unsupported, "pii": pii}


def score_traces(traces: list[dict]) -> dict:
    scored = [score_trace(t) for t in traces]
    flagged = [s for s in scored if s["flags"]]
    by_flag: dict[str, int] = {}
    for s in flagged:
        for f in s["flags"]:
            by_flag[f] = by_flag.get(f, 0) + 1
    return {"total": len(scored), "flagged": len(flagged), "flag_rate": len(flagged) / max(1, len(scored)),
            "by_flag": by_flag, "traces": scored}

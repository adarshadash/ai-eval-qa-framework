"""The ONE place that connects the eval framework to the system under test.

To evaluate your own app, change these two factories so they return objects with:
  agent.run(question, history=None) -> AgentResult-like object
      (.answer, .tool_calls[.name/.args], .contexts[{"id","text"}], .steps,
       .input_tokens, .output_tokens, .latency_ms, .model)
  classifier.classify(text) -> raw JSON string
"""
from __future__ import annotations

import os

from app.support_agent import SupportAgent
from app.ticket_classifier import TicketClassifier

from .config import load_config
from .llm_client import build_client


def make_agent() -> SupportAgent:
    cfg = load_config()
    live = cfg["mode"] == "live"
    return SupportAgent(
        mode=cfg["mode"],
        llm=build_client("sut", cfg) if live else None,
        model=cfg["models"]["sut"] if live else "mock-shopbot-v1",
        max_steps=cfg["agent"]["max_steps"],
        top_k=cfg["retrieval"]["top_k"],
        min_score=cfg["retrieval"]["min_score"],
        seed=cfg["seed"],
        inject_bugs=os.getenv("EVAL_INJECT_BUGS", "0") == "1",  # demo: watch the evals catch failures
    )


def make_classifier() -> TicketClassifier:
    cfg = load_config()
    live = cfg["mode"] == "live"
    return TicketClassifier(mode=cfg["mode"], llm=build_client("sut", cfg) if live else None)

"""Support-ticket classifier: the 'structured output' part of the SUT.

Returns JSON that must match TICKET_SCHEMA. The evals check schema validity
and per-field accuracy against labelled tickets.
"""
from __future__ import annotations

import json
import re

TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["refund", "shipping", "account", "technical", "other"]},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "summary": {"type": "string", "maxLength": 120},
    },
    "required": ["category", "priority", "sentiment", "summary"],
    "additionalProperties": False,
}

PROMPT = """Classify this customer support ticket.
Return ONLY a JSON object matching this JSON Schema, with no markdown:
{schema}

Priority rules: "high" for damaged items, fraud, safety issues or very angry customers;
"low" for general questions and compliments; otherwise "medium".

<ticket>{ticket}</ticket>"""


class TicketClassifier:
    def __init__(self, mode: str = "mock", llm=None):
        self.mode = mode
        self.llm = llm

    def classify(self, ticket: str) -> str:
        """Returns the raw model output (a JSON string) - parsing is part of what we evaluate."""
        if self.mode == "live":
            resp = self.llm.complete(messages=[{"role": "user", "content": PROMPT.format(
                schema=json.dumps(TICKET_SCHEMA), ticket=ticket)}])
            return resp.text
        return json.dumps(self._mock(ticket))

    @staticmethod
    def _mock(ticket: str) -> dict:
        t = ticket.lower()
        has = lambda *words: any(w in t for w in words)  # noqa: E731
        if has("refund", "money back", "return"):
            category = "refund"
        elif has("deliver", "shipping", "package", "tracking", "arrive"):
            category = "shipping"
        elif has("password", "login", "account", "sign in"):
            category = "account"
        elif has("app", "crash", "error", "bug", "website"):
            category = "technical"
        else:
            category = "other"

        if has("damaged", "broken", "fraud", "unauthori", "fire", "urgent", "furious", "worst"):
            priority = "high"
        elif has("thank", "love", "great", "question", "wondering"):
            priority = "low"
        else:
            priority = "medium"

        if has("terrible", "furious", "angry", "worst", "broken", "damaged", "unacceptable", "unauthori"):
            sentiment = "negative"
        elif has("thank", "love", "great", "awesome"):
            sentiment = "positive"
        else:
            sentiment = "neutral"

        summary = re.sub(r"\s+", " ", ticket).strip()[:100]
        return {"category": category, "priority": priority, "sentiment": sentiment, "summary": summary}


def parse_json_output(raw: str) -> dict | None:
    """Tolerate ```json fences; return None if the output isn't valid JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None

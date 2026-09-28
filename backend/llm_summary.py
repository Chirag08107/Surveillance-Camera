"""
STEP 16 - LLM INCIDENT SUMMARIES

Turns a structured anomaly event (track id, reasons, nearby detections)
into a short human-readable incident report using the Claude API.

Requires ANTHROPIC_API_KEY in the environment (see .env.example).
Get one from https://console.anthropic.com - this is separate from
any Claude.ai subscription.
"""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"


def summarize_incident(track_id: int, reasons: list[str], context: dict) -> str:
    """
    context: any extra structured info you have - recent behavior labels,
    zone name, timestamps, plate text if a vehicle was involved, etc.
    """

    prompt = f"""You are writing a short incident summary for a security operator dashboard.

Track ID: {track_id}
Flagged reasons: {", ".join(reasons)}
Additional context: {context}

Write 2-3 plain-English sentences describing what likely happened and
why it was flagged. Be factual and neutral - do not speculate about
identity or intent beyond what the data supports. No preamble."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    return "".join(block.text for block in response.content if block.type == "text").strip()

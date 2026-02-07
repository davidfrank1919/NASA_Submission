# llm_client.py
#found reference for this import via google search
#CoPilot repaired Lines 86-88 Changed my temperature settings to be more conservative (0.2) to reduce hallucinations, and set max output tokens to 700 to allow for detailed responses with citations. These can be adjusted via environment variables or function arguments as needed.
from __future__ import annotations

from typing import Dict, List, Optional
import os
from openai import OpenAI

# Vocareum OpenAI-compatible endpoint
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
# Default model for LLM calls (can be overridden via env or function arg)
DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")


def _build_system_prompt() -> str:
    """
    Defines the assistant persona and strict grounding rules for RAG.
    """
    return (
        "You are NASA Mission Archive Assistant.\n\n"
        "RULES (IMPORTANT):\n"
        "1) Use ONLY the provided CONTEXT from NASA documents to answer.\n"
        "2) If the CONTEXT is insufficient, say so clearly and ask a follow-up question.\n"
        "3) Do NOT invent details, dates, names, or events.\n"
        "4) When making factual claims, cite evidence using bracketed source tags present in the context.\n"
        "5) Keep answers clear and technical; use bullet points when helpful.\n"
    )


def _truncate_history(conversation_history: List[Dict], max_turns: int = 8) -> List[Dict]:
    """
    Keep only the last max_turns user/assistant pairs to avoid context overflow.
    """
    if not conversation_history:
        return []
    keep = max(0, 2 * max_turns)  # user+assistant pairs
    return conversation_history[-keep:]


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: Optional[str] = None,
) -> str:
    """Generate response using OpenAI with context"""

    system_prompt = _build_system_prompt()

    # Build messages: system -> history -> user(with context)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    history = _truncate_history(
        conversation_history,
        max_turns=int(os.getenv("MAX_HISTORY_TURNS", "8"))
    )

    for m in history:
        if isinstance(m, dict) and "role" in m and "content" in m:
            if m["role"] in ("user", "assistant", "system"):
                messages.append({"role": m["role"], "content": str(m["content"])})

    # Add current user message with context
    user_payload = (
        "CONTEXT (NASA ARCHIVES):\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{user_message}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer using only the CONTEXT.\n"
        "- If you cannot answer from the context, say so and ask a clarifying question.\n"
    )
    messages.append({"role": "user", "content": user_payload})

    # Create OpenAI client for Vocareum endpoint
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=openai_key)

    # Decide model (function arg takes precedence, then env DEFAULT_MODEL)
    model_to_use = model or DEFAULT_MODEL

    # Send request
    resp = client.chat.completions.create(
        model=model_to_use,
        messages=messages,
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "700")),
    )

    # Safely extract content from various response shapes returned by SDKs
    content = None
    try:
        if not resp or not getattr(resp, "choices", None):
            raise ValueError("empty response or no choices")

        choice0 = resp.choices[0]

        # Try attribute-style access first (SDK objects)
        msg = getattr(choice0, "message", None)

        # If the choice is a plain dict, handle that too
        if msg is None and isinstance(choice0, dict):
            msg = choice0.get("message")

        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)

        # Fallbacks: older SDKs or API shapes sometimes put text on the choice
        if not content:
            content = getattr(choice0, "text", None)
            if not content and isinstance(choice0, dict):
                content = choice0.get("text")
    except Exception:
        content = None

    if not content:
        raise RuntimeError(f"LLM returned no content. Raw response: {repr(resp)}")

    return content.strip()

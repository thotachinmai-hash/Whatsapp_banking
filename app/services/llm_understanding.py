"""LLM-assisted understanding for the literal-option confirmation/menu
resolution gates — a thin fallback layer consulted only when
app/workflows/nlu.py's own rule-based matching finds no signal, never a
replacement for it.

Mirrors the fail-safe pattern already used by app/services/language.py
and app/conversation/intent/llm_routing.py: the shared Sarvam client, a
strict prompt asking for structured output only, temperature=0, and a
try/except that returns None on any failure so a flaky/slow LLM call
degrades to the caller's existing rule-based behavior rather than ever
surfacing an error to the customer.

Every function here is read-only / advisory — none of them execute a
banking action or mutate workflow state themselves; callers decide what to
do with the result."""

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv

from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

def _get_client():
    from app.services.sarvam_client import get_sarvam_client

    return get_sarvam_client()


def _model() -> str:
    from app.services.sarvam_client import get_fast_model

    return get_fast_model()


# These are short, structured-output prompts (parse a choice, answer a side
# question, detect a jump) — well within get_fast_model()'s strengths, and
# it doesn't have sarvam-105b's hidden-reasoning token burn (see
# get_fast_model()'s docstring), so this headroom is generous rather than
# load-bearing the way it was for the old reasoning model.
_MAX_TOKENS = 800
_REASONING_EFFORT = "low"


def is_llm_fallback_enabled() -> bool:
    """Single combined flag gating every LLM-fallback behavior added on top
    of this app's rule-based matching (flexible yes/no & menu matching,
    mid-workflow side-question answering, step/workflow-jump detection,
    and the free-intent LLM classifier fallback). Default off so tests and
    any environment without a SARVAM_API_KEY stay on the original,
    deterministic behavior unless explicitly opted in."""
    return os.getenv("LLM_FALLBACK_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.lower().startswith("json"):
                text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def interpret_choice_llm(
    text: str,
    options: list[str],
    prompt_context: str = "",
    trace_id: str = "",
) -> Optional[str]:
    """Resolve a free-form reply against the literal list of valid choices
    for the CURRENT prompt (e.g. ["yes", "no"], or the actual beneficiary
    names/numbers on screen). Only ever called after the caller's own
    cheap rule-based match already returned no result. Returns exactly one
    value from `options`, or None if the model is unsure, fails, or
    returns something not literally in `options` — never invents a
    choice the customer wasn't actually offered."""
    if not text or not text.strip() or not options:
        return None

    start = time.time()
    try:
        response = _get_client().chat.completions(
            model=_model(),
            temperature=0,
            max_tokens=_MAX_TOKENS,
            reasoning_effort=_REASONING_EFFORT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "A WhatsApp banking assistant asked the user to pick one of a "
                        "fixed set of options. Decide which option the user's reply "
                        "means, if any. Reply with ONLY a JSON object of the exact "
                        'shape {"choice": "<one of the given options, or null>"}. '
                        "The value must be copied EXACTLY as given in the options "
                        "list, or null if the reply doesn't clearly match any of "
                        "them. Do not invent an option that wasn't given."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context: {prompt_context}\n"
                        f"Options: {json.dumps(options)}\n"
                        f"User reply: {text[:300]!r}"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[{trace_id}] LLM choice interpretation call failed | error={e}")
        return None

    parsed = _parse_json_object(raw)
    if not parsed:
        return None
    choice = parsed.get("choice")
    duration = (time.time() - start) * 1000
    if choice in options:
        logger.info(f"[{trace_id}] LLM choice resolved | choice={choice!r} | duration={duration:.2f}ms")
        return choice
    return None


# answer_side_question(), detect_step_or_workflow_jump(), and
# detect_soft_decline() were removed as part of the LLM-first routing
# migration (docs/current_architecture.md, "Phase 13") — each was its own
# separate Sarvam call duplicating a judgment the single LLM routing
# decision (app/conversation/intent/llm_routing.py::classify_and_route_llm)
# now makes once per turn: a mid-workflow side question is TOOL/RAG, a
# workflow pivot is SWITCH, and a natural-language decline is CANCEL. See
# app/workflows/manager.py::WorkflowManager.handle() for how those three
# outcomes are handled from that single decision.

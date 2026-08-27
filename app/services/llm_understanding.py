"""LLM-assisted understanding — a thin fallback layer on top of this app's
existing rule-based matching, not a replacement for it.

Mirrors the fail-safe pattern already used by
app/conversation/intent/classifier.py::default_llm_classify and
app/services/language.py: the shared Sarvam client, a strict prompt
asking for structured output only, temperature=0, and a try/except that
returns None on any failure so a flaky/slow LLM call degrades to the
caller's existing rule-based behavior rather than ever surfacing an error
to the customer.

Every function here is read-only / advisory — none of them execute a
banking action or mutate workflow state themselves; callers decide what to
do with the result, exactly like classify_intent()'s llm_classify hook.
"""

import json
import os
import time
from dataclasses import dataclass, field
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


def answer_side_question(
    text: str,
    workflow_type: str,
    step: Optional[str],
    trace_id: str = "",
) -> Optional[str]:
    """Short, read-only banking Q&A answer for an off-topic-but-in-scope
    question asked mid-workflow (e.g. a balance question while mid cheque
    deposit). No tools are bound, so this can only answer from general
    banking knowledge/policy, not look up real account data — callers must
    only use this for questions that don't need real customer data (the
    workflow-specific "what's missing" explainers already live in each
    processor and take priority over this). Returns None on failure/if the
    question is out of scope, so the caller can fall back to its existing
    behavior."""
    if not text or not text.strip():
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
                        "You are a WhatsApp banking assistant. The user is in the "
                        f"middle of a {workflow_type} workflow (step: {step or 'unknown'}) "
                        "and just asked a short side question instead of answering the "
                        "current step. Answer ONLY general banking questions briefly (2-3 "
                        "sentences max, no markdown headers). Do not invent account "
                        "numbers, balances, or transaction data you don't have. If the "
                        "question is out of scope for a bank assistant, or asks you to "
                        "do/approve/execute a banking action, reply with exactly: NONE"
                    ),
                },
                {"role": "user", "content": text[:500]},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"[{trace_id}] LLM side-question answer failed | error={e}")
        return None

    duration = (time.time() - start) * 1000
    if not answer or answer.upper() == "NONE":
        return None
    logger.info(f"[{trace_id}] LLM side-question answered | workflow={workflow_type} | duration={duration:.2f}ms")
    return answer


@dataclass
class JumpTarget:
    target_workflow: Optional[str] = None
    target_step: Optional[str] = None
    confidence: float = 0.0
    extracted: dict = field(default_factory=dict)


_JUMP_WORKFLOWS = ("cheque", "loan", "kyc", "transfer", "onboarding")


def detect_step_or_workflow_jump(
    text: str,
    current_workflow: str,
    current_step: Optional[str],
    trace_id: str = "",
) -> Optional[JumpTarget]:
    """Classify whether a mid-workflow message clearly signals wanting a
    DIFFERENT workflow than the one currently active (e.g. mid-cheque
    deposit, "actually let me transfer money instead"). Only ever called
    after a cheap keyword gate already flagged the message as a possible
    pivot (see app/workflows/manager.py::_looks_like_new_service_request) —
    this call decides whether that's a real, confident jump or a false
    positive. Returns None on failure, low confidence, or "no jump"."""
    if not text or not text.strip():
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
                        "A WhatsApp banking user is mid-way through a "
                        f"'{current_workflow}' request (step: {current_step or 'unknown'}). "
                        "Decide if their message CLEARLY signals they want to abandon "
                        "this and start a DIFFERENT one of these workflows instead: "
                        + ", ".join(_JUMP_WORKFLOWS) + ". "
                        "Reply with ONLY a JSON object of the exact shape "
                        '{"target_workflow": "<one of the workflow names, or null>", '
                        '"confidence": 0.0-1.0}. '
                        "Use null if the message is actually answering/continuing the "
                        "current step, is ambiguous, or doesn't clearly ask for a "
                        "different workflow. Never pick the same workflow as the "
                        f"current one ('{current_workflow}')."
                    ),
                },
                {"role": "user", "content": text[:300]},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[{trace_id}] LLM jump detection call failed | error={e}")
        return None

    parsed = _parse_json_object(raw)
    if not parsed:
        return None
    target = parsed.get("target_workflow")
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    duration = (time.time() - start) * 1000
    if target not in _JUMP_WORKFLOWS or target == current_workflow or confidence < 0.6:
        return None
    logger.info(
        f"[{trace_id}] LLM workflow jump detected | from={current_workflow} | to={target} | "
        f"confidence={confidence:.2f} | duration={duration:.2f}ms"
    )
    return JumpTarget(target_workflow=target, confidence=confidence)


def detect_soft_decline(text: str, workflow_type: str, trace_id: str = "") -> bool:
    """Catch a natural-language "I don't want to do this right now" that
    app.workflows.manager._is_cancel_command's regex misses because it
    doesn't contain a literal cancel/stop/end word ("okay I don't want to
    go with the transfer right now", "let's not do this", "actually never
    mind", "not this right now"). Only ever called as a fallback AFTER
    the cheap regex already found nothing — this exists so a customer's
    clearly-stated wish to pause isn't silently ignored (the workflow
    just sits there active with no acknowledgment) just because they
    didn't happen to say the exact word "cancel". Deliberately narrow:
    only a message that unambiguously signals wanting to stop/pause
    counts — an ordinary answer to the current step, or a side question,
    must never be misread as a decline. Returns False on any failure,
    low confidence, or ambiguity — matching this app's fail-safe pattern
    (see interpret_choice_llm/detect_step_or_workflow_jump above)."""
    if not text or not text.strip():
        return False

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
                        "A WhatsApp banking user is mid-way through a "
                        f"'{workflow_type}' request. Decide if their message "
                        "CLEARLY signals they want to stop or pause it right "
                        "now — even phrased casually, e.g. \"I don't want to "
                        "do this right now\", \"let's not do this\", \"actually "
                        "never mind\", \"not now\", \"hold off on that\". "
                        'Reply with ONLY a JSON object of the exact shape '
                        '{"wants_to_pause": true or false, "confidence": 0.0-1.0}. '
                        "Use false if the message is actually answering/"
                        "continuing the current step, asks an unrelated "
                        "question, or is at all ambiguous — only a genuinely "
                        "clear wish to stop counts as true."
                    ),
                },
                {"role": "user", "content": text[:300]},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[{trace_id}] LLM soft-decline detection call failed | error={e}")
        return False

    parsed = _parse_json_object(raw)
    if not parsed:
        return False
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    wants_to_pause = bool(parsed.get("wants_to_pause")) and confidence >= 0.6

    duration = (time.time() - start) * 1000
    if wants_to_pause:
        logger.info(
            f"[{trace_id}] LLM soft decline detected | workflow={workflow_type} | "
            f"confidence={confidence:.2f} | duration={duration:.2f}ms"
        )
    return wants_to_pause

"""Deterministic rules — the only intent-adjacent logic that runs before
the LLM router (app/conversation/intent/llm_routing.py).

Everything here is intentionally narrow: prompt-injection detection and a
short list of literal, exact-phrase control words (cancel/back/menu/repeat/
restart). Every other decision — including greeting, out-of-scope,
workflow-start/switch/continue/correct/cancel-by-natural-language, banking
questions, and status/lookup requests — is made by the LLM router, once,
per turn. See docs/current_architecture.md, "Phase 13 — LLM-First Routing
Migration" for why the old 9-layer keyword classifier was removed.

None of the functions below execute a banking action, call a tool, or
change workflow state — they only ever return an IntentResult or None.
"""

import re
from typing import Optional

from app.conversation.context import ConversationContext
from app.conversation.intent.models import IntentResult


def _normalize(text: str) -> str:
    """Lowercase and strip everything but letters/digits/spaces — used for
    exact-phrase membership checks (e.g. "cancel" vs "cancel it, please!")."""
    return re.sub(r"[^a-z0-9 ]", "", text.strip().lower()).strip()


# ─── prompt-injection / role-override detection ─────────────────────────

_INJECTION_PATTERN = re.compile(
    r"\bignore\s+(all\s+|your\s+|any\s+|the\s+|previous\s+)*instructions\b"
    r"|\bdisregard\s+(all\s+|your\s+|any\s+|the\s+|previous\s+)*instructions\b"
    r"|\bforget\s+(all\s+|your\s+|any\s+|the\s+|previous\s+)*instructions\b"
    r"|\byou\s+are\s+now\b"
    r"|\bact\s+as\s+(a|an)\b"
    r"|\bsystem\s+prompt\b"
    r"|\bnew\s+instructions\b",
    re.I,
)


def looks_like_injection(text: str) -> bool:
    """User text is untrusted input, never a new instruction to the
    router — checked before anything else, including the LLM call, so a
    role-override attempt never even reaches the model as something to
    reason about."""
    return bool(_INJECTION_PATTERN.search(text))


# ─── hard, literal navigation protocol ──────────────────────────────────
#
# Exact-phrase / prefix matching only — no synonym or semantic guessing.
# Greeting is deliberately NOT here: the LLM router classifies it like
# everything else (GREETING is one of its action values).

_CANCEL_WORDS = {"cancel", "cancel it", "stop", "stop it", "quit", "exit", "end", "never mind", "nevermind", "forget it"}
_BACK_WORDS = {"back", "go back", "previous", "previous step", "b"}
_MENU_WORDS = {
    "menu", "main menu", "show menu", "display menu", "show me the menu",
    "take me to the main menu", "open menu", "home",
}
_REPEAT_WORDS = {"repeat", "repeat that", "say again", "say that again", "what did you say"}
_RESTART_WORDS = {"start over", "start again", "restart", "begin again"}


def classify_hard_navigation(text: str) -> Optional[IntentResult]:
    """Context-independent control commands — these mean the same thing
    no matter what workflow (if any) is active."""
    normalized = _normalize(text)
    if not normalized:
        return None
    if (
        (normalized in _CANCEL_WORDS or normalized.startswith(("cancel ", "stop ")))
        and not any(ord(ch) > 127 for ch in text)
    ):
        # _normalize() strips every non-Latin character, so a compound
        # message like "never mind, मुझे लोन चाहिए" collapses to exactly
        # "never mind" and would match here even though the customer named
        # a real, different request in native script that the stripping
        # silently discarded. Pure-ASCII "never mind" and its siblings
        # still match deterministically and instantly; a message with
        # non-Latin content this rule can't safely interpret defers to the
        # LLM router instead of confidently guessing.
        return IntentResult(intent="cancel", confidence=0.99, method="rule")
    if normalized in _BACK_WORDS:
        return IntentResult(intent="back", confidence=0.98, method="rule")
    if normalized in _RESTART_WORDS:
        return IntentResult(intent="start_over", confidence=0.95, method="rule")
    if normalized in _MENU_WORDS:
        return IntentResult(intent="main_menu", confidence=0.95, method="rule")
    if normalized in _REPEAT_WORDS:
        return IntentResult(intent="repeat", confidence=0.9, method="rule")
    return None


# ─── workflow-context confirmation shorthand (protocol, not semantics) ──

_CONFIRM_ANSWERS = {"yes", "y", "confirm", "no", "n"}


def classify_workflow_conversation(text: str, context: ConversationContext) -> Optional[IntentResult]:
    """The one context-aware case that stays deterministic: a bare "yes"/
    "no"/"confirm" while a workflow's CONFIRM_* step is awaiting exactly
    that answer is a financial-confirmation-gate protocol reply, not a
    natural-language intent to classify — see app/workflows/nlu.py's
    interpret_confirmation, which every workflow processor's own
    STEP_CONFIRM_* step already uses as the real gate. This function only
    lets a bare answer skip the (otherwise unconditional) LLM router call
    for the common, unambiguous case; the processor's own interpretation
    remains the actual authority."""
    if not context or not context.current_workflow:
        return None
    step = (context.current_step or "").upper()
    if step.startswith("CONFIRM") and _normalize(text) in _CONFIRM_ANSWERS:
        return IntentResult(
            intent="workflow_confirmation",
            confidence=0.95,
            entities={"answer": _normalize(text)},
            method="context",
        )
    return None

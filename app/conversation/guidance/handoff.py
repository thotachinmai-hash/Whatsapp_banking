"""Guidance -> workflow handoff resolution — Task 9.2.

Pure text-matching helpers only: given the raw message text and the list
of GuidanceAction identifiers offered on the *previous* turn
(ConversationContext.allowed_actions), decide which one (if any) the user
just selected — by number, by a natural-language phrase, or "cancel"/
"back". Returns a GuidanceAction/None; never starts a workflow, never
touches Redis/Postgres, never calls a banking tool. app/conversation/manager.py
is the only thing that acts on the result, through the existing
WorkflowManager/workflow_adapter mechanisms — exactly like a RoutingDecision
from app/conversation/router.py is only ever acted on by the manager.
"""

import re
from typing import Optional

from app.conversation.guidance.models import GuidanceAction

# Maps a START_* action to the workflow type string used elsewhere in the
# app (app/workflows/constants.py's WORKFLOW_* values / router.py's
# _WORKFLOW_FOR_INTENT) — the only place this mapping needs to exist for
# the handoff path.
WORKFLOW_FOR_ACTION = {
    GuidanceAction.START_TRANSFER: "transfer",
    GuidanceAction.START_CHEQUE_DEPOSIT: "cheque",
    GuidanceAction.START_LOAN_APPLICATION: "loan",
    GuidanceAction.START_KYC_UPDATE: "kyc",
}

# Natural-language phrases a user might use to pick each offered action —
# section 16 explicitly asks for this without a large new classifier
# taxonomy, so this stays a small, fixed, per-action phrase list rather
# than a general-purpose NLU layer.
_ACTION_PHRASES: dict[GuidanceAction, tuple[str, ...]] = {
    GuidanceAction.START_TRANSFER: (
        "start transfer", "let's transfer", "lets transfer", "transfer it",
        "start it", "go ahead", "proceed", "yes", "y",
    ),
    GuidanceAction.START_CHEQUE_DEPOSIT: (
        "deposit it", "deposit the cheque", "deposit cheque", "start deposit",
        "start it", "go ahead", "proceed", "yes", "y",
    ),
    GuidanceAction.START_LOAN_APPLICATION: (
        "start application", "apply", "yes apply", "i want to apply",
        "start it", "go ahead", "proceed", "yes", "y",
    ),
    GuidanceAction.START_KYC_UPDATE: (
        "update kyc", "start kyc", "update my kyc", "start it",
        "go ahead", "proceed", "yes", "y",
    ),
    GuidanceAction.SHOW_LOAN_REQUIREMENTS: (
        "requirement", "requirements", "criteria", "what do i need",
        "tell me the requirements", "see requirements",
    ),
    GuidanceAction.SHOW_LOAN_DOCUMENTS: (
        "document", "documents", "paperwork", "show me the documents",
        "show documents", "see documents",
    ),
    GuidanceAction.SHOW_KYC_INFORMATION: (
        "explain", "tell me more", "more information", "ask another question",
    ),
    GuidanceAction.SHOW_CHEQUE_INFORMATION: (
        "explain", "explain cheque deposit", "tell me more", "how does it work",
        "more information",
    ),
    GuidanceAction.SHOW_TRANSFER_INFORMATION: (
        "explain", "explain the process", "tell me more", "how does it work",
        "learn more",
    ),
}

_CANCEL_WORDS = re.compile(r"^\s*cancel\b", re.I)
_BACK_WORDS = re.compile(r"^\s*back\b", re.I)


def resolve_pending_action(text: str, allowed_actions: list[str]) -> Optional[GuidanceAction]:
    """Return the GuidanceAction the message selects, GuidanceAction.CANCEL /
    GuidanceAction.BACK, or None if the message doesn't match any of the
    currently-offered actions (caller should then clear stale
    pending_action/allowed_actions and continue the normal turn flow)."""
    if not allowed_actions:
        return None

    normalized = (text or "").strip().lower()
    if not normalized:
        return None

    if _CANCEL_WORDS.match(normalized):
        return GuidanceAction.CANCEL
    if _BACK_WORDS.match(normalized):
        return GuidanceAction.BACK

    # Numbered reply — position in the SAME order the guidance renderer
    # offered them (see responses.py's `actions` lists).
    if normalized.isdigit():
        index = int(normalized) - 1
        if 0 <= index < len(allowed_actions):
            try:
                return GuidanceAction(allowed_actions[index])
            except ValueError:
                return None
        return None

    # Natural-language phrase match, restricted to the actions actually on
    # offer this turn — "yes"/"go ahead" only resolves to the single
    # relevant action because only one is ever in `allowed_actions` at once.
    # Word-boundary matching (not a bare substring check) — a short word
    # like "y" or "ok" must not match inside an unrelated word ("today",
    # "book").
    for action_value in allowed_actions:
        try:
            action = GuidanceAction(action_value)
        except ValueError:
            continue
        phrases = _ACTION_PHRASES.get(action, ())
        if any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in phrases):
            return action

    return None

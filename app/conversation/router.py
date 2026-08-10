"""Conversation router — Phase 3.

Translates an IntentResult (+ ConversationContext) into a RoutingDecision.
See docs/current_architecture.md, "Intent-Based Routing — Phase 3".

ARCHITECTURAL RULE: this module only decides *where a turn should go*. It
never calls a banking tool, never starts/advances/completes a workflow
itself, and never touches the database. app/agent/agent.py::run_agent()
is the only thing that acts on a RoutingDecision, and it does so entirely
through the existing WorkflowManager / LLM+tools mechanisms — see the
"execution" side of each action documented below.
"""

from typing import Optional

from pydantic import BaseModel

from app.conversation.context import ConversationContext
from app.conversation.intent.models import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    LLM_ELIGIBLE_INTENTS,
    NAVIGATION_INTENTS,
    WORKFLOW_EXECUTING_INTENTS,
)
from app.conversation.intent.models import IntentResult
from app.logger import get_logger

logger = get_logger(__name__)

# Every action a RoutingDecision can carry. run_agent() is expected to
# handle exactly these — see the module docstring in app/agent/agent.py's
# routing block for what each one actually executes.
ROUTING_ACTIONS = {
    "START_WORKFLOW",       # begin a new deterministic workflow (registration/transfer/loan/cheque/kyc)
    "WORKFLOW",             # an active workflow already owns this turn — let it (or its LLM-assisted explanation) continue
    "BANKING_LLM",          # answer via the existing LLM+tools agent
    "OUT_OF_SCOPE",         # friendly redirect; must never reach the LLM
    "CLARIFICATION_REQUIRED",  # confidence too low to safely act — ask, don't guess
    "SAFE_FALLBACK",        # router has no confident opinion — defer entirely to the pre-Phase-3 chain
}

# Maps a WORKFLOW_EXECUTING_INTENTS value to the workflow type string used
# elsewhere in the app (app/workflows/constants.py's WORKFLOW_* values).
_WORKFLOW_FOR_INTENT = {
    "registration_request": "onboarding",
    "transfer_request": "transfer",
    "loan_application_request": "loan",
    "cheque_deposit_request": "cheque",
    "kyc_update_request": "kyc",
}


class RoutingDecision(BaseModel):
    action: str
    workflow: Optional[str] = None
    reason: Optional[str] = None


def route_intent(intent_result: IntentResult, context: Optional[ConversationContext] = None) -> RoutingDecision:
    """Never raises — a routing failure defers to the pre-Phase-3 behavior
    (SAFE_FALLBACK) rather than breaking the turn."""
    try:
        return _route(intent_result, context)
    except Exception as e:
        logger.error(f"Routing decision failed | error={e}")
        return RoutingDecision(action="SAFE_FALLBACK", reason="router_error")


def _route(intent_result: IntentResult, context: Optional[ConversationContext]) -> RoutingDecision:
    intent = intent_result.intent
    confidence = intent_result.confidence

    # An active workflow was already given first refusal by
    # WorkflowManager.handle() before the router is ever consulted (see
    # run_agent()) — it only reaches here after explicitly punting a
    # conversational, in-scope question back (reprocess_query). That
    # in-scope check (_is_allowed_for_workflow) already ran, so it's safe
    # to let the existing LLM+tools agent explain — it has the active
    # workflow's context in its prompt and no tool that can execute or
    # advance that workflow, so it cannot bypass confirmation/OTP steps.
    if context and context.current_workflow:
        return RoutingDecision(action="WORKFLOW", workflow=context.current_workflow, reason="active_workflow_conversation")

    if intent == "out_of_scope":
        return RoutingDecision(action="OUT_OF_SCOPE", reason="out_of_scope")

    if intent in WORKFLOW_EXECUTING_INTENTS:
        # Intent classification alone must never authorize a financial
        # action — only start the workflow (which still requires its own
        # confirmation step before anything happens) at high confidence.
        # Below that, ask rather than guess.
        if confidence >= CONFIDENCE_HIGH:
            return RoutingDecision(action="START_WORKFLOW", workflow=_WORKFLOW_FOR_INTENT[intent], reason=intent)
        return RoutingDecision(action="CLARIFICATION_REQUIRED", reason=intent)

    if intent in NAVIGATION_INTENTS:
        # Navigation (cancel/back/menu/greeting/...) is already handled
        # deterministically upstream of the router (registration_gate.py,
        # workflows/manager.py's hard-navigation checks) — if one still
        # reaches here, defer to the existing chain rather than inventing
        # new behavior for something that already works.
        return RoutingDecision(action="SAFE_FALLBACK", reason="navigation_handled_upstream")

    if intent in LLM_ELIGIBLE_INTENTS:
        # Informational/question intents aren't a financial action, so
        # medium confidence is fine per the confidence policy — only truly
        # low confidence gets a clarification instead of a guessed answer.
        if confidence < CONFIDENCE_MEDIUM:
            return RoutingDecision(action="CLARIFICATION_REQUIRED", reason=intent)
        return RoutingDecision(action="BANKING_LLM", reason=intent)

    return RoutingDecision(action="SAFE_FALLBACK", reason="unmapped_intent")

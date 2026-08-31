"""Conversation router — shared types for the LLM-first routing decision.

ARCHITECTURAL RULE: this module only describes *where a turn should go*.
It never calls a banking tool, never starts/advances/completes a workflow
itself, and never touches the database. app/conversation/manager.py and
app/workflows/manager.py are the only things that act on a RoutingDecision,
and they do so entirely through the existing WorkflowManager / LLM+tools
mechanisms.

The rule-based route_intent()/confidence-threshold logic this module used
to own (Phase 3) was removed in the LLM-first routing migration
(docs/current_architecture.md, "Phase 13") — every message not caught by
the narrow deterministic pre-filter (app/conversation/intent/rules.py) is
now routed by a single LLM call
(app/conversation/intent/llm_routing.py::classify_and_route_llm), whose
own certainty band (not a numeric confidence threshold) gates whether a
workflow-starting decision is trusted. What remains here — RoutingDecision
and the intent-to-workflow table — is still the shared vocabulary that
LLMRoutingDecision.to_routing_decision() projects onto.
"""

from typing import Optional

from pydantic import BaseModel

# Maps a workflow-executing intent to the workflow type string used
# elsewhere in the app (app/workflows/constants.py's WORKFLOW_* values).
_WORKFLOW_FOR_INTENT = {
    "registration_request": "onboarding",
    "transfer_request": "transfer",
    "loan_application_request": "loan",
    "cheque_deposit_request": "cheque",
    "kyc_update_request": "kyc",
    "add_account_request": "add_account",
}


def get_workflow_for_intent(intent: str) -> Optional[str]:
    """Public accessor for _WORKFLOW_FOR_INTENT — used by
    LLMRoutingDecision.resolved_target_workflow() (app/conversation/intent/
    llm_routing.py) so a model response that only fills in `intent` still
    resolves to the right workflow without needing its own duplicate
    mapping."""
    return _WORKFLOW_FOR_INTENT.get(intent)


class RoutingDecision(BaseModel):
    """action is one of: GREETING, START_WORKFLOW, WORKFLOW, CANCEL,
    BANKING_LLM, OUT_OF_SCOPE, CLARIFICATION_REQUIRED, SAFE_FALLBACK — see
    app/conversation/manager.py and app/workflows/manager.py for what each
    one dispatches to. Not a validated enum here: the only producer is
    LLMRoutingDecision.to_routing_decision() (app/conversation/intent/
    llm_routing.py), which already constrains its own `action` field via
    LLM_ROUTING_ACTIONS before this class ever sees it."""

    action: str
    workflow: Optional[str] = None
    reason: Optional[str] = None

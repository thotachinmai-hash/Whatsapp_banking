"""Structured output model + intent taxonomy for the conversation intent
classifier. See docs/current_architecture.md,
"Conversation Intent Classifier — Phase 2".

This module only describes *what the user is trying to do*. It never
executes a banking action, never calls a banking tool, and never changes
workflow state — see app/conversation/intent/classifier.py for the
enforcement of that boundary.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

# ─── A. Navigation ────────────────────────────────────────────────────
NAVIGATION_INTENTS = {
    "greeting", "help", "back", "cancel", "main_menu", "repeat", "start_over",
}

# ─── B. Direct banking workflow requests ─────────────────────────────
WORKFLOW_REQUEST_INTENTS = {
    "registration_request", "transfer_request", "balance_request",
    "transaction_request", "cheque_deposit_request", "cheque_status_request",
    "loan_application_request", "kyc_update_request",
}

# ─── C. Banking questions / knowledge ─────────────────────────────────
BANKING_QUESTION_INTENTS = {
    "banking_question", "account_question", "transfer_question",
    "cheque_question", "loan_question", "kyc_question",
}

# ─── D. Personalized banking guidance ──────────────────────────────────
GUIDANCE_INTENTS = {
    "financial_guidance", "loan_eligibility_question", "transaction_insight_question",
}

# ─── E. Workflow conversation (requires ConversationContext) ──────────
WORKFLOW_CONVERSATION_INTENTS = {
    "workflow_help", "workflow_explanation", "workflow_clarification",
    "workflow_correction", "workflow_status", "workflow_confirmation",
}

# ─── F. Status / information requests ──────────────────────────────────
# Note: the cheque status case is deliberately represented by the single
# name "cheque_status_request" (from category B) rather than a second
# "cheque_status" spelling — see docs/current_architecture.md for why.
STATUS_INTENTS = {
    "account_information", "transfer_status", "loan_status", "kyc_status",
}

# ─── G/H. Out of scope / unknown ───────────────────────────────────────
OUT_OF_SCOPE_INTENTS = {"out_of_scope"}
UNKNOWN_INTENTS = {"unknown"}

ALL_INTENTS = frozenset().union(
    NAVIGATION_INTENTS,
    WORKFLOW_REQUEST_INTENTS,
    BANKING_QUESTION_INTENTS,
    GUIDANCE_INTENTS,
    WORKFLOW_CONVERSATION_INTENTS,
    STATUS_INTENTS,
    OUT_OF_SCOPE_INTENTS,
    UNKNOWN_INTENTS,
)

# Intents whose handling means "hand this turn to (or keep it with) an
# existing deterministic workflow". The classifier never acts on this
# itself — it is metadata for whichever future routing layer reads it.
WORKFLOW_EXECUTING_INTENTS = {
    "registration_request", "transfer_request", "cheque_deposit_request",
    "loan_application_request", "kyc_update_request",
}

# Intents that, today, are answered via the existing LLM+tools branch of
# run_agent() rather than a deterministic workflow or a hard-coded reply.
LLM_ELIGIBLE_INTENTS = {
    "balance_request", "transaction_request", "cheque_status_request",
    "loan_status", "transfer_status", "kyc_status", "account_information",
    "banking_question", "account_question", "transfer_question",
    "cheque_question", "loan_question", "kyc_question",
    "financial_guidance", "loan_eligibility_question", "transaction_insight_question",
    "unknown",
}

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.60


def confidence_band(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    if confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def flags_for_intent(intent: str) -> tuple[bool, bool]:
    """Single source of truth for requires_workflow/requires_llm per intent.

    Kept centralized (rather than set ad hoc by each rule) so the mapping
    can be audited/changed in one place. out_of_scope and every navigation
    intent resolve to (False, False) by omission — out_of_scope must never
    be marked as needing the general banking LLM.
    """
    requires_workflow = (
        intent in WORKFLOW_EXECUTING_INTENTS or intent in WORKFLOW_CONVERSATION_INTENTS
    )
    requires_llm = intent in LLM_ELIGIBLE_INTENTS
    return requires_workflow, requires_llm


class IntentResult(BaseModel):
    """The classifier's structured output for one message.

    This is a classification only — see the module docstring. `method`
    records how the result was reached (`rule`, `context`, or `llm`) for
    observability; it is not part of the taxonomy itself.
    """

    intent: str = "unknown"
    confidence: float = 0.0
    entities: dict[str, Any] = Field(default_factory=dict)
    requires_workflow: bool = False
    requires_llm: bool = False
    method: str = "rule"

    def confidence_band(self) -> str:
        return confidence_band(self.confidence)

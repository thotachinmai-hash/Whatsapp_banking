"""Structured output for the Banking Guidance Policy layer — Task 9.1.

A GuidanceResult only ever describes a *presentation intent* — how the
assistant should help (explain, guide, offer next actions, ask for
clarification, or redirect to the real workflow/status path). It never
represents a banking fact, a decision, or a completed action. See
policy.py's module docstring for the layer's full boundary.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GuidanceType(str, Enum):
    GENERAL_BANKING_GUIDANCE = "general_banking_guidance"
    LOAN_GUIDANCE = "loan_guidance"
    LOAN_ELIGIBILITY_GUIDANCE = "loan_eligibility_guidance"
    LOAN_DOCUMENT_GUIDANCE = "loan_document_guidance"
    TRANSFER_GUIDANCE = "transfer_guidance"
    CHEQUE_GUIDANCE = "cheque_guidance"
    CHEQUE_STATUS_GUIDANCE = "cheque_status_guidance"
    KYC_GUIDANCE = "kyc_guidance"
    ACCOUNT_GUIDANCE = "account_guidance"
    TRANSACTION_GUIDANCE = "transaction_guidance"
    WORKFLOW_HELP = "workflow_help"
    UNKNOWN = "unknown"


class ResponseMode(str, Enum):
    EXPLAIN = "explain"
    GUIDE = "guide"
    OFFER_ACTIONS = "offer_actions"
    ASK_CLARIFICATION = "ask_clarification"
    REDIRECT_TO_WORKFLOW = "redirect_to_workflow"


class SuggestedAction(str, Enum):
    """Presentation-only hints for what the reply could offer next — never
    executed by this layer; a future response/template layer would be the
    one to turn these into actual buttons/quick-replies, exactly like a
    workflow processor turns a RoutingDecision into a real workflow start."""

    CHECK_ELIGIBILITY = "check_eligibility"
    START_APPLICATION = "start_application"
    REQUIRED_DOCUMENTS = "required_documents"
    START_TRANSFER = "start_transfer"
    VIEW_BENEFICIARIES = "view_beneficiaries"
    DEPOSIT_CHEQUE = "deposit_cheque"
    CHECK_CHEQUE_STATUS = "check_cheque_status"
    START_KYC_UPDATE = "start_kyc_update"
    VIEW_ACCOUNT_SUMMARY = "view_account_summary"
    VIEW_TRANSACTIONS = "view_transactions"
    CONTINUE_WORKFLOW = "continue_workflow"


class GuidanceAction(str, Enum):
    """Stable action identifiers for guidance -> workflow handoff — Task 9.2.

    Never user-facing text (see app/conversation/guidance/responses.py's
    module docstring for why) — only ConversationContext.pending_action /
    allowed_actions and app/conversation/guidance/handoff.py read these.
    """

    START_TRANSFER = "START_TRANSFER"
    START_CHEQUE_DEPOSIT = "START_CHEQUE_DEPOSIT"
    START_LOAN_APPLICATION = "START_LOAN_APPLICATION"
    START_KYC_UPDATE = "START_KYC_UPDATE"
    SHOW_LOAN_REQUIREMENTS = "SHOW_LOAN_REQUIREMENTS"
    SHOW_LOAN_DOCUMENTS = "SHOW_LOAN_DOCUMENTS"
    SHOW_KYC_INFORMATION = "SHOW_KYC_INFORMATION"
    SHOW_CHEQUE_INFORMATION = "SHOW_CHEQUE_INFORMATION"
    SHOW_TRANSFER_INFORMATION = "SHOW_TRANSFER_INFORMATION"
    CANCEL = "CANCEL"
    BACK = "BACK"


class GuidanceResult(BaseModel):
    """The guidance policy's structured output for one message.

    `entities` is always a verbatim (or literally-present-in-text)
    pass-through — this layer never computes, guesses, or invents a
    value. `confidence` mirrors the confidence of the IntentResult this
    was derived from (or the guidance layer's own text-heuristic
    confidence for the narrow cases it resolves beyond the classifier —
    see policy.py).
    """

    guidance_type: GuidanceType = GuidanceType.UNKNOWN
    response_mode: ResponseMode = ResponseMode.ASK_CLARIFICATION
    suggested_actions: list[SuggestedAction] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0

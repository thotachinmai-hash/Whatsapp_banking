"""Smallest-necessary adapter for Phase 3 routing (see
docs/current_architecture.md, "Intent-Based Routing — Phase 3").

`WorkflowManager.start_requested()` remains the primary, unmodified
mechanism for starting a workflow from a message — the router's
START_WORKFLOW decision tries it first. This adapter exists only for the
gap where the intent classifier's phrasing coverage is broader than
`start_requested()`'s own keyword gate (e.g. "send 500 to Priya" doesn't
contain "transfer"/"send money"/"pay someone"/"make a payment", so
`start_requested()` alone wouldn't recognize it as a transfer). It is
called ONLY as a fallback when `start_requested()` returns handled=False.

It duplicates nothing new: each branch below is the same
create_workflow_model()/create_workflow() call and the same opening
message `start_requested()`'s own menu-digit branch already uses — see
app/workflows/manager.py::WorkflowManager.start_requested(). The transfer
branch defers entirely to start_transfer_from_text() (app/workflows/
processors/transfer.py), the same free-text parser start_requested()'s own
transfer branch uses, so a message that only reaches this adapter (because
it didn't match start_requested()'s narrower keyword gate) still gets its
beneficiary/amount parsed instead of being discarded.
"""

from typing import Any, Optional

from app.workflows.constants import (
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_CHEQUE,
    STEP_UPLOAD_KYC_FORM,
    WORKFLOW_ADD_ACCOUNT,
    WORKFLOW_CHEQUE,
    WORKFLOW_KYC,
    WORKFLOW_LOAN,
    WORKFLOW_ONBOARDING,
    WORKFLOW_TRANSFER,
)
from app.workflows.memory import create_workflow, create_workflow_model
from app.workflows.processors.transfer import start_transfer_from_text
from app.workflows.processors.onboarding import start_add_account_workflow
from app.workflows.processors.loan import detect_loan_type_from_text, loan_type_list_prompt
from app.conversation.responses.cheque import render_cheque_deposit_started
from app.conversation.responses.kyc import render_kyc_update_started


def start_workflow_directly(
    workflow_type: str,
    phone_number: str,
    transfer_handler: Optional[Any] = None,
    query: str = "",
    trace_id: str = "",
    entities: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Returns {"handled": True, "response": ...} for a supported
    workflow_type, or None if there's nothing extra for this adapter to do.

    `entities` is the LLM routing decision's own extracted-entities dict
    (app/conversation/intent/llm_routing.py::LLMRoutingDecision.entities) —
    passed through so a value the LLM already understood (e.g. loan_type
    from a native-script/romanized message English keyword matching can't
    read) is used directly instead of being silently discarded and
    re-derived by a narrower English-only regex. The regex/keyword
    extractors stay as the fallback for when entities don't include the
    value (a failed LLM call, or a caller with no decision at all).

    "onboarding" reaching this adapter specifically means the customer is
    ALREADY registered: registration_gate.py intercepts every message from
    an unregistered number before intent classification/routing ever run,
    so a "registration_request" intent that gets this far can only be a
    registered customer asking for a second/third account in free text
    ("I'd like to open another account") — start the add-account flow
    instead of the (inapplicable) fresh-registration one."""
    entities = entities or {}

    if workflow_type in (WORKFLOW_ONBOARDING, WORKFLOW_ADD_ACCOUNT):
        # add_account_request (a registered customer asking for an
        # additional account, e.g. "create another account") maps to the
        # same starter as the registration_request/"already registered"
        # case just above — see this function's own docstring for why
        # WORKFLOW_ONBOARDING reaching here always means the latter.
        return start_add_account_workflow(phone_number, trace_id)

    if workflow_type == WORKFLOW_TRANSFER:
        if transfer_handler is None:
            return None
        # Parse whatever beneficiary/amount the customer already gave in
        # the triggering message (e.g. "send 500 to Priya") instead of
        # discarding it and asking from scratch — see start_transfer_from_text's
        # docstring for why this lives there rather than being duplicated here.
        # entities is passed through for the same reason the loan branch
        # below does: a native-language/Romanized message's regex-free
        # beneficiary/amount extraction fails, but the LLM already
        # understood it.
        return start_transfer_from_text(phone_number, query, transfer_handler, trace_id, entities=entities)

    if workflow_type == WORKFLOW_CHEQUE:
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(phone_number, workflow)
        return {"handled": True, "response": render_cheque_deposit_started()}

    if workflow_type == WORKFLOW_LOAN:
        from app.workflows.processors.loan import LoanWorkflowHandler

        workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
        create_workflow(phone_number, workflow)
        # If the loan type was already stated in this same message ("I'd
        # like a personal loan"), skip straight past the "which type?"
        # step instead of asking again — matches the same behavior
        # start_requested()'s own (now-removed) free-text loan branch had.
        # Prefer the LLM's own extracted entity first (it understands
        # native-script/romanized/code-mixed phrasing this regex can't —
        # e.g. "నాకు వ్యక్తిగత ఋణం కావాలి") — detect_loan_type_from_text()
        # runs the SAME normalization against whichever string is present,
        # so "personal", "Personal Loan", "personal loan" all resolve the
        # same way regardless of source.
        loan_type = detect_loan_type_from_text(str(entities.get("loan_type", ""))) or detect_loan_type_from_text(query)
        if loan_type:
            return LoanWorkflowHandler()._select_type(workflow, phone_number, query, trace_id)
        return {"handled": True, "response": loan_type_list_prompt(
            "\U0001F4DD Let's get your loan application going! What kind of loan are you after?"
        )}

    if workflow_type == WORKFLOW_KYC:
        workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
        create_workflow(phone_number, workflow)
        return {"handled": True, "response": render_kyc_update_started()}

    return None

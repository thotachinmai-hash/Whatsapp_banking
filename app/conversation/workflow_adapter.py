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
    WORKFLOW_CHEQUE,
    WORKFLOW_KYC,
    WORKFLOW_LOAN,
    WORKFLOW_TRANSFER,
)
from app.workflows.memory import create_workflow, create_workflow_model
from app.workflows.processors.transfer import start_transfer_from_text
from app.conversation.responses.cheque import render_cheque_deposit_started
from app.conversation.responses.loan import render_loan_application_started
from app.conversation.responses.kyc import render_kyc_update_started


def start_workflow_directly(
    workflow_type: str,
    phone_number: str,
    transfer_handler: Optional[Any] = None,
    query: str = "",
    trace_id: str = "",
) -> Optional[dict[str, Any]]:
    """Returns {"handled": True, "response": ...} for a supported
    workflow_type, or None if there's nothing extra for this adapter to do
    (e.g. "onboarding" — registration_gate.py already starts that for any
    unregistered customer on any message, so a registered customer
    matching registration_request has nothing to start)."""

    if workflow_type == WORKFLOW_TRANSFER:
        if transfer_handler is None:
            return None
        # Parse whatever beneficiary/amount the customer already gave in
        # the triggering message (e.g. "send 500 to Priya") instead of
        # discarding it and asking from scratch — see start_transfer_from_text's
        # docstring for why this lives there rather than being duplicated here.
        return start_transfer_from_text(phone_number, query, transfer_handler, trace_id)

    if workflow_type == WORKFLOW_CHEQUE:
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(phone_number, workflow)
        return {"handled": True, "response": render_cheque_deposit_started()}

    if workflow_type == WORKFLOW_LOAN:
        workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
        create_workflow(phone_number, workflow)
        return {"handled": True, "response": render_loan_application_started()}

    if workflow_type == WORKFLOW_KYC:
        workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
        create_workflow(phone_number, workflow)
        return {"handled": True, "response": render_kyc_update_started()}

    return None

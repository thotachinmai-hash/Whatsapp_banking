"""Builds a ConversationContext from existing system state.

Reads the existing `workflow:{phone_number}` Redis state (via
app.workflows.memory.get_workflow, unchanged) and the existing customer
lookup (via app.database.get_customer_by_phone, unchanged) — this module
introduces no new source of truth. See docs/current_architecture.md,
"Conversation Context — Phase 1".
"""

from typing import Optional

from app.conversation.context import ConversationContext, sanitize_workflow_data
from app.conversation.context_store import ConversationContextStore
from app.database import get_customer_by_phone
from app.logger import get_logger
from app.workflows.memory import get_workflow

logger = get_logger(__name__)

_store = ConversationContextStore()


def build_context(phone_number: str, trace_id: str = "") -> ConversationContext:
    """
    Build (or refresh) a ConversationContext for this phone number from the
    existing workflow state and customer lookup.

    A customer-lookup failure is never treated as proof the customer is
    unregistered: once a context has recorded is_registered=True, that value
    is trusted for later turns without re-querying (registration is one-way
    in this system); if a lookup does fail before registration is known, the
    previous context's value (if any) is kept and the failure is recorded in
    last_error, rather than silently downgrading a customer to unregistered.
    """
    existing = _store.get(phone_number, trace_id=trace_id)
    context = existing or ConversationContext(phone_number=phone_number)

    workflow = get_workflow(phone_number)

    last_error: Optional[str] = None
    if existing and existing.is_registered:
        # Already confirmed registered on a previous turn — registration
        # doesn't get revoked, so skip the redundant DB lookup the
        # registration gate already performs this same turn.
        pass
    else:
        try:
            customer = get_customer_by_phone(phone_number)
            context.is_registered = customer is not None
            context.customer_id = customer.get("id") if customer else None
        except Exception as e:
            logger.error(
                f"[{trace_id}] Customer lookup failed while building conversation context | "
                f"phone={phone_number} | error={e}"
            )
            last_error = "customer_lookup_failed"
            # Deliberately do not touch context.is_registered here — keep
            # whatever registration state was already known (False by
            # default for a brand-new context) rather than asserting it.

    context.last_error = last_error

    if workflow:
        context.current_workflow = workflow.get("type")
        context.current_step = workflow.get("step")
        context.workflow_id = workflow.get("workflow_id")
        context.workflow_data = sanitize_workflow_data(workflow.get("data", {}))
    else:
        context.current_workflow = None
        context.current_step = None
        context.workflow_id = None
        context.workflow_data = {}

    return context

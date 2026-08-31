from typing import Any, Optional

from app.database import get_accounts_by_phone, get_customer_by_phone
from app.logger import get_logger
from app.memory import get_session_history
from app.conversation.intent.llm_routing import LLMRoutingDecision
from app.conversation.responses.common import render_main_menu_list
from app.services.menu import build_accounts_summary, build_onboarding_welcome_message
from app.workflows.constants import STEP_COLLECT_AADHAAR, WORKFLOW_ONBOARDING
from app.workflows.memory import create_workflow, create_workflow_model

logger = get_logger(__name__)

# The row ids of app/conversation/responses/common.py's _MAIN_MENU_ROWS /
# app/workflows/manager.py's start_requested() menu_actions. A tapped
# WhatsApp list row arrives as this bare digit — a purely structural
# protocol check, not a semantic one.
MENU_DIGITS = {"1", "2", "3", "4", "5", "6", "7", "8"}

# Decision actions that mean "the customer wants to actually use a real
# banking service" (as opposed to a pure greeting, a general question, or
# something out of scope) — the deciding factor for whether an
# unregistered customer must register first. Driven by the single LLM
# routing decision, not keyword-sniffing the raw text.
_SERVICE_ACTIONS = {"START_WORKFLOW", "SWITCH", "TOOL"}


def check_registration_gate(
    phone_number: str,
    query: str,
    decision: Optional[LLMRoutingDecision],
    is_registered: bool,
    trace_id: str = "",
) -> dict[str, Any] | None:
    """
    Runs before workflow handling and before the LLM agent, only when no
    workflow is already active (the caller — ConversationManager — skips
    this entirely once a workflow owns the turn).

    `decision` is the SAME single LLM routing decision computed once for
    this turn (or None if the deterministic pre-filter already resolved
    the message without needing one) — this function never classifies the
    message itself.

    Returns None if the caller should proceed as normal (workflow manager,
    then LLM agent). Returns {"handled": True, "response": "..."} if this
    turn is fully handled by the gate.
    """

    query = str(query or "")
    action = decision.action if decision is not None else None

    if is_registered:
        customer = get_customer_by_phone(phone_number)
        if not customer:
            # is_registered was true from a stale/cached context value but
            # the customer record is genuinely gone — defer to the normal
            # flow rather than crash on customer["full_name"] below.
            return None

        history = get_session_history(phone_number)

        # A tapped main-menu row (bare digit "1".."8") must never be
        # swallowed by the "first message ever" greeting fallback below —
        # otherwise a registered customer's very first message being a
        # menu tap (or any tap after their session history has expired)
        # just re-shows the same menu instead of acting on it.
        is_menu_tap = query.strip() in MENU_DIGITS
        is_first_message_with_no_clear_request = not history and action not in _SERVICE_ACTIONS and not is_menu_tap
        if action == "GREETING" or is_first_message_with_no_clear_request:
            logger.info(
                f"[{trace_id}] Registration gate | greeting shown | phone={phone_number[-4:]}"
            )
            return {
                "handled": True,
                "response": render_main_menu_list(
                    customer["full_name"], greeting=True,
                    prefix=build_accounts_summary(get_accounts_by_phone(phone_number)),
                ),
            }

        return None

    # Unregistered customer.
    if action not in ("GREETING",) and action not in _SERVICE_ACTIONS:
        # A general/informational question ("what documents do I need for
        # a loan"), a RAG-eligible banking question, an out-of-scope
        # message, or a genuinely ambiguous one — answer it (via RAG/the
        # LLM further down the pipeline) rather than force-starting
        # onboarding on every single message. Only a greeting or an actual
        # attempt to use a service requires registering first.
        logger.info(
            f"[{trace_id}] Registration gate | unregistered, question — deferring registration | "
            f"phone={phone_number[-4:]}"
        )
        return None

    logger.info(
        f"[{trace_id}] Registration gate | unregistered | phone={phone_number[-4:]}"
    )

    workflow = create_workflow_model(
        workflow_type=WORKFLOW_ONBOARDING,
        step=STEP_COLLECT_AADHAAR,
    )
    if action in _SERVICE_ACTIONS:
        # Remember what they actually wanted so it can be resumed
        # automatically once registration finishes — see
        # WorkflowManager.resume_pending_request().
        workflow["data"]["pending_service_query"] = query
    create_workflow(phone_number, workflow)

    return {
        "handled": True,
        "response": build_onboarding_welcome_message(),
    }

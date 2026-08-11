from typing import Any

from app.database import get_accounts_by_phone, get_customer_by_phone
from app.logger import get_logger
from app.memory import get_session_history
from app.conversation.responses.common import render_main_menu_list
from app.services.menu import build_accounts_summary, build_onboarding_welcome_message
from app.workflows.constants import STEP_COLLECT_AADHAAR, WORKFLOW_ONBOARDING
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow

logger = get_logger(__name__)

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "hii", "hiya",
    "menu", "start", "help", "services",
}


def _is_greeting(query: str) -> bool:
    normalized = query.strip().lower().strip("!.? ")
    return normalized in GREETING_KEYWORDS


async def check_registration_gate(
    phone_number: str,
    query: str,
    trace_id: str = "",
) -> dict[str, Any] | None:
    """
    Runs before workflow handling and before the LLM agent.

    Returns None if the caller should proceed as normal (workflow manager,
    then LLM agent). Returns {"handled": True, "response": "..."} if this
    turn is fully handled by the gate.
    """

    if get_workflow(phone_number):
        # A workflow (onboarding, cheque, etc.) already owns this turn.
        return None

    customer = get_customer_by_phone(phone_number)

    if customer:
        history = get_session_history(phone_number)

        if _is_greeting(query) or (not history and not _is_service_request(query)):
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

    if not _is_greeting(query) and not _is_unregistered_service_request(query):
        # An unregistered customer asking a general/informational question
        # ("what documents do I need for a loan") should get an answer —
        # via RAG/the LLM further down the pipeline — rather than being
        # force-started into onboarding on every single message. Only a
        # greeting or an actual attempt to use a service (checked above)
        # requires registering first.
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
    if not _is_greeting(query) and _is_unregistered_service_request(query):
        # Remember what they actually wanted so it can be resumed
        # automatically once registration finishes — see
        # WorkflowManager.handle()'s WORKFLOW_ONBOARDING branch.
        workflow["data"]["pending_service_query"] = query
    create_workflow(phone_number, workflow)

    return {
        "handled": True,
        "response": build_onboarding_welcome_message(),
    }


def _looks_like_question(query: str) -> bool:
    """Mirrors the same guard app/conversation/intent/rules.py's
    classify_workflow_request() already applies — a banking keyword inside
    a question ("what documents do I need for a home loan?") must not be
    read as a request to actually start that service."""
    stripped = query.strip()
    if stripped.endswith("?"):
        return True
    normalized = stripped.lower()
    return normalized.startswith((
        "what", "how", "why", "can i", "could i", "is it", "does",
        "do i", "am i", "when", "where", "will i",
    ))


def _is_service_request(query: str) -> bool:
    text = query.strip().lower()
    return any(term in text for term in (
        "loan", "borrow", "finance", "kyc", "know your customer",
        "cheque", "check", "balance", "transaction", "transfer",
    ))


def _is_unregistered_service_request(query: str) -> bool:
    """Stricter than _is_service_request() above — that helper also decides
    whether to show a REGISTERED first-time customer the welcome menu, a
    different decision where a keyword-only match is fine either way. Here
    the stakes are higher (force registration vs. let a question through),
    so a question containing a banking keyword ("what documents do I need
    for a home loan?") must not count as a request to actually start that
    service."""
    return _is_service_request(query) and not _looks_like_question(query)

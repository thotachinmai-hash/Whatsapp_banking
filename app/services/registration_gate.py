from typing import Any

from app.database import get_accounts_by_phone, get_customer_by_phone
from app.logger import get_logger
from app.memory import get_session_history
from app.services.menu import build_accounts_summary, build_menu_response
from app.workflows.constants import STEP_COLLECT_NAME, WORKFLOW_ONBOARDING
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
                "response": (
                    build_accounts_summary(get_accounts_by_phone(phone_number))
                    + build_menu_response(customer["full_name"])
                ),
            }

        return None

    logger.info(
        f"[{trace_id}] Registration gate | unregistered | phone={phone_number[-4:]}"
    )

    workflow = create_workflow_model(
        workflow_type=WORKFLOW_ONBOARDING,
        step=STEP_COLLECT_NAME,
    )
    create_workflow(phone_number, workflow)

    return {
        "handled": True,
        "response": (
            "👋 Welcome to HSBC WhatsApp Banking!\n\n"
            "You're not registered with us yet. Let's get you set up — "
            "it only takes a minute.\n\n"
            "First, what is your full name?"
        ),
    }


def _is_service_request(query: str) -> bool:
    text = query.strip().lower()
    return any(term in text for term in (
        "loan", "borrow", "finance", "kyc", "know your customer",
        "cheque", "check", "balance", "transaction", "transfer",
    ))

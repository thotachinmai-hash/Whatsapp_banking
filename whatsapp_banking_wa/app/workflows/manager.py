from typing import Any

from app.logger import get_logger
from app.workflows.constants import (
    WORKFLOW_CHEQUE,
    WORKFLOW_LOAN,
    WORKFLOW_KYC,
    WORKFLOW_ONBOARDING,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_KYC_FORM,
    STEP_UPLOAD_CHEQUE,
)
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow, update_workflow_data

from app.workflows.processors.cheque import ChequeWorkflowProcessor
from app.workflows.processors.loan import LoanWorkflowHandler
from app.workflows.processors.kyc import KYCWorkflowHandler
from app.workflows.processors.onboarding import OnboardingWorkflowHandler

logger = get_logger(__name__)


class WorkflowManager:
    """
    Routes incoming requests to the appropriate workflow handler
    if the customer has an active workflow.
    """

    def __init__(self):

        self.cheque_handler = ChequeWorkflowProcessor()
        self.loan_handler = LoanWorkflowHandler()
        self.kyc_handler = KYCWorkflowHandler()
        self.onboarding_handler = OnboardingWorkflowHandler()

    async def handle(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None = None,
    ) -> dict[str, Any]:
        """
        Handle an active workflow.

        Returns:
        {
            "handled": True/False,
            "response": "..."
        }
        """

        workflow = get_workflow(phone_number)

        if workflow is None:

            logger.info(
                f"No active workflow found | phone={phone_number}"
            )

            return {
                "handled": False,
                "response": None,
            }

        workflow_type = workflow["type"]

        # An incomplete document workflow must not swallow unrelated requests.
        # Ask for explicit confirmation before abandoning the customer's data.
        pending = workflow.get("data", {}).get("pending_interrupt")
        if pending:
            answer = query.strip().lower()
            if answer in {"yes", "y", "cancel", "cancel it", "continue"}:
                from app.workflows.memory import complete_workflow
                interrupted_query = pending
                complete_workflow(phone_number)
                return {
                    "handled": False,
                    "response": None,
                    "reprocess_query": interrupted_query,
                }
            if answer in {"no", "n", "keep", "continue cheque", "continue loan", "continue kyc"}:
                from app.workflows.memory import clear_workflow_data
                clear_workflow_data(phone_number, "pending_interrupt")
                return {
                    "handled": True,
                    "response": f"Okay, we will continue your {workflow_type} process. Please provide the requested details.",
                }

        if (
            workflow_type in {WORKFLOW_CHEQUE, WORKFLOW_LOAN, WORKFLOW_KYC}
            and not _is_current_workflow_input(workflow, query)
            and _looks_like_new_service_request(query)
        ):
            update_workflow_data(phone_number, {"pending_interrupt": query})
            return {
                "handled": True,
                "response": (
                    f"You have an active {workflow_type} process with pending information. "
                    "Would you like to cancel it and start this new request? Reply YES to cancel or NO to continue."
                ),
            }

        logger.info(
            f"Active workflow found | "
            f"type={workflow_type} | "
            f"step={workflow['step']}"
        )

        if workflow_type == WORKFLOW_CHEQUE:

            return await self.cheque_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
            )

        elif workflow_type == WORKFLOW_LOAN:

            return await self.loan_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
            )

        elif workflow_type == WORKFLOW_KYC:

            return await self.kyc_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
            )

        elif workflow_type == WORKFLOW_ONBOARDING:

            return await self.onboarding_handler.handle(
                workflow=workflow,
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
            )

        logger.warning(
            f"Unknown workflow type: {workflow_type}"
        )

        return {
            "handled": False,
            "response": None,
        }

    def start_requested(self, phone_number: str, query: str) -> dict[str, Any]:
        """Start deterministic workflows without depending on an LLM intent call."""
        normalized = query.strip().lower()
        if any(word in normalized for word in ("cheque", "check deposit", "deposit a check")):
            workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
            create_workflow(phone_number, workflow)
            return {
                "handled": True,
                "response": "Cheque deposit started. Please upload a clear cheque image to continue.",
            }
        if any(word in normalized for word in ("loan", "borrow", "finance")):
            workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
            create_workflow(phone_number, workflow)
            return {
                "handled": True,
                "response": (
                    "Loan application started. Available loan types:\n\n"
                    "1. Personal Loan\n2. Home Loan\n3. Vehicle Loan\n4. Education Loan\n\n"
                    "Reply with the number or name. I will then ask you to complete and upload the loan form."
                ),
            }
        if any(word in normalized for word in ("kyc", "know your customer", "update my details")):
            workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
            create_workflow(phone_number, workflow)
            return {
                "handled": True,
                "response": (
                    "KYC update started. Please upload a clear KYC form or document.\n\n"
                    "Required details: full name, date of birth, address, Aadhaar number, and PAN number."
                ),
            }
        return {"handled": False, "response": None}


def _looks_like_new_service_request(query: str) -> bool:
    text = query.strip().lower()
    return any(term in text for term in (
        "balance", "transaction", "statement", "loan", "kyc", "cheque", "menu",
        "help", "account", "transfer", "service",
    ))


def _is_current_workflow_input(workflow: dict[str, Any], query: str) -> bool:
    """Do not treat a field correction as a request to abandon its workflow."""
    if workflow.get("type") != WORKFLOW_LOAN:
        return False
    fields = (
        "applicant", "name", "income", "monthly income", "salary", "employment", "amount",
        "loan amount", "requested", "tenure", "loan tenure", "purpose", "loan purpose",
    )
    return any(
        any(line.lstrip().lower().startswith(f"{field}:") for field in fields)
        for line in query.splitlines()
    )

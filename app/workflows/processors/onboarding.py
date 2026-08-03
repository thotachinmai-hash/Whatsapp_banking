import re
from typing import Any

import psycopg2

from app.database import create_customer, create_zero_balance_account
from app.logger import get_logger
from app.memory import cache_active_account
from app.services.menu import build_menu_response
from app.workflows.constants import (
    STEP_COLLECT_NAME,
    STEP_COLLECT_AADHAAR,
    STEP_COLLECT_PAN,
    STEP_CONFIRM_REGISTRATION,
    STEP_SELECT_ACCOUNT_TYPE,
)
from app.workflows.memory import (
    complete_workflow,
    set_workflow_step,
    update_workflow_data,
)

logger = get_logger(__name__)

AADHAAR_PATTERN = re.compile(r"^\d{12}$")
PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


class OnboardingWorkflowHandler:
    """
    Collects the name over text, then Aadhaar and PAN as images processed by
    the document parser, before creating the customer record.
    """

    async def handle(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        parsed_document: dict | None = None,
    ) -> dict[str, Any]:

        step = workflow["step"]

        if step == STEP_COLLECT_NAME:
            return self._handle_collect_name(phone_number, query)

        elif step == STEP_COLLECT_AADHAAR:
            return self._handle_collect_aadhaar(phone_number, query, parsed_document)

        elif step == STEP_COLLECT_PAN:
            return self._handle_collect_pan(phone_number, query, parsed_document)

        elif step == STEP_CONFIRM_REGISTRATION:
            return self._handle_confirm_registration(workflow, phone_number, query)

        elif step == STEP_SELECT_ACCOUNT_TYPE:
            return self._handle_select_account_type(workflow, phone_number, query)

        return {
            "handled": True,
            "response": "Unknown registration step."
        }

    def _handle_collect_name(self, phone_number: str, query: str) -> dict[str, Any]:

        name = query.strip()

        if len(name) < 3 or not re.match(r"^[A-Za-z .'-]+$", name):
            return {
                "handled": True,
                "response": (
                    "That doesn't look like a valid name. "
                    "Please enter your full name (letters only)."
                )
            }

        update_workflow_data(phone_number, {"full_name": name})
        set_workflow_step(phone_number, STEP_COLLECT_AADHAAR)

        return {
            "handled": True,
            "response": (
                f"Thanks, {name}. Now please upload a clear image of your Aadhaar card."
            )
        }

    def _handle_collect_aadhaar(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None,
    ) -> dict[str, Any]:

        if not parsed_document or not parsed_document.get("mime_type", "").startswith("image/"):
            return {
                "handled": True,
                "response": "Please upload a clear image of your Aadhaar card (not the number as text)."
            }

        digits = re.sub(r"[\s-]", "", self._document_value(
            parsed_document.get("content"),
            "aadhaar_number", "aadhaar", "aadhar_number", "aadhar",
        ))

        if not AADHAAR_PATTERN.match(digits):
            return {
                "handled": True,
                "response": (
                    "I couldn't read a valid 12-digit Aadhaar number from that image. "
                    "Please upload a clearer image of the Aadhaar card."
                )
            }

        update_workflow_data(phone_number, {"aadhaar_number": digits})
        set_workflow_step(phone_number, STEP_COLLECT_PAN)

        return {
            "handled": True,
            "response": (
                "Got it. Now please upload a clear image of your PAN card."
            )
        }

    def _handle_collect_pan(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None,
    ) -> dict[str, Any]:

        if not parsed_document or not parsed_document.get("mime_type", "").startswith("image/"):
            return {
                "handled": True,
                "response": "Please upload a clear image of your PAN card (not the number as text)."
            }

        pan = re.sub(r"\s", "", self._document_value(
            parsed_document.get("content"),
            "pan_number", "pan", "pan_card_number",
        )).upper()

        if not PAN_PATTERN.match(pan):
            return {
                "handled": True,
                "response": (
                    "I couldn't read a valid PAN number from that image. "
                    "Please upload a clearer image of the PAN card."
                )
            }

        update_workflow_data(phone_number, {"pan_number": pan})
        set_workflow_step(phone_number, STEP_CONFIRM_REGISTRATION)

        return {
            "handled": True,
            "response": (
                "Please confirm your details:\n\n"
                f"PAN: {pan}\n"
                "Aadhaar: (on file)\n\n"
                "Reply *YES* to confirm and complete registration, or *NO* to start over."
            )
        }

    def _handle_confirm_registration(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
    ) -> dict[str, Any]:

        answer = query.strip().lower()

        if answer in {"yes", "y", "confirm"}:

            data = workflow.get("data", {})

            try:
                customer = create_customer(
                    phone_number=phone_number,
                    full_name=data.get("full_name", ""),
                    aadhaar_number=data.get("aadhaar_number", ""),
                    pan_number=data.get("pan_number", ""),
                )
            except psycopg2.Error as e:
                logger.error(
                    f"Registration failed | phone={phone_number} | error={e}"
                )
                complete_workflow(phone_number)
                return {
                    "handled": True,
                    "response": (
                        "❌ We couldn't complete your registration — this Aadhaar or PAN "
                        "may already be registered. Please contact support."
                    )
                }

            set_workflow_step(phone_number, STEP_SELECT_ACCOUNT_TYPE)

            return {
                "handled": True,
                "response": (
                    "✅ Registration complete!\n\n"
                    "Which account would you like to open?\n"
                    "1. Savings Account\n2. Current Account\n3. Salary Account\n\n"
                    "Reply with 1, 2, or the account type."
                )
            }

        elif answer in {"no", "n", "restart"}:
            complete_workflow(phone_number)
            return {
                "handled": True,
                "response": "No problem — send any message to start registration again."
            }

        return {
            "handled": True,
            "response": "Please reply *YES* to confirm or *NO* to start over."
        }

    def _handle_select_account_type(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
    ) -> dict[str, Any]:
        aliases = {
            "1": "savings", "savings": "savings", "savings account": "savings",
            "2": "current", "current": "current", "current account": "current",
            "3": "salary", "salary": "salary", "salary account": "salary",
        }
        account_type = aliases.get(query.strip().lower())
        if not account_type:
            return {
                "handled": True,
                "response": (
                    "Please choose one of the available account types:\n"
                    "1. Savings Account\n2. Current Account\n3. Salary Account"
                ),
            }

        data = workflow.get("data", {})
        try:
            account = create_zero_balance_account(
                phone_number=phone_number,
                account_holder=data.get("full_name", ""),
                account_type=account_type,
            )
            if not account:
                raise RuntimeError("Account creation returned no account")
            cache_active_account(phone_number, account)
        except Exception as e:
            logger.error(f"Account creation failed | phone={phone_number} | error={e}")
            return {
                "handled": True,
                "response": "We couldn't open the account right now. Please try again shortly.",
            }

        complete_workflow(phone_number)
        return {
            "handled": True,
            "response": (
                "Account opened successfully!\n\n"
                f"Account Number: {account['account_number']}\n"
                f"Account Type: {account_type.title()} Account\n"
                "Balance: GBP 0.00\n\n"
                "Your Relationship Manager will contact you via email. "
                "For any queries, contact finacle@infi.com.\n\n"
                + build_menu_response(data.get("full_name", "Customer"), greeting=False)
            ),
        }

    @staticmethod
    def _document_value(content: Any, *preferred_keys: str) -> str:
        """Find an OCR value in structured or nested document-parser output."""
        if isinstance(content, dict):
            normalized = {
                re.sub(r"[^a-z0-9]", "", str(key).lower()): value
                for key, value in content.items()
            }
            for key in preferred_keys:
                value = normalized.get(re.sub(r"[^a-z0-9]", "", key.lower()))
                if value is not None and not isinstance(value, (dict, list)):
                    return str(value)
            for value in content.values():
                found = OnboardingWorkflowHandler._document_value(value, *preferred_keys)
                if found:
                    return found
        elif isinstance(content, list):
            for value in content:
                found = OnboardingWorkflowHandler._document_value(value, *preferred_keys)
                if found:
                    return found
        elif content is not None:
            text = str(content)
            if "aadhaar" in " ".join(preferred_keys) or "aadhar" in " ".join(preferred_keys):
                match = re.search(r"(?<!\d)(?:\d[ -]?){12}(?!\d)", text)
                return re.sub(r"[ -]", "", match.group(0)) if match else ""
            match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text.upper())
            return match.group(0) if match else ""
        return ""

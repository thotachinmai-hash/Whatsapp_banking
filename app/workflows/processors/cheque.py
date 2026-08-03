import uuid
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import psycopg2

from app.database import create_cheque_request
from app.logger import get_logger
from app.workflows.constants import STEP_UPLOAD_CHEQUE, STEP_CORRECT_CHEQUE
from app.workflows.memory import complete_workflow, set_workflow_step, update_workflow_data

logger = get_logger(__name__)

MANDATORY_FIELDS = ["payee", "amount_in_figures"]

FIELD_LABELS = {
    "payee": "Payee",
    "amount_in_figures": "Amount",
    "bank_name": "Bank",
    "branch": "Branch",
    "amount_in_words": "Amount (Words)",
    "numbers": "Cheque Number",
    "signatory_title": "Signatory",
}

# Free-text "Key: value" correction keys mapped onto the extracted content keys.
CORRECTION_KEY_MAP = {
    "payee": "payee",
    "pay to": "payee",
    "amount": "amount_in_figures",
    "amount in figures": "amount_in_figures",
    "amount in words": "amount_in_words",
    "bank": "bank_name",
    "bank name": "bank_name",
    "branch": "branch",
    "cheque number": "numbers",
    "cheque numbers": "numbers",
    "signatory": "signatory_title",
}


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "not detected", "none", "n/a")


def _missing_fields(content: dict) -> list[str]:
    return [field for field in MANDATORY_FIELDS if not _is_present(content.get(field))]


def _invalid_fields(content: dict) -> list[str]:
    amount = content.get("amount_in_figures")
    if not _is_present(amount):
        return []
    parsed = _parse_amount(amount)
    if parsed is None or parsed <= 0:
        return ["amount_in_figures"]
    return []


def _parse_amount(value: Any) -> Decimal | None:
    """Parse common human/OCR cheque amount formats into a Decimal."""
    if value is None:
        return None

    text = str(value).strip().upper()
    # Handle Indian and other common cheque notation: 50,000/-, Rs. 50,000,
    # INR 50,000.00, ₹50,000, £50,000, etc.
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^(?:INR|RS\.?|GBP|USD|EUR|₹|£|\$)", "", text)
    text = re.sub(r"(?:/[-–—]|/-)$", "", text)
    text = text.replace(",", "")

    if not re.fullmatch(r"\d+(?:\.\d{1,2})?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _normalized_amount(value: Any) -> str | None:
    parsed = _parse_amount(value)
    if parsed is None:
        return None
    return f"{parsed:.2f}"


def _normalize_content(content: dict) -> dict:
    """
    The OCR/document parser returns "pay_to" for the payee field. Normalize
    it to "payee" (the field name used everywhere else in this workflow and
    in cheque_requests) without losing the original key.
    """
    if not isinstance(content, dict):
        return {}

    normalized = dict(content)
    if "payee" not in normalized or not _is_present(normalized.get("payee")):
        if _is_present(normalized.get("pay_to")):
            normalized["payee"] = normalized["pay_to"]
    aliases = {
        "amount": "amount_in_figures",
        "amount_figures": "amount_in_figures",
        "cheque_number": "numbers",
        "check_number": "numbers",
        "signatory": "signatory_title",
        "bank": "bank_name",
    }
    for source, target in aliases.items():
        if not _is_present(normalized.get(target)) and _is_present(normalized.get(source)):
            normalized[target] = normalized[source]
    if _is_present(normalized.get("amount_in_figures")):
        amount = _normalized_amount(normalized["amount_in_figures"])
        if amount is not None:
            normalized["amount_in_figures"] = amount
    return normalized


class ChequeWorkflowProcessor:
    """
    Handles the cheque deposit workflow.
    """

    async def handle(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        parsed_document: dict | None = None,
    ) -> dict[str, Any]:
        """
        Continue an active cheque workflow.
        """

        step = workflow["step"]

        if step == STEP_UPLOAD_CHEQUE:
            return await self._handle_upload_cheque(
                workflow,
                phone_number,
                parsed_document,
            )

        elif step == STEP_CORRECT_CHEQUE:
            return await self._handle_correct_cheque(
                workflow,
                phone_number,
                query,
                parsed_document,
            )

        return {
            "handled": True,
            "response": "Unknown cheque workflow step."
        }

    async def _handle_upload_cheque(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        parsed_document: dict | None,
    ) -> dict[str, Any]:
        """
        Process the uploaded cheque.
        """

        if parsed_document is None:
            return {
                "handled": True,
                "response": (
                    "Please upload the cheque image to continue your cheque deposit."
                )
            }

        mime_type = parsed_document.get("mime_type")
        if mime_type and not mime_type.startswith("image/"):
            return {
                "handled": True,
                "response": "Please upload the cheque as a clear image (JPG, PNG, or WEBP).",
            }

        if not parsed_document.get("success"):
            return {
                "handled": True,
                "response": (
                    "❌ Unable to process the uploaded cheque.\n\n"
                    f"Reason: {parsed_document.get('error', 'Unknown error')}\n\n"
                    "Please upload a clear cheque image and try again."
                )
            }

        content = _normalize_content(parsed_document.get("content", {}))

        return self._validate_or_finalize(phone_number, content)

    async def _handle_correct_cheque(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        parsed_document: dict | None,
    ) -> dict[str, Any]:
        """
        Handle correction of a cheque with missing mandatory fields — either
        a re-uploaded image or free-text "Key: value" corrections.
        """

        content = dict(workflow.get("data", {}).get("partial_content", {}))

        if parsed_document is not None:

            mime_type = parsed_document.get("mime_type")
            if mime_type and not mime_type.startswith("image/"):
                return {
                    "handled": True,
                    "response": "Please upload the cheque as a clear image (JPG, PNG, or WEBP).",
                }

            if not parsed_document.get("success"):
                return {
                    "handled": True,
                    "response": (
                        "❌ Unable to process the uploaded cheque.\n\n"
                        f"Reason: {parsed_document.get('error', 'Unknown error')}\n\n"
                        "Please upload a clear cheque image and try again."
                    )
                }

            new_content = _normalize_content(parsed_document.get("content", {}))

            for key, value in new_content.items():
                if _is_present(value):
                    content[key] = value

            return self._validate_or_finalize(phone_number, content)

        if query and query.strip():

            updated_any = False

            for line in query.strip().splitlines():

                if ":" not in line:
                    continue

                key_raw, value_raw = line.split(":", 1)
                key = key_raw.strip().lower()
                value = value_raw.strip()

                content_key = CORRECTION_KEY_MAP.get(key)

                if content_key and _is_present(value):
                    content[content_key] = value
                    updated_any = True

            if not updated_any:
                return {
                    "handled": True,
                    "response": (
                        "I couldn't read those details. Please either re-upload a "
                        "clearer cheque image, or reply using this format:\n\n"
                        "Payee: John Smith\n"
                        "Amount: 500.00"
                    )
                }

            return self._validate_or_finalize(phone_number, content)

        return {
            "handled": True,
            "response": (
                "Please re-upload a clearer cheque image, or provide the missing "
                "details as text (e.g. `Payee: John Smith`)."
            )
        }

    def _validate_or_finalize(
        self,
        phone_number: str,
        content: dict,
    ) -> dict[str, Any]:

        missing = list(dict.fromkeys(
            _missing_fields(content) + _invalid_fields(content)
        ))

        logger.info(
            f"Cheque validation | phone={phone_number[-4:]} | missing_or_invalid={missing}"
        )

        if missing:

            update_workflow_data(phone_number, {"partial_content": content})
            set_workflow_step(phone_number, STEP_CORRECT_CHEQUE)

            missing_labels = ", ".join(FIELD_LABELS.get(f, f) for f in missing)

            return {
                "handled": True,
                "response": (
                    "⚠️ I couldn't detect the following required field(s) on your cheque: "
                    f"{missing_labels}.\n\n"
                    "Please re-upload a clearer image, or reply with the missing "
                    "details, e.g.:\n\n"
                    "Payee: John Smith\n"
                    "Amount: 500.00"
                )
            }

        return self._finalize_cheque_request(phone_number, content)

    def _finalize_cheque_request(
        self,
        phone_number: str,
        content: dict,
    ) -> dict[str, Any]:

        bank_name = content.get("bank_name") if _is_present(content.get("bank_name")) else None
        branch = content.get("branch") if _is_present(content.get("branch")) else None
        payee = content.get("payee")
        amount_figures = content.get("amount_in_figures")
        amount_words = content.get("amount_in_words") if _is_present(content.get("amount_in_words")) else None
        cheque_numbers = content.get("numbers") if _is_present(content.get("numbers")) else None
        signatory = content.get("signatory_title") if _is_present(content.get("signatory_title")) else None

        request_id = ""
        for _ in range(3):
            candidate = f"CHQ-{uuid.uuid4().hex[:8].upper()}"
            try:
                create_cheque_request(
                    request_id=candidate,
                    phone_number=phone_number,
                    bank_name=bank_name,
                    branch=branch,
                    payee=payee,
                    amount_in_figures=amount_figures,
                    amount_in_words=amount_words,
                    cheque_number=cheque_numbers,
                    signatory=signatory,
                    status="PENDING",
                )
                request_id = candidate
                break
            except psycopg2.errors.UniqueViolation:
                logger.warning("Cheque request ID collision; retrying")
            except psycopg2.Error as error:
                logger.error(f"Cheque request persistence failed | phone={phone_number} | error={error}")
                return {
                    "handled": True,
                    "response": "I validated the cheque, but could not save the request right now. Please try again shortly.",
                }

        if not request_id:
            raise RuntimeError("Unable to generate a unique cheque request ID")

        complete_workflow(phone_number)

        return {
            "handled": True,
            "response": (
                "✅ Cheque deposit request created!\n\n"
                f"🆔 Request ID: {request_id}\n"
                f"👤 Payee: {payee}\n"
                f"💰 Amount: {amount_figures}\n"
                f"🏦 Bank: {bank_name or 'Not detected'}\n"
                f"📌 Status: PENDING\n\n"
                "You can check the status anytime — just ask, e.g. "
                f'"check status of {request_id}".'
            )
        }

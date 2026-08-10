import uuid
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import psycopg2

from app.database import create_cheque_request, get_customer_by_phone
from app.logger import get_logger
from app.workflows.constants import STEP_UPLOAD_CHEQUE, STEP_CORRECT_CHEQUE
from app.workflows.memory import complete_workflow, set_workflow_step, update_workflow_data
from app.conversation.responses import cheque as templates

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
    "date_written": "Date",
    "drawer_name": "Drawer",
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


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    return text


def _payee_matches_customer(content: dict, phone_number: str) -> bool:
    payee = content.get("payee")
    if not _is_present(payee):
        return False

    customer = get_customer_by_phone(phone_number)
    if not customer:
        return True

    stored_name = customer.get("full_name") or ""
    return _normalize_name(stored_name) == _normalize_name(payee)


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
        "date": "date_written",
        "cheque_date": "date_written",
        "drawer": "drawer_name",
        "issuer_name": "drawer_name",
        "written_by": "drawer_name",
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
        trace_id: str = "",
    ) -> dict[str, Any]:
        """
        Continue an active cheque workflow.
        """

        step = workflow["step"]
        logger.info(f"[{trace_id}] Cheque workflow step | phone={phone_number[-4:]} | step={step}")

        if step == STEP_UPLOAD_CHEQUE:
            return await self._handle_upload_cheque(
                workflow,
                phone_number,
                parsed_document,
                trace_id,
            )

        elif step == STEP_CORRECT_CHEQUE:
            return await self._handle_correct_cheque(
                workflow,
                phone_number,
                query,
                parsed_document,
                trace_id,
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
        trace_id: str = "",
    ) -> dict[str, Any]:
        """
        Process the uploaded cheque.
        """

        if parsed_document is None:
            return {
                "handled": True,
                "response": templates.render_cheque_upload_prompt()
            }

        mime_type = parsed_document.get("mime_type")
        if mime_type and not mime_type.startswith("image/"):
            return {
                "handled": True,
                "response": templates.render_cheque_not_an_image(),
            }

        if not parsed_document.get("success"):
            return {
                "handled": True,
                "response": templates.render_cheque_invalid(parsed_document.get("error", "Unknown error"))
            }

        content = _normalize_content(parsed_document.get("content", {}))

        return self._validate_or_finalize(phone_number, content, trace_id)

    async def _handle_correct_cheque(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        parsed_document: dict | None,
        trace_id: str = "",
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
                "response": templates.render_cheque_not_an_image(),
                }

            if not parsed_document.get("success"):
                return {
                    "handled": True,
                    "response": templates.render_cheque_invalid(parsed_document.get("error", "Unknown error"))
                }

            new_content = _normalize_content(parsed_document.get("content", {}))

            for key, value in new_content.items():
                if _is_present(value):
                    content[key] = value

            return self._validate_or_finalize(phone_number, content, trace_id)

        if query and query.strip():

            if self._is_explanation_question(query):
                return {
                    "handled": True,
                    "response": templates.render_cheque_explanation(workflow.get("data", {}).get("validation_error")),
                }

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

                if self._is_acknowledgment(query):
                    pending_message = workflow.get("data", {}).get("validation_error")
                    return {
                        "handled": True,
                        "response": templates.render_cheque_pending_reminder(pending_message)
                    }

                return {
                    "handled": True,
                    "response": templates.render_cheque_correction_prompt()
                }

            return self._validate_or_finalize(phone_number, content, trace_id)

        return {
            "handled": True,
            "response": templates.render_cheque_reupload_prompt()
        }

    @staticmethod
    def _is_explanation_question(query: str) -> bool:
        text = query.strip().lower()
        return "?" in text or any(phrase in text for phrase in (
            "what is wrong", "what's wrong", "what is the error", "what's the error",
            "why did", "why is", "why are", "explain", "what happened",
        ))

    @staticmethod
    def _is_acknowledgment(query: str) -> bool:
        """
        A plain acknowledgment ("okay", "sure", "got it") is not a failed
        attempt at giving cheque data — it must not get the same "I couldn't
        read those details" response as genuinely garbled input, since that
        implies they tried and failed when they didn't try at all.
        """
        text = re.sub(r"[^a-z ]", "", query.strip().lower())
        return text in {
            "ok", "okay", "kk", "alright", "all right", "sure", "fine",
            "got it", "gotit", "understood", "noted", "roger", "cool",
            "right", "yes", "yeah", "yep", "thanks", "thank you",
            "no problem", "sounds good", "will do",
        }

    def _validate_or_finalize(
        self,
        phone_number: str,
        content: dict,
        trace_id: str = "",
    ) -> dict[str, Any]:

        missing = list(dict.fromkeys(
            _missing_fields(content) + _invalid_fields(content)
        ))

        if not _payee_matches_customer(content, phone_number):
            missing.append("payee")

        logger.info(
            f"[{trace_id}] Cheque validation | phone={phone_number[-4:]} | missing_or_invalid={missing}"
        )

        if missing:

            set_workflow_step(phone_number, STEP_CORRECT_CHEQUE)

            if "payee" in missing and not _is_present(content.get("payee")):
                message = templates.render_cheque_payee_missing()
            elif "payee" in missing:
                customer = get_customer_by_phone(phone_number)
                expected_name = (customer.get("full_name") if customer else "your registered name")
                message = templates.render_cheque_payee_mismatch(expected_name)
            else:
                message = templates.render_cheque_missing_fields(missing)

            update_workflow_data(phone_number, {
                "partial_content": content,
                "validation_error": message,
            })

            return {
                "handled": True,
                "response": message
            }

        return self._finalize_cheque_request(phone_number, content, trace_id)

    def _finalize_cheque_request(
        self,
        phone_number: str,
        content: dict,
        trace_id: str = "",
    ) -> dict[str, Any]:

        bank_name = content.get("bank_name") if _is_present(content.get("bank_name")) else None
        branch = content.get("branch") if _is_present(content.get("branch")) else None
        payee = content.get("payee")
        amount_figures = content.get("amount_in_figures")
        amount_words = content.get("amount_in_words") if _is_present(content.get("amount_in_words")) else None
        cheque_numbers = content.get("numbers") if _is_present(content.get("numbers")) else None
        signatory = content.get("signatory_title") if _is_present(content.get("signatory_title")) else None
        date_written = content.get("date_written") if _is_present(content.get("date_written")) else None
        drawer_name = content.get("drawer_name") if _is_present(content.get("drawer_name")) else None

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
                    date_written=date_written,
                    drawer_name=drawer_name,
                    status="PENDING",
                )
                request_id = candidate
                break
            except psycopg2.errors.UniqueViolation:
                logger.warning(f"[{trace_id}] Cheque request ID collision; retrying")
            except psycopg2.Error as error:
                logger.error(f"[{trace_id}] Cheque request persistence failed | phone={phone_number[-4:]} | error={error}")
                return {
                    "handled": True,
                    "response": templates.render_cheque_failed(),
                }

        if not request_id:
            raise RuntimeError("Unable to generate a unique cheque request ID")

        complete_workflow(phone_number)

        logger.info(
            f"[{trace_id}] Cheque request created | phone={phone_number[-4:]} | request_id={request_id}"
        )

        return {
            "handled": True,
            "response": templates.render_cheque_summary(
                request_id=request_id,
                payee=payee,
                amount_label=amount_figures,
                date_written=date_written,
                drawer_name=drawer_name,
                bank_name=bank_name,
            )
        }

import re
import uuid
from typing import Any

import psycopg2

from app.database import create_kyc_request
from app.logger import get_logger
from app.workflows.constants import STEP_CONFIRM_KYC, STEP_UPLOAD_KYC_FORM
from app.workflows.memory import complete_workflow, set_workflow_step, update_workflow_data
from app.workflows.nlu import interpret_confirmation
from app.services.llm_understanding import interpret_choice_llm, is_llm_fallback_enabled
from app.conversation.renderer import InteractiveButton, StructuredResponse
from app.conversation.responses import kyc as templates
from app.conversation.responses.common import with_nav_buttons

logger = get_logger(__name__)

# Only these five government-issued IDs are accepted — each maps to the
# regex its own id_number must match once separators (spaces/dashes) are
# stripped. Aadhaar/PAN/voter-ID/passport formats are fixed nationwide;
# driving licence format (state code + RTO code + year + serial) is the
# common 15-character scheme used by most states.
ID_VALIDATORS: dict[str, re.Pattern] = {
    "aadhaar": re.compile(r"^[2-9]\d{11}$"),
    "pan": re.compile(r"^[A-Z]{5}\d{4}[A-Z]$"),
    "passport": re.compile(r"^[A-Z]\d{7}$"),
    "voter_id": re.compile(r"^[A-Z]{3}\d{7}$"),
    "driving_license": re.compile(r"^[A-Z]{2}\d{13}$"),
}


def _normalize_id_number(value: str) -> str:
    return re.sub(r"[\s-]", "", str(value or "")).upper()


def _yes_no_prompt(body: str) -> StructuredResponse:
    """Tap-to-reply Yes/No confirmation — ids "yes"/"no" are the exact
    tokens interpret_confirmation()'s regex matches."""
    return StructuredResponse.buttons_of(
        body, [InteractiveButton(id="yes", title="Yes, submit"), InteractiveButton(id="no", title="No, cancel")]
    )


class KYCWorkflowHandler:
    """Collects KYC details from a photo of one government-issued ID —
    Aadhaar, PAN, Passport, Voter ID, or Driving Licence — extracts the
    name/date of birth/ID number/address, confirms with the customer, then
    submits. Deliberately document-only: unlike the old version, it never
    accepts typed "Field: value" corrections, since there's no reliable
    way to validate a typed ID number against the actual document without
    OCR — a customer who mistypes a digit would otherwise get KYC data on
    file that was never actually verified against their real ID."""

    async def handle(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None = None, trace_id: str = "") -> dict[str, Any]:
        logger.info(f"[{trace_id}] KYC workflow step | phone={phone_number[-4:]} | step={workflow['step']}")
        if workflow["step"] == STEP_UPLOAD_KYC_FORM:
            return self._collect(workflow, phone_number, query, parsed_document, trace_id)
        if workflow["step"] == STEP_CONFIRM_KYC:
            return self._confirm(workflow, phone_number, query, trace_id)
        return {"handled": True, "response": "The KYC update is in an invalid state. Please start again."}

    def _collect(self, workflow: dict, phone_number: str, query: str, parsed_document: dict | None, trace_id: str = "") -> dict[str, Any]:
        if parsed_document is None:
            # No document attached — a typed reply here can't be validated
            # against a real ID, so it's never accepted as KYC data (see
            # class docstring). Just re-show what's needed.
            return {"handled": True, "response": with_nav_buttons(templates.render_kyc_upload_prompt())}

        if not parsed_document.get("success"):
            return {"handled": True, "response": with_nav_buttons(templates.render_kyc_invalid())}

        content = parsed_document.get("content", {}) or {}
        id_type = str(content.get("id_type", "")).strip().lower()

        if id_type not in ID_VALIDATORS:
            logger.info(f"[{trace_id}] KYC document rejected | phone={phone_number[-4:]} | id_type={id_type!r}")
            return {"handled": True, "response": with_nav_buttons(templates.render_kyc_unsupported_document())}

        full_name = str(content.get("full_name", "")).strip()
        date_of_birth = str(content.get("date_of_birth", "")).strip()
        address = str(content.get("address", "")).strip()
        id_number = _normalize_id_number(content.get("id_number", ""))

        if not full_name or not date_of_birth or not ID_VALIDATORS[id_type].fullmatch(id_number):
            logger.info(
                f"[{trace_id}] KYC document unreadable/invalid | phone={phone_number[-4:]} | id_type={id_type} | "
                f"has_name={bool(full_name)} | has_dob={bool(date_of_birth)} | id_number_valid={bool(ID_VALIDATORS[id_type].fullmatch(id_number))}"
            )
            return {"handled": True, "response": with_nav_buttons(templates.render_kyc_could_not_read(id_type))}

        data = {
            "id_type": id_type,
            "id_number": id_number,
            "full_name": full_name,
            "date_of_birth": date_of_birth,
            "address": address,
        }
        update_workflow_data(phone_number, data)
        set_workflow_step(phone_number, STEP_CONFIRM_KYC)
        logger.info(f"[{trace_id}] KYC document verified, awaiting confirmation | phone={phone_number[-4:]} | id_type={id_type}")
        summary = templates.render_kyc_summary(id_type, full_name, date_of_birth, address)
        return {"handled": True, "response": _yes_no_prompt(summary)}

    def _confirm(self, workflow: dict, phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        llm_fallback = (
            (lambda text: interpret_choice_llm(text, ["yes", "no"], "Confirm or cancel the KYC details summary just shown."))
            if is_llm_fallback_enabled() else None
        )
        answer = interpret_confirmation(query, llm_fallback=llm_fallback)
        if answer == "no":
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] KYC update declined at confirmation | phone={phone_number[-4:]}")
            return {"handled": True, "response": templates.render_kyc_cancelled()}
        if answer != "yes":
            return {"handled": True, "response": _yes_no_prompt(templates.render_kyc_confirmation())}
        request_id = ""
        for _ in range(3):
            candidate = f"KYC-{uuid.uuid4().hex[:8].upper()}"
            try:
                create_kyc_request(candidate, phone_number, workflow.get("data", {}))
                request_id = candidate
                break
            except psycopg2.errors.UniqueViolation:
                logger.warning(f"[{trace_id}] KYC request ID collision; retrying")
        if not request_id:
            logger.error(f"[{trace_id}] KYC request creation failed | phone={phone_number[-4:]}")
            return {"handled": True, "response": with_nav_buttons(templates.render_kyc_failed())}
        complete_workflow(phone_number)
        logger.info(f"[{trace_id}] KYC request created | phone={phone_number[-4:]} | request_id={request_id}")
        return {"handled": True, "response": templates.render_kyc_success(request_id)}

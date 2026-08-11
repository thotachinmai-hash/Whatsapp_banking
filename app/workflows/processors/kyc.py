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

logger = get_logger(__name__)

REQUIRED_FIELDS = ("full_name", "date_of_birth", "address", "aadhaar_number", "pan_number")
LABELS = {"full_name": "Full name", "date_of_birth": "Date of birth", "address": "Address", "aadhaar_number": "Aadhaar number", "pan_number": "PAN number"}
ALIASES = {
    "name": "full_name", "fullname": "full_name", "dateofbirth": "date_of_birth", "dob": "date_of_birth",
    "address": "address", "aadhaar": "aadhaar_number", "aadhaarnumber": "aadhaar_number",
    "aadhar": "aadhaar_number", "pan": "pan_number", "pannumber": "pan_number",
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _extract(content: Any, result: dict | None = None) -> dict:
    result = result or {}
    if isinstance(content, dict):
        for raw_key, value in content.items():
            target = ALIASES.get(_key(str(raw_key)))
            if target and not isinstance(value, (dict, list)) and str(value).strip():
                result[target] = str(value).strip()
            _extract(value, result)
    elif isinstance(content, list):
        for value in content:
            _extract(value, result)
    return result


def _invalid(data: dict) -> list[str]:
    invalid = []
    if data.get("aadhaar_number") and not re.fullmatch(r"\d[\d -]{10,14}\d", data["aadhaar_number"]):
        invalid.append("aadhaar_number")
    if data.get("pan_number") and not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", data["pan_number"].replace(" ", "").upper()):
        invalid.append("pan_number")
    return invalid


def _yes_no_prompt(body: str) -> StructuredResponse:
    """Tap-to-reply Yes/No confirmation — ids "yes"/"no" are the exact
    tokens interpret_confirmation()'s regex matches."""
    return StructuredResponse.buttons_of(
        body, [InteractiveButton(id="yes", title="Yes, submit"), InteractiveButton(id="no", title="No, cancel")]
    )


def _is_acknowledgment(query: str) -> bool:
    """A plain acknowledgment isn't a failed data-entry attempt — don't
    respond as if they tried and garbled their KYC details."""
    text = re.sub(r"[^a-z ]", "", query.strip().lower())
    return text in {
        "ok", "okay", "kk", "alright", "all right", "sure", "fine",
        "got it", "gotit", "understood", "noted", "roger", "cool",
        "right", "yes", "yeah", "yep", "thanks", "thank you",
        "no problem", "sounds good", "will do",
    }


class KYCWorkflowHandler:
    async def handle(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None = None, trace_id: str = "") -> dict[str, Any]:
        logger.info(f"[{trace_id}] KYC workflow step | phone={phone_number[-4:]} | step={workflow['step']}")
        if workflow["step"] == STEP_UPLOAD_KYC_FORM:
            return self._collect(workflow, phone_number, query, parsed_document, trace_id)
        if workflow["step"] == STEP_CONFIRM_KYC:
            return self._confirm(workflow, phone_number, query, trace_id)
        return {"handled": True, "response": "The KYC update is in an invalid state. Please start again."}

    def _collect(self, workflow: dict, phone_number: str, query: str, parsed_document: dict | None, trace_id: str = "") -> dict[str, Any]:
        data = dict(workflow.get("data", {}))
        extracted = {}
        if parsed_document is not None:
            if not parsed_document.get("success"):
                return {"handled": True, "response": templates.render_kyc_invalid()}
            extracted = _extract(parsed_document.get("content", {}))
        elif ":" in query:
            for line in query.splitlines():
                if ":" in line:
                    raw, value = line.split(":", 1)
                    target = ALIASES.get(_key(raw.strip()))
                    if target and value.strip():
                        extracted[target] = value.strip()
        elif query.strip():
            missing_now = [field for field in REQUIRED_FIELDS if not str(data.get(field, "")).strip()]
            labels_now = ", ".join(LABELS[field] for field in missing_now) or "the remaining details"
            if _is_acknowledgment(query):
                return {"handled": True, "response": f"No problem! Whenever you're ready, please upload the KYC document or share {labels_now} as `Field: value`."}
            return {"handled": True, "response": f"I couldn't read that as KYC information. Please upload the KYC document or provide {labels_now} as `Field: value`."}
        data.update(extracted)
        missing = [field for field in REQUIRED_FIELDS if not str(data.get(field, "")).strip()]
        problems = list(dict.fromkeys(missing + _invalid(data)))
        if problems:
            update_workflow_data(phone_number, data)
            logger.info(f"[{trace_id}] KYC form missing/invalid fields | phone={phone_number[-4:]} | fields={problems}")
            return {"handled": True, "response": templates.render_kyc_missing_fields(problems)}
        data["aadhaar_number"] = re.sub(r"[ -]", "", data["aadhaar_number"])
        data["pan_number"] = data["pan_number"].replace(" ", "").upper()
        update_workflow_data(phone_number, data)
        set_workflow_step(phone_number, STEP_CONFIRM_KYC)
        logger.info(f"[{trace_id}] KYC form complete, awaiting confirmation | phone={phone_number[-4:]}")
        summary = templates.render_kyc_summary(data["full_name"], data["date_of_birth"], data["address"])
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
            return {"handled": True, "response": templates.render_kyc_failed()}
        complete_workflow(phone_number)
        logger.info(f"[{trace_id}] KYC request created | phone={phone_number[-4:]} | request_id={request_id}")
        return {"handled": True, "response": templates.render_kyc_success(request_id)}

import re
import uuid
from typing import Any

import psycopg2

from app.database import create_kyc_request
from app.workflows.constants import STEP_CONFIRM_KYC, STEP_UPLOAD_KYC_FORM
from app.workflows.memory import complete_workflow, set_workflow_step, update_workflow_data

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


class KYCWorkflowHandler:
    async def handle(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None = None) -> dict[str, Any]:
        if workflow["step"] == STEP_UPLOAD_KYC_FORM:
            return self._collect(workflow, phone_number, query, parsed_document)
        if workflow["step"] == STEP_CONFIRM_KYC:
            return self._confirm(workflow, phone_number, query)
        return {"handled": True, "response": "The KYC update is in an invalid state. Please start again."}

    def _collect(self, workflow: dict, phone_number: str, query: str, parsed_document: dict | None) -> dict[str, Any]:
        data = dict(workflow.get("data", {}))
        extracted = {}
        if parsed_document is not None:
            if not parsed_document.get("success"):
                return {"handled": True, "response": "I could not read that KYC document. Please upload a clearer image or PDF."}
            extracted = _extract(parsed_document.get("content", {}))
        elif ":" in query:
            for line in query.splitlines():
                if ":" in line:
                    raw, value = line.split(":", 1)
                    target = ALIASES.get(_key(raw.strip()))
                    if target and value.strip():
                        extracted[target] = value.strip()
        elif query.strip():
            return {"handled": True, "response": "Please upload the KYC document or provide corrections as `Field: value`."}
        data.update(extracted)
        missing = [field for field in REQUIRED_FIELDS if not str(data.get(field, "")).strip()]
        problems = list(dict.fromkeys(missing + _invalid(data)))
        if problems:
            update_workflow_data(phone_number, data)
            labels = ", ".join(LABELS[field] for field in problems)
            return {"handled": True, "response": f"I still need these KYC details: {labels}. Please upload a clearer document or reply with `Field: value`."}
        data["aadhaar_number"] = re.sub(r"[ -]", "", data["aadhaar_number"])
        data["pan_number"] = data["pan_number"].replace(" ", "").upper()
        update_workflow_data(phone_number, data)
        set_workflow_step(phone_number, STEP_CONFIRM_KYC)
        return {"handled": True, "response": ("Please confirm your KYC details:\n\n"
            f"Name: {data['full_name']}\nDate of birth: {data['date_of_birth']}\nAddress: {data['address']}\n"
            f"Aadhaar: {data['aadhaar_number']}\nPAN: {data['pan_number']}\n\n"
            "Reply YES to submit the KYC update or NO to cancel.")}

    def _confirm(self, workflow: dict, phone_number: str, query: str) -> dict[str, Any]:
        answer = query.strip().lower()
        if answer in {"no", "n", "cancel"}:
            complete_workflow(phone_number)
            return {"handled": True, "response": "Your KYC update was cancelled."}
        if answer not in {"yes", "y", "confirm"}:
            return {"handled": True, "response": "Reply YES to submit the KYC update or NO to cancel."}
        request_id = ""
        for _ in range(3):
            candidate = f"KYC-{uuid.uuid4().hex[:8].upper()}"
            try:
                create_kyc_request(candidate, phone_number, workflow.get("data", {}))
                request_id = candidate
                break
            except psycopg2.errors.UniqueViolation:
                continue
        if not request_id:
            return {"handled": True, "response": "I could not submit the KYC update. Please try again."}
        complete_workflow(phone_number)
        return {"handled": True, "response": f"KYC update submitted successfully.\n\nRequest ID: {request_id}\nStatus: PENDING\n\nOur team will verify your documents and contact you if anything else is required."}

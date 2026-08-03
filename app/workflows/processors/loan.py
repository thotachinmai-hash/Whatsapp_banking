import re
import uuid
from typing import Any

import psycopg2

from app.database import create_loan_request
from app.logger import get_logger
from app.workflows.constants import (
    STEP_CONFIRM_LOAN,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_LOAN_FORM,
)
from app.workflows.memory import complete_workflow, set_workflow_step, update_workflow_data

logger = get_logger(__name__)

LOAN_TYPES = {"1": "personal", "2": "home", "3": "vehicle", "4": "education"}
LOAN_LABELS = {"personal": "Personal Loan", "home": "Home Loan", "vehicle": "Vehicle Loan", "education": "Education Loan"}
REQUIRED_FIELDS = ("applicant_name", "monthly_income", "employment_type", "requested_amount", "tenure_months", "purpose")
FIELD_LABELS = {
    "applicant_name": "Applicant name", "monthly_income": "Monthly income",
    "employment_type": "Employment type", "requested_amount": "Requested amount",
    "tenure_months": "Tenure in months", "purpose": "Loan purpose",
}
ALIASES = {
    "name": "applicant_name", "applicant": "applicant_name", "applicantname": "applicant_name",
    "monthlyincome": "monthly_income", "income": "monthly_income", "salary": "monthly_income",
    "monthlynetsalary": "monthly_income", "netsalary": "monthly_income",
    "employment": "employment_type", "employmenttype": "employment_type",
    "amount": "requested_amount", "requestedamount": "requested_amount", "loanamount": "requested_amount",
    "tenure": "tenure_months", "tenuremonths": "tenure_months", "loantenure": "tenure_months",
    "loanpurpose": "purpose",
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _extract(content: Any, result: dict | None = None) -> dict:
    result = {} if result is None else result
    if isinstance(content, dict):
        for raw_key, value in content.items():
            target = ALIASES.get(_key(str(raw_key)))
            if target and not isinstance(value, (dict, list)) and str(value).strip():
                result[target] = _normalize_value(target, str(value).strip())
            _extract(value, result)
    elif isinstance(content, list):
        for value in content:
            _extract(value, result)
    return result


def _missing(data: dict) -> list[str]:
    return [field for field in REQUIRED_FIELDS if not str(data.get(field, "")).strip()]


def _normalize_value(field: str, value: str) -> str:
    value = value.strip()
    if field == "tenure_months":
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(years?|yrs?|months?|mos?)?", value.lower())
        if match:
            number = float(match.group(1))
            unit = match.group(2) or "months"
            if unit.startswith(("year", "yr")):
                number *= 12
            return str(int(number)) if number.is_integer() else str(number)
    return value


def _invalid(data: dict) -> list[str]:
    invalid = []
    for field in ("monthly_income", "requested_amount", "tenure_months"):
        if field in data:
            try:
                numeric = re.sub(r"[^0-9.\-]", "", str(data[field]))
                if not numeric or float(numeric) <= 0:
                    invalid.append(field)
            except ValueError:
                invalid.append(field)
    return invalid


class LoanWorkflowHandler:
    async def handle(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None = None) -> dict[str, Any]:
        step = workflow["step"]
        if step == STEP_SELECT_LOAN_TYPE:
            return self._select_type(phone_number, query)
        if step == STEP_UPLOAD_LOAN_FORM:
            return self._collect_form(workflow, phone_number, query, parsed_document)
        if step == STEP_CONFIRM_LOAN:
            return self._confirm(workflow, phone_number, query)
        return {"handled": True, "response": "The loan application is in an invalid state. Please start again."}

    def _select_type(self, phone_number: str, query: str) -> dict[str, Any]:
        value = query.strip().lower()
        loan_type = LOAN_TYPES.get(value)
        if not loan_type:
            for candidate in LOAN_TYPES.values():
                if candidate in value:
                    loan_type = candidate
                    break
        if not loan_type:
            return {"handled": True, "response": "Please choose 1 Personal, 2 Home, 3 Vehicle, or 4 Education Loan."}
        update_workflow_data(phone_number, {"loan_type": loan_type})
        set_workflow_step(phone_number, STEP_UPLOAD_LOAN_FORM)
        return {"handled": True, "response": f"{LOAN_LABELS[loan_type]} selected. Please complete and upload the loan form.\n\nRequired: applicant name, monthly income, employment type, requested amount, tenure in months, and loan purpose."}

    def _collect_form(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None) -> dict[str, Any]:
        data = dict(workflow.get("data", {}))
        extracted = {}
        if parsed_document is not None:
            if not parsed_document.get("success"):
                return {"handled": True, "response": "I could not read that form. Please upload a clearer loan form."}
            extracted = _extract(parsed_document.get("content", {}))
        elif ":" in query or "=" in query:
            for line in query.splitlines():
                match = re.match(r"^\s*([^:=]+?)\s*[:=]\s*(.+?)\s*$", line)
                if match:
                    raw, value = match.groups()
                    target = ALIASES.get(_key(raw.strip()))
                    if target and value.strip():
                        extracted[target] = _normalize_value(target, value)
        elif query.strip():
            return {"handled": True, "response": "Please upload the completed loan form, or provide corrections as `Field: value`."}
        data.update(extracted)
        problems = list(dict.fromkeys(_missing(data) + _invalid(data)))
        if problems:
            update_workflow_data(phone_number, data)
            labels = ", ".join(FIELD_LABELS[field] for field in problems)
            return {"handled": True, "response": f"I still need these loan details: {labels}. Please upload the completed form or reply with `Field: value`."}
        update_workflow_data(phone_number, data)
        set_workflow_step(phone_number, STEP_CONFIRM_LOAN)
        return {"handled": True, "response": self._confirmation(data)}

    def _confirmation(self, data: dict) -> str:
        return ("Please confirm your loan application:\n\n"
                f"Loan type: {LOAN_LABELS.get(data.get('loan_type'), data.get('loan_type'))}\n"
                f"Applicant: {data['applicant_name']}\nMonthly income: {data['monthly_income']}\n"
                f"Requested amount: {data['requested_amount']}\nTenure: {data['tenure_months']} months\n"
                f"Employment: {data['employment_type']}\nPurpose: {data['purpose']}\n\n"
                "Reply YES to create the request or NO to cancel.")

    def _confirm(self, workflow: dict[str, Any], phone_number: str, query: str) -> dict[str, Any]:
        answer = query.strip().lower()
        if answer in {"no", "n", "cancel"}:
            complete_workflow(phone_number)
            return {"handled": True, "response": "Your loan application was cancelled."}
        if answer not in {"yes", "y", "confirm"}:
            return {"handled": True, "response": "Reply YES to create the loan request or NO to cancel."}
        data = workflow.get("data", {})
        request_id = ""
        for _ in range(3):
            candidate = f"LOAN-{uuid.uuid4().hex[:8].upper()}"
            try:
                create_loan_request(candidate, phone_number, data["loan_type"], data)
                request_id = candidate
                break
            except psycopg2.errors.UniqueViolation:
                continue
        if not request_id:
            return {"handled": True, "response": "I could not create the loan request. Please try again."}
        complete_workflow(phone_number)
        return {"handled": True, "response": f"Loan request created successfully.\n\nRequest ID: {request_id}\nStatus: PENDING\n\nOur team will review your application and contact you with the next steps."}

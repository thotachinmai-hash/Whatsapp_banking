import re
import uuid
from typing import Any

import psycopg2

from app.database import create_loan_request, get_accounts_by_phone
from app.logger import get_logger
from app.workflows.constants import (
    STEP_CONFIRM_LOAN,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_LOAN_FORM,
)
from app.workflows.memory import clear_workflow_data, complete_workflow, set_workflow_step, update_workflow_data
from app.workflows.nlu import interpret_confirmation, interpret_menu_choice
from app.services.llm_understanding import interpret_choice_llm, is_llm_fallback_enabled
from app.conversation.renderer import InteractiveButton, InteractiveListRow, InteractiveListSection, StructuredResponse
from app.conversation.responses import loan as templates

logger = get_logger(__name__)

LOAN_TYPES = {"1": "personal", "2": "home", "3": "vehicle", "4": "education"}
LOAN_LABELS = {"personal": "Personal Loan", "home": "Home Loan", "vehicle": "Vehicle Loan", "education": "Education Loan"}
# Natural synonyms beyond the canonical words themselves — so "I'd like a
# mortgage" or "loan for a car" resolve without needing a number or the
# exact loan-type word. Mirrors the aliases already used for the same
# purpose in app/agent/tools.py's get_loan_product_info tool.
LOAN_TYPE_SYNONYMS = {
    "mortgage": "home", "housing": "home", "house": "home",
    "car": "vehicle", "auto": "vehicle", "bike": "vehicle", "motorcycle": "vehicle",
    "student": "education", "college": "education", "university": "education", "study": "education",
}

# Asked one at a time, in this order — account number first, so the loan is
# tied to a real account on the customer's profile from the start.
FIELD_ORDER = (
    "account_number", "applicant_name", "monthly_income", "employment_type",
    "requested_amount", "tenure_months", "purpose",
)
FIELD_LABELS = {
    "account_number": "Account number", "applicant_name": "Applicant name",
    "monthly_income": "Monthly income", "employment_type": "Employment type",
    "requested_amount": "Requested amount", "tenure_months": "Tenure in months",
    "purpose": "Loan purpose",
}
FIELD_PROMPTS = {
    "account_number": "What is the account number this loan should be linked to?",
    "applicant_name": "What is the applicant's full name?",
    "monthly_income": "What is your monthly income?",
    "employment_type": "What is your employment type? (e.g. Salaried, Self-employed, Business)",
    "requested_amount": "How much would you like to borrow?",
    "tenure_months": "What repayment tenure would you like, in months? (e.g. 24)",
    "purpose": "What is the purpose of this loan?",
}
ALIASES = {
    "accountnumber": "account_number", "account": "account_number", "acno": "account_number",
    "name": "applicant_name", "applicant": "applicant_name", "applicantname": "applicant_name",
    "monthlyincome": "monthly_income", "income": "monthly_income", "salary": "monthly_income",
    "monthlynetsalary": "monthly_income", "netsalary": "monthly_income",
    "employment": "employment_type", "employmenttype": "employment_type",
    "amount": "requested_amount", "requestedamount": "requested_amount", "loanamount": "requested_amount",
    "tenure": "tenure_months", "tenuremonths": "tenure_months", "loantenure": "tenure_months",
    "purpose": "purpose", "loanpurpose": "purpose", "reason": "purpose",
}
ACKNOWLEDGMENTS = {
    "ok", "okay", "kk", "alright", "all right", "sure", "fine",
    "got it", "gotit", "understood", "noted", "roger", "cool",
    "right", "yes", "yeah", "yep", "thanks", "thank you",
    "no problem", "sounds good", "will do",
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _llm_fallback(options: list[str], context: str):
    if not is_llm_fallback_enabled():
        return None
    return lambda text: interpret_choice_llm(text, options, context)


def loan_type_list_prompt(intro: str) -> StructuredResponse:
    """Tap-to-reply loan-type list — row ids "1".."4" match LOAN_TYPES'
    exact dict lookup (see detect_loan_type_from_text), so a typed digit
    still works exactly as before if the list send falls back to text.
    Shared by this processor's own re-prompt and by
    app/workflows/manager.py::start_requested's loan-start branch, so the
    two can no longer drift out of sync the way their separate hardcoded
    copies used to."""
    rows = [
        InteractiveListRow(id=digit, title=LOAN_LABELS[loan_type])
        for digit, loan_type in LOAN_TYPES.items()
    ]
    return StructuredResponse.list_of(intro, "Choose loan type", [InteractiveListSection(title="Loan types", rows=rows)])


def _yes_no_prompt(body: str) -> StructuredResponse:
    """Tap-to-reply Yes/No confirmation — ids "yes"/"no" are the exact
    tokens interpret_confirmation()'s regex matches (no digit fast-path
    exists for this prompt, unlike transfer's numbered confirm)."""
    return StructuredResponse.buttons_of(
        body, [InteractiveButton(id="yes", title="Yes, submit"), InteractiveButton(id="no", title="No, cancel")]
    )


def detect_loan_type_from_text(text: str) -> str | None:
    """Pull a loan type out of free text ("I'd like a home loan of 50000")
    so a customer who already states it when starting the workflow skips
    the separate "which loan type?" step — same digit/name/synonym
    matching _select_type uses, factored out so the workflow starter
    (app/workflows/manager.py::WorkflowManager.start_requested) can reuse
    it instead of always asking from scratch."""
    value = (text or "").strip().lower()
    loan_type = LOAN_TYPES.get(value)
    if not loan_type:
        for candidate in LOAN_TYPES.values():
            if candidate in value:
                loan_type = candidate
                break
    if not loan_type:
        for synonym, candidate in LOAN_TYPE_SYNONYMS.items():
            if synonym in value:
                loan_type = candidate
                break
    return loan_type


def _is_acknowledgment(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.strip().lower()) in ACKNOWLEDGMENTS


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


def _next_missing_field(data: dict) -> str | None:
    for field in FIELD_ORDER:
        if not str(data.get(field, "")).strip():
            return field
    return None


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
    if field == "account_number":
        return re.sub(r"\s", "", value).upper()
    return value


class LoanWorkflowHandler:
    """
    Collects the loan application one field at a time (account number,
    applicant name, monthly income, employment type, requested amount,
    tenure, purpose), verifies once with the customer, then submits.
    Uploading a completed form still works as a shortcut — it fills
    whatever fields are readable and the wizard picks up wherever data is
    still missing.
    """

    async def handle(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None = None, trace_id: str = "") -> dict[str, Any]:
        step = workflow["step"]
        logger.info(f"[{trace_id}] Loan workflow step | phone={phone_number[-4:]} | step={step}")
        if step == STEP_SELECT_LOAN_TYPE:
            return self._select_type(workflow, phone_number, query, trace_id)
        if step == STEP_UPLOAD_LOAN_FORM:
            return self._collect_field(workflow, phone_number, query, parsed_document, trace_id)
        if step == STEP_CONFIRM_LOAN:
            return self._confirm(workflow, phone_number, query, trace_id)
        return {"handled": True, "response": "The loan application is in an invalid state. Please start again."}

    def _select_type(self, workflow: dict[str, Any], phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        loan_type = detect_loan_type_from_text(query)
        if not loan_type:
            menu_options = list(LOAN_TYPES.keys()) + list(LOAN_TYPES.values())
            llm_choice = interpret_menu_choice(
                query, menu_options,
                llm_fallback=_llm_fallback(menu_options, "Choose a loan type: 1 Personal, 2 Home, 3 Vehicle, 4 Education."),
            )
            loan_type = LOAN_TYPES.get(llm_choice, llm_choice) if llm_choice else None
        if not loan_type:
            return {"handled": True, "response": loan_type_list_prompt(
                "\U0001F4DD Sorry, I didn't catch that — which of these?"
            )}

        pending_content = workflow.get("data", {}).get("pending_document_content")
        update_workflow_data(phone_number, {"loan_type": loan_type})
        set_workflow_step(phone_number, STEP_UPLOAD_LOAN_FORM)
        logger.info(f"[{trace_id}] Loan type selected | phone={phone_number[-4:]} | loan_type={loan_type}")

        if pending_content:
            # A loan form was already uploaded before the loan type was
            # known (see message_handler.py's bare-upload auto-detect) —
            # apply it now instead of asking the customer to upload it
            # again.
            clear_workflow_data(phone_number, "pending_document_content")
            data = dict(workflow.get("data", {}))
            data.pop("pending_document_content", None)
            data["loan_type"] = loan_type
            extracted = _extract(pending_content)
            data.update(extracted)
            update_workflow_data(phone_number, extracted)
            logger.info(
                f"[{trace_id}] Applied pending loan form after type selection | "
                f"phone={phone_number[-4:]} | fields={list(extracted)}"
            )
            return self._ask_next_or_confirm(phone_number, data, trace_id)

        return {
            "handled": True,
            "response": templates.render_loan_type_selected(loan_type),
        }

    def _collect_field(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None, trace_id: str = "") -> dict[str, Any]:
        data = dict(workflow.get("data", {}))

        if parsed_document is not None:
            if not parsed_document.get("success"):
                return {"handled": True, "response": templates.render_loan_document_unreadable()}
            extracted = _extract(parsed_document.get("content", {}))
            data.update(extracted)
            update_workflow_data(phone_number, data)
            logger.info(f"[{trace_id}] Loan form document processed | phone={phone_number[-4:]} | fields={list(extracted)}")
            return self._ask_next_or_confirm(phone_number, data, trace_id)

        current_field = _next_missing_field(data)
        if current_field is None:
            set_workflow_step(phone_number, STEP_CONFIRM_LOAN)
            return {"handled": True, "response": self._confirmation(data)}

        if self._is_question(query):
            return {"handled": True, "response": self._answer_question(data, query, current_field)}

        text = query.strip()
        if not text:
            return {"handled": True, "response": templates.render_loan_field_prompt(current_field)}

        if _is_acknowledgment(text):
            return {"handled": True, "response": "No problem! " + templates.render_loan_field_prompt(current_field)}

        # Still accept "Field: value" for anyone who prefers to paste ahead,
        # but it no longer applies to just the current field.
        if ":" in text or "=" in text:
            matched_any = False
            for line in text.splitlines():
                match = re.match(r"^\s*([^:=]+?)\s*[:=]\s*(.+?)\s*$", line)
                if match:
                    raw, value = match.groups()
                    target = ALIASES.get(_key(raw.strip()))
                    if target and value.strip():
                        data[target] = _normalize_value(target, value)
                        matched_any = True
            if matched_any:
                update_workflow_data(phone_number, data)
                return self._ask_next_or_confirm(phone_number, data, trace_id)

        value, error = self._validate_field(current_field, text, phone_number)
        if error:
            return {"handled": True, "response": templates.render_loan_field_invalid(current_field, error)}

        data[current_field] = value
        update_workflow_data(phone_number, data)
        logger.info(f"[{trace_id}] Loan field collected | phone={phone_number[-4:]} | field={current_field}")
        return self._ask_next_or_confirm(phone_number, data, trace_id, just_completed_field=current_field)

    def _ask_next_or_confirm(self, phone_number: str, data: dict, trace_id: str = "", just_completed_field: str | None = None) -> dict[str, Any]:
        next_field = _next_missing_field(data)
        if next_field:
            return {"handled": True, "response": templates.render_loan_field_prompt(next_field, just_completed_field)}
        set_workflow_step(phone_number, STEP_CONFIRM_LOAN)
        logger.info(f"[{trace_id}] Loan form complete, awaiting confirmation | phone={phone_number[-4:]}")
        return {"handled": True, "response": self._confirmation(data)}

    @staticmethod
    def _validate_field(field: str, text: str, phone_number: str) -> tuple[str, str | None]:
        """Returns (normalized_value, error_message_or_None)."""
        if field == "account_number":
            candidate = _normalize_value(field, text)
            accounts = get_accounts_by_phone(phone_number)
            match = next((a for a in accounts if a["account_number"] == candidate), None)
            if not match:
                available = ", ".join(a["account_number"] for a in accounts) or "none on file"
                return "", f"I couldn't find that account on your profile (your accounts: {available})."
            return candidate, None
        if field == "applicant_name":
            if len(text) < 2 or not re.match(r"^[A-Za-z .'-]+$", text):
                return "", "That doesn't look like a valid name."
            return text.strip(), None
        if field in {"monthly_income", "requested_amount", "tenure_months"}:
            normalized = _normalize_value(field, text)
            numeric = re.sub(r"[^0-9.\-]", "", normalized)
            try:
                if not numeric or float(numeric) <= 0:
                    return "", "That doesn't look like a valid number."
            except ValueError:
                return "", "That doesn't look like a valid number."
            return normalized, None
        return text.strip(), None

    @staticmethod
    def _is_question(query: str) -> bool:
        text = query.strip().lower()
        return "?" in text or any(word in text.split() for word in ("what", "why", "how", "which", "meaning"))

    def _answer_question(self, data: dict[str, Any], query: str, current_field: str) -> str:
        text = query.lower()
        definitions = {
            "account_number": "the account this loan will be linked to and disbursed into",
            "applicant_name": "the full name of the person applying",
            "monthly_income": "your regular monthly income",
            "employment_type": "your work type, such as salaried, self-employed, or business",
            "requested_amount": "the amount you want to borrow",
            "tenure_months": "how many months you want to take to repay the loan",
            "purpose": "what you need the loan for",
        }
        aliases = {
            "account": "account_number", "name": "applicant_name", "income": "monthly_income",
            "salary": "monthly_income", "employment": "employment_type", "work": "employment_type",
            "amount": "requested_amount", "borrow": "requested_amount", "tenure": "tenure_months",
            "repay": "tenure_months", "purpose": "purpose",
        }
        for word, field in aliases.items():
            if word in text:
                current = data.get(field)
                return templates.render_loan_field_explanation(field, definitions[field], current)
        return f"ℹ️ {FIELD_LABELS[current_field]} means {definitions[current_field]}. {FIELD_PROMPTS[current_field]}"

    def _confirmation(self, data: dict) -> StructuredResponse:
        summary = templates.render_loan_summary(
            loan_type=data.get("loan_type"),
            account_number=data["account_number"],
            applicant_name=data["applicant_name"],
            monthly_income=data["monthly_income"],
            requested_amount=data["requested_amount"],
            tenure_months=data["tenure_months"],
            employment_type=data["employment_type"],
            purpose=data["purpose"],
        )
        return _yes_no_prompt(summary)

    def _confirm(self, workflow: dict[str, Any], phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        answer = interpret_confirmation(
            query, llm_fallback=_llm_fallback(["yes", "no"], "Confirm or cancel the loan application summary just shown.")
        )
        if answer == "no":
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] Loan application declined at confirmation | phone={phone_number[-4:]}")
            return {"handled": True, "response": templates.render_loan_cancelled()}
        if answer != "yes":
            # Deliberately distinct wording from render_loan_confirmation() —
            # matches pre-Phase-4 behavior (this reprompt already used
            # slightly different phrasing than the initial summary).
            return {"handled": True, "response": _yes_no_prompt("Reply YES to submit the loan request or NO to cancel.")}
        data = workflow.get("data", {})
        request_id = ""
        for _ in range(3):
            candidate = f"LOAN-{uuid.uuid4().hex[:8].upper()}"
            try:
                create_loan_request(candidate, phone_number, data["loan_type"], data)
                request_id = candidate
                break
            except psycopg2.errors.UniqueViolation:
                logger.warning(f"[{trace_id}] Loan request ID collision; retrying")
        if not request_id:
            logger.error(f"[{trace_id}] Loan request creation failed | phone={phone_number[-4:]}")
            return {"handled": True, "response": templates.render_loan_failed()}
        complete_workflow(phone_number)
        logger.info(f"[{trace_id}] Loan request created | phone={phone_number[-4:]} | request_id={request_id}")
        return {
            "handled": True,
            "response": templates.render_loan_success(request_id),
        }

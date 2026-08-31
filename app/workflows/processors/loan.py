import re
import uuid
from typing import Any

import psycopg2

from app.database import (
    create_loan_request,
    get_accounts_by_phone,
    get_customer_by_phone,
    get_frequently_used_account,
)
from app.logger import get_logger
from app.workflows.constants import (
    STEP_CONFIRM_LOAN,
    STEP_CONFIRM_LOAN_ACCOUNT,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_LOAN_FORM,
)
from app.workflows.memory import clear_workflow_data, complete_workflow, set_workflow_step, update_workflow_data
from app.workflows.nlu import interpret_confirmation, interpret_menu_choice
from app.services.llm_understanding import interpret_choice_llm, is_llm_fallback_enabled
from app.conversation.renderer import InteractiveButton, InteractiveListRow, InteractiveListSection, StructuredResponse
from app.conversation.responses.common import format_currency
from app.conversation.responses import loan as templates
from app.conversation.responses.common import with_nav_buttons
from app.services.receipts import build_receipt_response

logger = get_logger(__name__)

# Namespaced ids (not bare digits, and not plain words) — a bare "1".."4"
# here would collide with the main menu's own digit ids ("1"=transfer,
# "2"=balance, "3"=transactions, "4"=cheque, etc. — see
# WorkflowManager.start_requested's menu_actions). If a tap on this list
# ever arrives with no active loan workflow to interpret it (observed
# live: a stray/late list_reply after the workflow was already gone), a
# bare digit falls through to that unrelated main-menu dispatch and
# silently starts the wrong thing — confirmed live: tapping "Education
# Loan" (old id "4") started a cheque deposit instead.
#
# A plain-word id like "loan_education" or "home" isn't safe either:
# start_requested()'s free-text keyword matching is substring-based
# (`"loan" in normalized`), so "loan_education" alone re-triggers the
# loan-start keyword path — confirmed live, it silently created and
# advanced a fresh loan workflow. "home" also exactly matches the
# main-menu-navigation word set (`_MENU_WORDS`) elsewhere. The "lt_"
# prefix isn't a substring any keyword/navigation matcher looks for, so a
# stray reply with no active workflow now safely falls through to
# "out of scope" instead of silently starting an unrelated action.
LOAN_TYPES = {"lt_personal": "personal", "lt_home": "home", "lt_vehicle": "vehicle", "lt_education": "education"}
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
# "purpose" and "applicant_name" are deliberately NOT here — both are
# derived automatically instead of asked (see _select_type): applicant_name
# from the customer's own profile, purpose from the loan type itself
# (a "Home Loan" doesn't need a separate "what's this for?" question).
FIELD_ORDER = (
    "account_number", "monthly_income", "employment_type",
    "requested_amount", "tenure_months",
)
FIELD_LABELS = {
    "account_number": "Account number", "applicant_name": "Applicant name",
    "monthly_income": "Monthly income", "employment_type": "Employment type",
    "requested_amount": "Requested amount", "tenure_months": "Tenure in months",
    "purpose": "Loan purpose",
}
FIELD_PROMPTS = {
    "account_number": "Which account should this loan be linked to?",
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
}
ACKNOWLEDGMENTS = {
    "ok", "okay", "kk", "alright", "all right", "sure", "fine",
    "got it", "gotit", "understood", "noted", "roger", "cool",
    "right", "yes", "yeah", "yep", "thanks", "thank you",
    "no problem", "sounds good", "will do",
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _resolve_account_selection(phone_number: str, text: str) -> str | None:
    """Map a tapped account-menu row id, or a typed full account number,
    back to a real account on the customer's profile — same numbered-menu
    resolution as the transfer workflow's source-account step (see
    processors/transfer.py::_source_account)."""
    accounts = get_accounts_by_phone(phone_number)
    choice = text.strip()
    if choice.isdigit() and 1 <= int(choice) <= len(accounts):
        return accounts[int(choice) - 1]["account_number"]
    candidate = re.sub(r"\s", "", choice).upper()
    match = next((a for a in accounts if a["account_number"] == candidate), None)
    return match["account_number"] if match else None


def _llm_fallback(options: list[str], context: str):
    if not is_llm_fallback_enabled():
        return None
    return lambda text: interpret_choice_llm(text, options, context)


def loan_type_list_prompt(intro: str) -> StructuredResponse:
    """Tap-to-reply loan-type list — row ids ("lt_personal", "lt_home",
    "lt_vehicle", "lt_education") match LOAN_TYPES' exact dict lookup
    (see detect_loan_type_from_text), so a typed row id still works
    exactly as before if the list send falls back to text. Deliberately
    NOT bare digits "1".."4" — unlike the transfer workflow's numbered
    account/beneficiary lists, this menu can also be reached via
    app/workflows/manager.py::start_requested's loan-start branch before
    any workflow is active, where a bare "1"-"4" would collide with the
    main menu's own digit map (see app/conversation/manager.py's
    OUT_OF_SCOPE/CLARIFICATION_REQUIRED digit fallback). Shared by this
    processor's own re-prompt and by start_requested's loan-start branch,
    so the two can no longer drift out of sync the way their separate
    hardcoded copies used to."""
    rows = [
        InteractiveListRow(id=row_id, title=LOAN_LABELS[loan_type])
        for row_id, loan_type in LOAN_TYPES.items()
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


# Closed vocabulary for employment type — same low-risk approach as
# LOAN_TYPE_SYNONYMS above (a fixed word list, not a guess), so "I'm
# salaried" or "self employed" resolve without a separate question.
_EMPLOYMENT_TYPE_WORDS = {
    "self-employed": "Self-employed", "self employed": "Self-employed", "selfemployed": "Self-employed",
    "self": "Self-employed",
    "salaried": "Salaried",
    "business owner": "Business", "businessman": "Business", "businesswoman": "Business", "business": "Business",
    "unemployed": "Unemployed", "retired": "Retired", "student": "Student",
}
_TENURE_HINT_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*(years?|yrs?|months?|mos?)\b", re.I)
# Indian-English amount shorthand ("5 lakh", "50k", "2 crore") — captured
# as an optional group alongside the digits so a stated amount/income
# isn't silently truncated to its bare leading digits (confirmed live:
# "a personal loan of ₹5 lakh" was recorded as a requested_amount of "5").
_AMOUNT_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000, "lac": 100_000, "lakh": 100_000, "l": 100_000,
    "crore": 10_000_000, "cr": 10_000_000,
}
# "l"/"cr" as bare-letter shorthand for lakh/crore (e.g. "40l", "2cr") --
# placed after their longer counterparts so "lakh"/"crore" still match in
# full rather than backtracking to the single-letter alternative first
# (confirmed live: a customer answering "How much would you like to
# borrow?" with "40l" had it stored as the literal string "40l" instead
# of 4000000, since fullmatch had no alternative to consume the "l").
_AMOUNT_SUFFIX_RE = r"\s*(k|thousand|lac|lakhs?|crores?|l|cr)?\s*(?:rupees?)?"
_INCOME_HINT_RE = re.compile(
    rf"\b(?:income|salary|earn\w*|take[\s-]?home)\D{{0,12}}?([0-9][0-9,]*(?:\.[0-9]{{1,2}})?){_AMOUNT_SUFFIX_RE}", re.I
)
_AMOUNT_HINT_RE = re.compile(
    rf"\b(?:loan\s*(?:of|amount)?|borrow\w*|requested\s*amount)\D{{0,12}}?([0-9][0-9,]*(?:\.[0-9]{{1,2}})?){_AMOUNT_SUFFIX_RE}", re.I
)


def _apply_amount_suffix(number_str: str, suffix: str | None) -> str:
    """Multiply a captured digit string by its Indian-English shorthand
    suffix ("lakh" -> ×100,000, "crore" -> ×10,000,000, "k"/"thousand" ->
    ×1,000), or return it unchanged if there's no suffix."""
    value = float(number_str.replace(",", ""))
    if suffix:
        value *= _AMOUNT_MULTIPLIERS.get(suffix.strip().lower().rstrip("s"), 1)
    return str(int(value)) if value.is_integer() else str(value)


def extract_loan_fields_from_text(text: str) -> dict[str, str]:
    """Pull whatever loan fields a free-form sentence already states —
    e.g. "personal loan of 5,00,000 for 24 months, I'm salaried, income
    50000" — so the wizard only asks for what's still missing instead of
    one field at a time for data already given.

    Deliberately narrow: only tenure, monthly income, requested amount,
    and employment type are extracted here, each behind its own anchor
    keyword (or, for tenure, a mandatory units suffix) so three
    plain-number fields can't get swapped for one another by an
    unanchored guess. Applicant name, purpose, and account number are
    NOT extracted from free text — a name/purpose has no reliable anchor
    to key off, and an account number must still be validated against
    the customer's real accounts (see _validate_field), so those stay
    asked explicitly.
    """
    result: dict[str, str] = {}
    lowered = text.lower()

    tenure_match = _TENURE_HINT_RE.search(text)
    if tenure_match:
        result["tenure_months"] = _normalize_value(
            "tenure_months", f"{tenure_match.group(1)} {tenure_match.group(2)}"
        )

    income_match = _INCOME_HINT_RE.search(text)
    if income_match:
        result["monthly_income"] = _apply_amount_suffix(income_match.group(1), income_match.group(2))

    amount_match = _AMOUNT_HINT_RE.search(text)
    if amount_match:
        amount_value = _apply_amount_suffix(amount_match.group(1), amount_match.group(2))
        if amount_value != result.get("monthly_income"):
            result["requested_amount"] = amount_value

    for phrase, label in sorted(_EMPLOYMENT_TYPE_WORDS.items(), key=lambda kv: -len(kv[0])):
        if phrase in lowered:
            result["employment_type"] = label
            break

    return result


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
    if field in {"monthly_income", "requested_amount"}:
        # A customer answering "How much would you like to borrow?"
        # directly with "5 lakh"/"50k" must not be truncated to its bare
        # leading digits — same Indian-English shorthand handled for the
        # free-text trigger message in extract_loan_fields_from_text.
        match = re.fullmatch(
            rf"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?){_AMOUNT_SUFFIX_RE}", value, re.I
        )
        if match:
            return _apply_amount_suffix(match.group(1), match.group(2))
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

    def handle(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None = None, trace_id: str = "") -> dict[str, Any]:
        step = workflow["step"]
        logger.info(f"[{trace_id}] Loan workflow step | phone={phone_number[-4:]} | step={step}")
        if step == STEP_SELECT_LOAN_TYPE:
            return self._select_type(workflow, phone_number, query, trace_id)
        if step == STEP_UPLOAD_LOAN_FORM:
            return self._collect_field(workflow, phone_number, query, parsed_document, trace_id)
        if step == STEP_CONFIRM_LOAN_ACCOUNT:
            return self._confirm_account(workflow, phone_number, query, trace_id)
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
                "\U0001F4DD Okay, which of these?"
            )}

        pending_content = workflow.get("data", {}).get("pending_document_content")

        # applicant_name and purpose are derived, never asked: the customer
        # is already registered (their name is on file), and the loan type
        # itself already says what the loan is for ("Home Loan" doesn't
        # need a separate "what's this for?" question).
        customer = get_customer_by_phone(phone_number)
        derived_fields = {
            "loan_type": loan_type,
            "applicant_name": (customer or {}).get("full_name", ""),
            "purpose": LOAN_LABELS.get(loan_type, loan_type),
        }
        update_workflow_data(phone_number, derived_fields)
        set_workflow_step(phone_number, STEP_UPLOAD_LOAN_FORM)
        logger.info(f"[{trace_id}] Loan type selected | phone={phone_number[-4:]} | loan_type={loan_type}")

        if pending_content:
            # A loan form was already uploaded before the loan type was
            # known (see message_handler.py's bare-upload auto-detect) —
            # apply it now instead of asking the customer to upload it
            # again.
            clear_workflow_data(phone_number, "pending_document_content")
            data = {**workflow.get("data", {}), **derived_fields}
            data.pop("pending_document_content", None)
            extracted = _extract(pending_content)
            data.update(extracted)
            update_workflow_data(phone_number, extracted)
            logger.info(
                f"[{trace_id}] Applied pending loan form after type selection | "
                f"phone={phone_number[-4:]} | fields={list(extracted)}"
            )
            return self._ask_next_or_confirm(phone_number, data, trace_id)

        # No document pending — but the same message that named the loan
        # type may already state other fields too ("personal loan of
        # 500000 for 24 months, I'm salaried") — fill whatever's
        # recognizable so the wizard only asks for what's still missing.
        extracted = extract_loan_fields_from_text(query)
        if extracted:
            data = {**workflow.get("data", {}), **derived_fields}
            data.update(extracted)
            update_workflow_data(phone_number, extracted)
            logger.info(
                f"[{trace_id}] Loan fields extracted from trigger text | "
                f"phone={phone_number[-4:]} | fields={list(extracted)}"
            )
            return self._ask_next_or_confirm(phone_number, data, trace_id)

        intro = templates.render_loan_type_selected(loan_type)
        offer = self._offer_account_confirmation(phone_number, {**workflow.get("data", {}), **derived_fields}, trace_id, intro=intro)
        if offer is not None:
            return offer
        return self._account_prompt(phone_number, intro=intro)

    def _collect_field(self, workflow: dict[str, Any], phone_number: str, query: str, parsed_document: dict | None, trace_id: str = "") -> dict[str, Any]:
        data = dict(workflow.get("data", {}))

        if parsed_document is not None:
            if not parsed_document.get("success"):
                return {"handled": True, "response": with_nav_buttons(templates.render_loan_document_unreadable())}
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
            return {"handled": True, "response": with_nav_buttons(self._answer_question(data, query, current_field))}

        text = query.strip()
        if not text:
            if current_field == "account_number":
                return self._account_prompt(phone_number)
            return {"handled": True, "response": templates.render_loan_field_prompt(current_field, loan_type=data.get("loan_type"))}

        if _is_acknowledgment(text):
            if current_field == "account_number":
                return self._account_prompt(phone_number, intro="No problem!")
            return {"handled": True, "response": "No problem! " + templates.render_loan_field_prompt(current_field, loan_type=data.get("loan_type"))}

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

        value, error = self._validate_field(current_field, text, phone_number, data.get("loan_type"))
        if error:
            if current_field == "account_number":
                return self._account_prompt(phone_number, error=error)
            return {"handled": True, "response": templates.render_loan_field_invalid(current_field, error)}

        data[current_field] = value
        update_workflow_data(phone_number, data)
        logger.info(f"[{trace_id}] Loan field collected | phone={phone_number[-4:]} | field={current_field}")
        return self._ask_next_or_confirm(phone_number, data, trace_id, just_completed_field=current_field)

    @staticmethod
    def _account_confirmation_prompt(account_number: str, intro: str | None = None) -> StructuredResponse:
        body = f"{intro}\n\n" if intro else ""
        body += f"{account_number} is your frequently used account. Would you like to proceed with it?"
        return StructuredResponse.buttons_of(
            body,
            [InteractiveButton(id="acct_yes", title="Yes, proceed"), InteractiveButton(id="acct_no", title="No, thanks")],
        )

    def _offer_account_confirmation(self, phone_number: str, data: dict, trace_id: str = "", intro: str | None = None) -> dict[str, Any] | None:
        """Ask the customer to confirm their frequently used account (see
        app/database.py::get_frequently_used_account) before assuming it,
        instead of silently picking it — only a declined confirmation
        falls through to the full account-picker list (_account_prompt)."""
        if data.get("account_number"):
            return None
        accounts = get_accounts_by_phone(phone_number)
        account = get_frequently_used_account(phone_number, accounts=accounts)
        if not account:
            return None

        update_workflow_data(phone_number, {"suggested_account_number": account["account_number"]})
        set_workflow_step(phone_number, STEP_CONFIRM_LOAN_ACCOUNT)
        logger.info(
            f"[{trace_id}] Offering frequently used loan account | "
            f"phone={phone_number[-4:]} | account={account['account_number']}"
        )
        return {"handled": True, "response": self._account_confirmation_prompt(account["account_number"], intro=intro)}

    def _confirm_account(self, workflow: dict[str, Any], phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        data = dict(workflow.get("data", {}))
        suggested = data.get("suggested_account_number")
        text = query.strip().lower()
        if text == "acct_yes":
            answer = "yes"
        elif text == "acct_no":
            answer = "no"
        else:
            answer = interpret_confirmation(query, llm_fallback=_llm_fallback(
                ["yes", "no"], "Confirm the suggested account, or reply no to choose a different one."
            ))

        if answer == "yes" and suggested:
            data.pop("suggested_account_number", None)
            data["account_number"] = suggested
            update_workflow_data(phone_number, {"account_number": suggested})
            clear_workflow_data(phone_number, "suggested_account_number")
            set_workflow_step(phone_number, STEP_UPLOAD_LOAN_FORM)
            logger.info(f"[{trace_id}] Loan account confirmed | phone={phone_number[-4:]} | account={suggested}")
            result = self._ask_next_or_confirm(phone_number, data, trace_id)
            ack = f"✅ Continuing with {suggested}."
            response = result["response"]
            if isinstance(response, str):
                result["response"] = f"{ack}\n\n{response}"
            else:
                response.text = f"{ack}\n\n{response.text}"
            return result

        if answer == "no":
            clear_workflow_data(phone_number, "suggested_account_number")
            set_workflow_step(phone_number, STEP_UPLOAD_LOAN_FORM)
            logger.info(f"[{trace_id}] Loan account declined, showing full account list | phone={phone_number[-4:]}")
            return self._account_prompt(phone_number)

        return {
            "handled": True,
            "response": self._account_confirmation_prompt(suggested or "", intro="Sorry, I didn't quite catch that."),
        }

    def _ask_next_or_confirm(self, phone_number: str, data: dict, trace_id: str = "", just_completed_field: str | None = None) -> dict[str, Any]:
        next_field = _next_missing_field(data)
        if next_field == "account_number":
            ack = templates.FIELD_ACKS.get(just_completed_field, "") if just_completed_field else ""
            offer = self._offer_account_confirmation(phone_number, data, trace_id, intro=ack or None)
            if offer is not None:
                return offer
            return self._account_prompt(phone_number, intro=ack or None)
        if next_field:
            return {"handled": True, "response": templates.render_loan_field_prompt(next_field, just_completed_field, loan_type=data.get("loan_type"))}
        set_workflow_step(phone_number, STEP_CONFIRM_LOAN)
        logger.info(f"[{trace_id}] Loan form complete, awaiting confirmation | phone={phone_number[-4:]}")
        return {"handled": True, "response": self._confirmation(data)}

    @staticmethod
    def _account_prompt(phone_number: str, error: str | None = None, intro: str | None = None) -> dict[str, Any]:
        """Tap-to-reply account picker — a WhatsApp interactive list (Meta
        CTA), one row per account showing its type and a masked account
        number ("xxxx 2026"). A tapped row id or a typed full account
        number both resolve via _resolve_account_selection in
        _validate_field, so either input still works."""
        accounts = get_accounts_by_phone(phone_number)
        if not accounts:
            return {
                "handled": True,
                "response": "You don't have any accounts on file to link this loan to — please contact support.",
            }
        rows = [
            InteractiveListRow(
                id=str(index),
                title=f"{str(a['account_type']).title()} · xxxx {a['account_number'][-4:]}"[:24],
                description=f"Balance {format_currency(a['balance'], a.get('currency', 'INR'))}",
            )
            for index, a in enumerate(accounts, 1)
        ]
        body = f"{intro}\n\n" if intro else ""
        if error:
            body += f"{error}\n\n"
        body += FIELD_PROMPTS["account_number"]
        return {"handled": True, "response": StructuredResponse.list_of(
            body, "Choose account", [InteractiveListSection(title="Your accounts", rows=rows)]
        )}

    @staticmethod
    def _validate_field(field: str, text: str, phone_number: str, loan_type: str | None = None) -> tuple[str, str | None]:
        """Returns (normalized_value, error_message_or_None).

        No range/numeric validation for monthly_income, requested_amount, or
        tenure_months — whatever the customer enters for those is accepted
        as-is (normalized only). account_number and applicant_name still
        validate, since those aren't the checks that were asked to go.
        """
        if field == "account_number":
            resolved = _resolve_account_selection(phone_number, text)
            if not resolved:
                accounts = get_accounts_by_phone(phone_number)
                available = ", ".join(a["account_number"] for a in accounts) or "none on file"
                return "", f"I couldn't find that account on your profile (your accounts: {available})."
            return resolved, None
        if field == "applicant_name":
            if len(text) < 2 or not re.match(r"^[A-Za-z .'-]+$", text):
                return "", "That doesn't look like a valid name."
            return text.strip(), None
        if field in {"monthly_income", "requested_amount", "tenure_months"}:
            return _normalize_value(field, text), None
        if field == "employment_type":
            # Same closed-vocabulary matching extract_loan_fields_from_text()
            # already applies to the trigger message — previously only
            # consulted there, so answering the direct "What is your
            # employment type?" prompt with "business" (or "I run a
            # business") stored the raw, unnormalized text instead of
            # resolving to "Business". Longest-phrase-first so "business
            # owner" matches before the bare "business" substring does.
            lowered = text.strip().lower()
            for phrase, label in sorted(_EMPLOYMENT_TYPE_WORDS.items(), key=lambda kv: -len(kv[0])):
                if phrase in lowered:
                    return label, None
            return text.strip(), None
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
        prompt = templates.render_loan_field_prompt(current_field, loan_type=data.get("loan_type"))
        return f"ℹ️ {FIELD_LABELS[current_field]} means {definitions[current_field]}. {prompt}"

    def _confirmation(self, data: dict, intro: str | None = None) -> StructuredResponse:
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
        body = f"{intro}\n\n{summary}" if intro else summary
        return _yes_no_prompt(body)

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
            return {"handled": True, "response": with_nav_buttons(templates.render_loan_failed())}
        complete_workflow(phone_number)
        logger.info(f"[{trace_id}] Loan request created | phone={phone_number[-4:]} | request_id={request_id}")
        return {
            "handled": True,
            "response": build_receipt_response(
                templates.render_loan_success(request_id), "Loan Application Receipt", request_id,
                [
                    ("Loan Type", LOAN_LABELS.get(data.get("loan_type"), data.get("loan_type"))),
                    ("Applicant Name", data.get("applicant_name")),
                    ("Account Number", data.get("account_number")),
                    ("Monthly Income", data.get("monthly_income")),
                    ("Employment Type", data.get("employment_type")),
                    ("Requested Amount", data.get("requested_amount")),
                    ("Tenure (months)", data.get("tenure_months")),
                    ("Purpose", data.get("purpose")),
                    ("Status", "PENDING"),
                ],
            ),
        }

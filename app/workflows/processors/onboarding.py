import re
from datetime import datetime
from typing import Any

import psycopg2

from app.database import create_customer, create_zero_balance_account
from app.logger import get_logger
from app.memory import cache_active_account
from app.conversation.responses.common import render_main_menu_list
from app.conversation.responses import onboarding as templates
from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_COLLECT_PAN,
    STEP_CONFIRM_REGISTRATION,
    STEP_SELECT_ACCOUNT_TYPE,
)
from app.workflows.memory import (
    complete_workflow,
    get_workflow,
    set_workflow_step,
    update_workflow_data,
)
from app.workflows.nlu import interpret_confirmation
from app.services.llm_understanding import interpret_choice_llm, is_llm_fallback_enabled
from app.conversation.renderer import InteractiveButton, InteractiveListRow, InteractiveListSection, StructuredResponse

logger = get_logger(__name__)

AADHAAR_PATTERN = re.compile(r"^\d{12}$")
PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
HONORIFIC_TOKENS = {"mr", "mrs", "ms", "miss", "mx", "dr", "shri", "smt", "shrimati"}
PROFILE_FIELD_ALIASES = {
    "full_name": ("full_name", "name", "fullname", "holder_name", "cardholdername"),
    "date_of_birth": ("date_of_birth", "dob", "dateofbirth", "birth_date"),
    "address": ("address", "address_line", "addressline", "residential_address"),
    "guardian_name": ("father_name", "fathers_name", "spouse_name", "husband_name", "wife_name", "guardian_name"),
}


_ACCOUNT_TYPE_ROWS = [("1", "savings", "Savings Account"), ("2", "current", "Current Account"), ("3", "salary", "Salary Account")]


def account_type_list_prompt(intro: str) -> StructuredResponse:
    """Tap-to-reply account-type list — row ids "1"/"2"/"3" match
    _handle_select_account_type's `aliases` dict exact lookup."""
    rows = [InteractiveListRow(id=digit, title=title) for digit, _key, title in _ACCOUNT_TYPE_ROWS]
    return StructuredResponse.list_of(intro, "Choose account", [InteractiveListSection(title="Account types", rows=rows)])


def _yes_no_prompt(body: str) -> StructuredResponse:
    """Tap-to-reply Yes/No confirmation — ids "yes"/"no" are the exact
    tokens interpret_confirmation()'s regex matches."""
    return StructuredResponse.buttons_of(
        body, [InteractiveButton(id="yes", title="Yes, confirm"), InteractiveButton(id="no", title="No, restart")]
    )


class OnboardingWorkflowHandler:
    """
    Collects Aadhaar and PAN as images processed by the document parser,
    before creating the customer record. The customer's name, date of
    birth, address, and guardian name are all extracted from these two
    documents (see _extract_profile_fields) rather than asked for
    separately — Aadhaar and PAN are the source of truth for identity, and
    asking the customer to also type their name invites a mismatch the
    Aadhaar/PAN cross-check (_validate_profile_data) would then have to
    police anyway.
    """

    async def handle(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        parsed_document: dict | None = None,
        trace_id: str = "",
    ) -> dict[str, Any]:

        step = workflow["step"]
        logger.info(f"[{trace_id}] Onboarding workflow step | phone={phone_number[-4:]} | step={step}")

        if step == STEP_COLLECT_AADHAAR:
            return self._handle_collect_aadhaar(phone_number, query, parsed_document, trace_id)

        elif step == STEP_COLLECT_PAN:
            return self._handle_collect_pan(phone_number, query, parsed_document, trace_id)

        elif step == STEP_CONFIRM_REGISTRATION:
            return self._handle_confirm_registration(workflow, phone_number, query, trace_id)

        elif step == STEP_SELECT_ACCOUNT_TYPE:
            return self._handle_select_account_type(workflow, phone_number, query, trace_id)

        return {
            "handled": True,
            "response": "Unknown registration step."
        }

    def _handle_collect_aadhaar(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None,
        trace_id: str = "",
    ) -> dict[str, Any]:

        if self._is_stop_command(query):
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] Registration cancelled at Aadhaar step | phone={phone_number[-4:]}")
            return {
                "handled": True,
                "response": templates.render_registration_cancelled()
            }

        if not parsed_document or not parsed_document.get("mime_type", "").startswith("image/"):
            return {
                "handled": True,
                "response": templates.render_document_expected_not_text("Aadhaar card")
            }

        digits = re.sub(r"[\s-]", "", self._document_value(
            parsed_document.get("content"),
            "aadhaar_number", "aadhaar", "aadhar_number", "aadhar",
        ))

        if not AADHAAR_PATTERN.match(digits):
            logger.info(f"[{trace_id}] Aadhaar OCR unreadable/invalid | phone={phone_number[-4:]}")
            return {
                "handled": True,
                "response": (
                    "I couldn't read a valid 12-digit Aadhaar number from that image. "
                    "Please upload a clearer image of the Aadhaar card."
                )
            }

        workflow = get_workflow(phone_number) or {}
        stored_data = workflow.get("data", {}) or {}
        document_profile = self._extract_profile_fields(parsed_document.get("content"))
        mismatches = self._validate_profile_data(stored_data, document_profile)

        if mismatches:
            return {
                "handled": True,
                "response": (
                    "The Aadhaar details do not match the information you already shared. "
                    f"Please upload the correct Aadhaar card. Mismatch: {', '.join(self._label_for_fields(mismatches))}."
                ),
            }

        profile_updates = {
            field: value
            for field, value in document_profile.items()
            if value and not stored_data.get(field)
        }
        update_workflow_data(phone_number, {"aadhaar_number": digits, **profile_updates})
        set_workflow_step(phone_number, STEP_COLLECT_PAN)
        logger.info(f"[{trace_id}] Registration Aadhaar collected | phone={phone_number[-4:]}")

        return {
            "handled": True,
            "response": templates.render_ask_pan()
        }

    def _handle_collect_pan(
        self,
        phone_number: str,
        query: str,
        parsed_document: dict | None,
        trace_id: str = "",
    ) -> dict[str, Any]:

        if self._is_stop_command(query):
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] Registration cancelled at PAN step | phone={phone_number[-4:]}")
            return {
                "handled": True,
                "response": templates.render_registration_cancelled()
            }

        if not parsed_document or not parsed_document.get("mime_type", "").startswith("image/"):
            return {
                "handled": True,
                "response": templates.render_document_expected_not_text("PAN card")
            }

        pan = re.sub(r"\s", "", self._document_value(
            parsed_document.get("content"),
            "pan_number", "pan", "pan_card_number",
        )).upper()

        if not PAN_PATTERN.match(pan):
            logger.info(f"[{trace_id}] PAN OCR unreadable/invalid | phone={phone_number[-4:]}")
            return {
                "handled": True,
                "response": (
                    "I couldn't read a valid PAN number from that image. "
                    "Please upload a clearer image of the PAN card."
                )
            }

        workflow = get_workflow(phone_number) or {}
        stored_data = workflow.get("data", {}) or {}
        document_profile = self._extract_profile_fields(parsed_document.get("content"))
        mismatches = self._validate_profile_data(stored_data, document_profile)

        if mismatches:
            return {
                "handled": True,
                "response": (
                    "The PAN details do not match the information you already shared. "
                    f"Please upload the correct PAN card. Mismatch: {', '.join(self._label_for_fields(mismatches))}."
                ),
            }

        profile_updates = {
            field: value
            for field, value in document_profile.items()
            if value and not stored_data.get(field)
        }
        update_workflow_data(phone_number, {"pan_number": pan, **profile_updates})
        set_workflow_step(phone_number, STEP_CONFIRM_REGISTRATION)
        logger.info(f"[{trace_id}] Registration PAN collected | phone={phone_number[-4:]}")

        workflow_data = (get_workflow(phone_number) or {}).get("data", {}) or {}

        summary = templates.render_registration_summary(
            phone_number=phone_number,
            full_name=workflow_data.get("full_name", ""),
            date_of_birth=workflow_data.get("date_of_birth", ""),
            guardian_name=workflow_data.get("guardian_name", ""),
            address=workflow_data.get("address", ""),
        )
        return {"handled": True, "response": _yes_no_prompt(summary)}

    def _handle_confirm_registration(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        trace_id: str = "",
    ) -> dict[str, Any]:

        # "restart" is kept as its own literal synonym for decline here
        # (pre-dates the shared interpreter and is specific to this step —
        # not a general "no" phrasing worth adding to interpret_confirmation
        # itself).
        llm_fallback = (
            (lambda text: interpret_choice_llm(text, ["yes", "no"], "Confirm or restart the registration details summary just shown."))
            if is_llm_fallback_enabled() else None
        )
        answer = interpret_confirmation(query, llm_fallback=llm_fallback)
        is_restart = query.strip().lower() == "restart"

        if self._is_stop_command(query):
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] Registration cancelled at confirmation step | phone={phone_number[-4:]}")
            return {
                "handled": True,
                "response": templates.render_registration_cancelled()
            }

        if answer == "yes":

            data = workflow.get("data", {})

            try:
                customer = create_customer(
                    phone_number=phone_number,
                    full_name=data.get("full_name", ""),
                    aadhaar_number=data.get("aadhaar_number", ""),
                    pan_number=data.get("pan_number", ""),
                    date_of_birth=data.get("date_of_birth", ""),
                    guardian_name=data.get("guardian_name", ""),
                    address=data.get("address", ""),
                )
            except psycopg2.Error as e:
                logger.error(
                    f"[{trace_id}] Registration failed | phone={phone_number[-4:]} | error={e}"
                )
                complete_workflow(phone_number)
                return {
                    "handled": True,
                    "response": templates.render_registration_failed()
                }

            set_workflow_step(phone_number, STEP_SELECT_ACCOUNT_TYPE)
            logger.info(
                f"[{trace_id}] Customer record created | phone={phone_number[-4:]} | "
                f"customer_id={customer.get('id') if customer else None}"
            )

            return {
                "handled": True,
                "response": account_type_list_prompt(
                    templates.render_registration_success() + "\n\nWhich account would you like to open?"
                ),
            }

        elif answer == "no" or is_restart:
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] Registration declined at confirmation step | phone={phone_number[-4:]}")
            return {
                "handled": True,
                "response": templates.render_registration_declined()
            }

        return {
            "handled": True,
            "response": _yes_no_prompt("Please reply *YES* to confirm or *NO* to start over."),
        }

    def _handle_select_account_type(
        self,
        workflow: dict[str, Any],
        phone_number: str,
        query: str,
        trace_id: str = "",
    ) -> dict[str, Any]:
        aliases = {
            "1": "savings", "savings": "savings", "savings account": "savings",
            "2": "current", "current": "current", "current account": "current",
            "3": "salary", "salary": "salary", "salary account": "salary",
        }
        normalized = query.strip().lower()
        account_type = aliases.get(normalized)
        if not account_type:
            # Accept a full sentence ("I'd like a savings account please")
            # instead of requiring an exact match on the bare word/number —
            # check the longer alias keys first so "current account" isn't
            # matched by a shorter unrelated substring.
            for alias, candidate in sorted(aliases.items(), key=lambda pair: -len(pair[0])):
                if alias in normalized:
                    account_type = candidate
                    break
        if not account_type:
            return {
                "handled": True,
                "response": account_type_list_prompt("Sorry, I didn't catch that — which account type?"),
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
            logger.error(f"[{trace_id}] Account creation failed | phone={phone_number[-4:]} | error={e}")
            return {
                "handled": True,
                "response": templates.render_account_creation_failed(),
            }

        logger.info(
            f"[{trace_id}] Account opened | phone={phone_number[-4:]} | "
            f"account_number={account['account_number']} | type={account_type}"
        )

        complete_workflow(phone_number)
        return {
            "handled": True,
            "response": render_main_menu_list(
                data.get("full_name", "Customer"), greeting=False,
                prefix=templates.render_account_created(account["account_number"], account_type),
            ),
        }

    @staticmethod
    def _is_stop_command(query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            return False
        stop_phrases = {
            "stop", "quit", "exit", "cancel", "cancel it", "stop it",
            "i want to stop", "please stop", "end registration",
            "abort", "abort registration", "nevermind", "never mind",
            "that's enough", "enough", "done"
        }
        return any(phrase in text for phrase in stop_phrases)

    @staticmethod
    def _normalize_profile_value(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())

    def _extract_profile_fields(self, content: Any) -> dict[str, str]:
        extracted: dict[str, str] = {}
        for field_name, aliases in PROFILE_FIELD_ALIASES.items():
            value = self._document_value(content, *aliases)
            normalized = self._normalize_profile_value(value)
            if normalized:
                extracted[field_name] = str(value).strip()
        return extracted

    def _validate_profile_data(
        self,
        stored_data: dict[str, Any],
        document_data: dict[str, Any],
    ) -> list[str]:
        mismatches: list[str] = []
        for field in ("full_name", "date_of_birth", "address", "guardian_name"):
            stored_value = stored_data.get(field)
            document_value = document_data.get(field)
            if not stored_value or not document_value:
                continue
            if not self._values_match(field, stored_value, document_value):
                mismatches.append(field)
        return mismatches

    @staticmethod
    def _label_for_fields(fields: list[str]) -> list[str]:
        labels = {
            "full_name": "name",
            "date_of_birth": "date of birth",
            "address": "address",
            "guardian_name": "guardian / father / spouse name",
        }
        return [labels.get(field, field) for field in fields]

    def _values_match(self, field: str, stored_value: Any, document_value: Any) -> bool:
        if field == "date_of_birth":
            return self._normalize_date(stored_value) == self._normalize_date(document_value)
        if field in {"full_name", "guardian_name"}:
            return self._names_match(stored_value, document_value)
        return self._normalize_profile_value(stored_value) == self._normalize_profile_value(document_value)

    @staticmethod
    def _normalize_date(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return ""
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def _names_match(stored_value: Any, document_value: Any) -> bool:
        """
        Compare a customer-typed name against an OCR'd ID card name.

        Case-insensitive by construction (_normalize_profile_value lowercases
        before comparing). Also tolerant of the ways typed and printed names
        commonly diverge without actually being a different person: honorific
        prefixes (Mr/Dr/Shri/...), a middle name/initial present on the ID
        but omitted when typed (or vice versa), and surname-first ordering —
        none of those should be reported as a mismatch. A genuinely different
        name (extra or substituted tokens beyond an initial) still fails.
        """
        stored_tokens = [
            OnboardingWorkflowHandler._normalize_profile_value(token)
            for token in re.split(r"[\s.-]+", str(stored_value)) if token
        ]
        document_tokens = [
            OnboardingWorkflowHandler._normalize_profile_value(token)
            for token in re.split(r"[\s.-]+", str(document_value)) if token
        ]
        stored_tokens = [t for t in stored_tokens if t and t not in HONORIFIC_TOKENS]
        document_tokens = [t for t in document_tokens if t and t not in HONORIFIC_TOKENS]

        if not stored_tokens or not document_tokens:
            return False

        def _token_matches(a: str, b: str) -> bool:
            if a == b:
                return True
            if len(a) == 1 and b.startswith(a):
                return True
            if len(b) == 1 and a.startswith(b):
                return True
            return False

        def _covers(shorter: list[str], longer: list[str]) -> bool:
            remaining = list(longer)
            for token in shorter:
                match_index = next(
                    (i for i, candidate in enumerate(remaining) if _token_matches(token, candidate)),
                    None,
                )
                if match_index is None:
                    return False
                remaining.pop(match_index)
            return True

        return _covers(stored_tokens, document_tokens) or _covers(document_tokens, stored_tokens)

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

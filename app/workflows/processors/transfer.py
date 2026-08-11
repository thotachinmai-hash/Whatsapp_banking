import re
import secrets
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from app.database import create_beneficiary, create_transfer, get_accounts_by_phone, get_beneficiaries_by_phone
from app.logger import get_logger
from app.workflows.constants import (
    STEP_COLLECT_AMOUNT,
    STEP_COLLECT_BENEFICIARY_ACCOUNT,
    STEP_COLLECT_BENEFICIARY_NAME,
    STEP_CONFIRM_TRANSFER,
    STEP_SELECT_AMOUNT,
    STEP_SELECT_BENEFICIARY,
    STEP_SELECT_SOURCE_ACCOUNT,
    WORKFLOW_TRANSFER,
)
from app.workflows.memory import complete_workflow, create_workflow, create_workflow_model, set_workflow_step, update_workflow_data
from app.workflows.nlu import interpret_confirmation, interpret_menu_choice
from app.services.llm_understanding import interpret_choice_llm, is_llm_fallback_enabled
from app.conversation.renderer import InteractiveButton, InteractiveListRow, InteractiveListSection, StructuredResponse
from app.conversation.responses.common import format_currency, mask_account_number as _mask_account
from app.conversation.responses import transfer as templates

logger = get_logger(__name__)

# Trailing filler a customer often tacks onto the beneficiary's name in a
# free-form request ("send 500 to Priya please", "...to Priya now") — these
# get trimmed off the captured name rather than swallowed into it or (the
# old failure mode) blocking the match entirely because they weren't the
# very last word in the message.
_NAME_TRAILERS = re.compile(
    r"\s+(?:please|now|today|thanks|thank you|asap|urgently)\W*$", re.I
)
_BENEFICIARY_INTENT_RE = re.compile(
    r"(?:transfer|send|pay)\b.*?\bto\s+([A-Za-z][A-Za-z .'\-]{1,50})", re.I
)
_AMOUNT_INTENT_RE = re.compile(r"(?:£|GBP|Rs\.?|INR)?\s*([0-9]+(?:[.,][0-9]{1,2})?)\s*(k)?", re.I)


def _looks_like_account_number(candidate: str) -> bool:
    """The account-number step used to accept ANY text of 6+ characters —
    so an unrelated message typed at that step (e.g. "List all my saved
    beneficiary", asked instead of answering) got its spaces stripped and
    saved as a literal beneficiary account number ("LISTALLMYSAVED
    BENEFICIARY"), and a real transfer was then created against that
    garbage account. A real account number is alphanumeric, contains at
    least one digit, and isn't absurdly long — a stray sentence typically
    fails at least one of those."""
    return (
        6 <= len(candidate) <= 34
        and candidate.isalnum()
        and any(ch.isdigit() for ch in candidate)
    )


def _llm_fallback(options: list[str], context: str):
    """Bind an LLM choice fallback for `options`, or None when the feature
    flag is off — see interpret_menu_choice/interpret_confirmation in
    app/workflows/nlu.py for how this is consumed."""
    if not is_llm_fallback_enabled():
        return None
    return lambda text: interpret_choice_llm(text, options, context)


def has_transferable_balance(phone_number: str) -> bool:
    """True if the customer has at least one account with a positive balance."""
    accounts = get_accounts_by_phone(phone_number)
    return any(float(a["balance"]) > 0 for a in accounts)


def start_transfer_from_text(
    phone_number: str,
    query: str,
    transfer_handler: "TransferWorkflowProcessor",
    trace_id: str = "",
) -> dict[str, Any]:
    """Start a transfer workflow, parsing whatever beneficiary name and/or
    amount the customer already gave in the message that triggered it
    (e.g. "send 500 to Priya", "transfer 200 to Amit") — so those steps are
    skipped instead of being asked again. Falls back to the plain
    beneficiary-selection prompt when nothing usable is found in the text.

    This is the one place that logic lives — both the deterministic
    keyword-triggered path (WorkflowManager.start_requested) and the
    LLM-intent-triggered path (workflow_adapter.start_workflow_directly)
    call this instead of each re-parsing (or, previously, one of them not
    parsing at all and just re-asking from scratch)."""
    if not has_transferable_balance(phone_number):
        return {"handled": True, "response": templates.render_insufficient_balance()}

    requested_name = ""
    name_match = _BENEFICIARY_INTENT_RE.search(query.strip())
    if name_match:
        candidate = _NAME_TRAILERS.sub("", name_match.group(1).strip())
        requested_name = candidate.strip().title()

    beneficiaries = get_beneficiaries_by_phone(phone_number)
    saved = next(
        (
            item for item in beneficiaries
            if item["beneficiary_name"].lower() in requested_name.lower()
            or requested_name.lower() in item["beneficiary_name"].lower()
        ),
        None,
    ) if requested_name else None

    amount: Optional[str] = None
    amount_match = _AMOUNT_INTENT_RE.search(query)
    if amount_match:
        try:
            raw_amount = amount_match.group(1).replace(",", "")
            amount_value = float(raw_amount) * (1000 if amount_match.group(2) else 1)
            if 0 < amount_value <= 100000:
                amount = f"£{amount_value:,.2f}"
        except ValueError:
            amount = None

    data: dict[str, Any] = {}
    if amount:
        data["amount"] = amount
    if saved:
        data.update({"beneficiary_name": saved["beneficiary_name"], "beneficiary_account": saved["account_number"]})
        step = STEP_SELECT_SOURCE_ACCOUNT if amount else STEP_SELECT_AMOUNT
    elif requested_name:
        data["beneficiary_name"] = requested_name
        step = STEP_COLLECT_BENEFICIARY_ACCOUNT
    else:
        step = STEP_SELECT_BENEFICIARY

    workflow = create_workflow_model(WORKFLOW_TRANSFER, step, data=data)
    create_workflow(phone_number, workflow)
    logger.info(
        f"[{trace_id}] Transfer started from free text | phone={phone_number[-4:]} | "
        f"beneficiary={'saved' if saved else ('new' if requested_name else 'none')} | amount={'set' if amount else 'none'}"
    )

    if requested_name and saved:
        response = (
            transfer_handler._source_prompt(phone_number)["response"] if amount
            else templates.render_transfer_amount_prompt()
        )
        if not amount:
            response = f"\U0001F44B Found {saved['beneficiary_name']} in your saved beneficiaries.\n\n" + response
    elif requested_name:
        response = (
            f"\U0001F44B I couldn't find {requested_name} in your saved beneficiaries — "
            f"let's add them. What's {requested_name}'s account number?"
        )
        if amount:
            response += f"\n\n(The amount's already set to {amount}.)"
    else:
        response = transfer_handler._beneficiary_prompt(phone_number)["response"]

    return {"handled": True, "response": response}


class TransferWorkflowProcessor:
    """Small, explicit state machine for a low-error transfer conversation."""

    def handle(self, workflow: dict, phone_number: str, query: str, trace_id: str = "") -> dict:
        text = query.strip()
        command = text.lower()
        logger.info(f"[{trace_id}] Transfer workflow step | phone={phone_number[-4:]} | step={workflow['step']}")
        if command in {"cancel", "c", "stop", "quit", "exit", "please cancel", "please stop"}:
            complete_workflow(phone_number)
            logger.info(f"[{trace_id}] Transfer cancelled | phone={phone_number[-4:]}")
            return {"handled": True, "response": templates.render_transfer_cancelled()}
        if command in {"back", "b", "go back", "please go back", "previous", "previous step"}:
            return self._back(workflow, phone_number)

        step = workflow["step"]
        if step == STEP_SELECT_BENEFICIARY:
            return self._beneficiary(workflow, phone_number, text)
        if step == STEP_COLLECT_BENEFICIARY_NAME:
            if len(text) < 2 or not re.match(r"^[A-Za-z .'-]+$", text):
                return self._ask(templates.render_invalid_beneficiary_name(), "Reply Back or Cancel.")
            update_workflow_data(phone_number, {"beneficiary_name": text.strip().title()})
            set_workflow_step(phone_number, STEP_COLLECT_BENEFICIARY_ACCOUNT)
            return {"handled": True, "response": templates.render_beneficiary_account_prompt(text.strip().title())}
        if step == STEP_COLLECT_BENEFICIARY_ACCOUNT:
            account_number = re.sub(r"\s", "", text).upper()
            if not _looks_like_account_number(account_number):
                return self._ask(templates.render_invalid_beneficiary_account(), "Reply Back or Cancel.")
            beneficiary_name = workflow.get("data", {}).get("beneficiary_name", "")
            create_beneficiary(phone_number, beneficiary_name, account_number)
            logger.info(
                f"[{trace_id}] Beneficiary saved | phone={phone_number[-4:]} | "
                f"beneficiary={beneficiary_name} | account={_mask_account(account_number)}"
            )
            update_workflow_data(phone_number, {"beneficiary_account": account_number})
            if workflow.get("data", {}).get("amount"):
                set_workflow_step(phone_number, STEP_SELECT_SOURCE_ACCOUNT)
                return self._source_prompt(phone_number)
            set_workflow_step(phone_number, STEP_SELECT_AMOUNT)
            return self._amount_prompt(f"✅ Saved {beneficiary_name} as a new beneficiary.")
        if step in {STEP_SELECT_AMOUNT, STEP_COLLECT_AMOUNT}:
            return self._amount(workflow, phone_number, text)
        if step == STEP_SELECT_SOURCE_ACCOUNT:
            return self._source_account(workflow, phone_number, text)
        if step == STEP_CONFIRM_TRANSFER:
            confirmation = interpret_confirmation(
                text, llm_fallback=_llm_fallback(["yes", "no"], "Confirm or edit the transfer summary just shown.")
            )
            if command in {"1", "confirm transfer"} or confirmation == "yes":
                data = workflow.get("data", {})
                reference = f"TRF-{secrets.token_hex(4).upper()}"
                amount_value = self._parse_amount(data.get("amount", "")) or Decimal("0")
                try:
                    create_transfer(
                        reference=reference,
                        phone_number=phone_number,
                        source_account=data.get("source_account", ""),
                        beneficiary_name=data.get("beneficiary_name", ""),
                        beneficiary_account=data.get("beneficiary_account", ""),
                        amount=amount_value,
                        status="INITIATED",
                    )
                except Exception as e:
                    logger.error(f"[{trace_id}] Transfer persistence failed | phone={phone_number[-4:]} | error={e}")
                    return self._ask(
                        templates.render_transfer_failed(),
                        "Reply *Back* to review or *Cancel* to stop.",
                    )
                complete_workflow(phone_number)
                logger.info(
                    f"[{trace_id}] Transfer initiated | phone={phone_number[-4:]} | "
                    f"reference={reference} | amount={data.get('amount')}"
                )
                return {
                    "handled": True,
                    "response": templates.render_transfer_success(
                        reference=reference,
                        beneficiary_name=data.get("beneficiary_name"),
                        beneficiary_account_masked=data.get("beneficiary_account"),
                        amount_label=data.get("amount"),
                        source_account_label=data.get("source_account"),
                    ),
                }
            if command == "2" or confirmation == "no" or "edit" in command or "change" in command:
                set_workflow_step(phone_number, STEP_SELECT_AMOUNT)
                return self._amount_prompt()
            return self._ask("Please choose 1 to confirm or 2 to edit.", "Reply Back or Cancel.")
        return {"handled": False, "response": None}

    def _beneficiary(self, workflow: dict, phone_number: str, text: str) -> dict:
        beneficiaries = get_beneficiaries_by_phone(phone_number)
        choice = text.lower()
        selected = None
        if choice.isdigit() and 1 <= int(choice) <= len(beneficiaries):
            selected = beneficiaries[int(choice) - 1]
        else:
            selected = next(
                (item for item in beneficiaries if item["beneficiary_name"].lower() in choice),
                None,
            )
        if not selected:
            # Accept natural language such as “I want to pay to Ramya” at the
            # beneficiary step instead of forcing the customer to repeat it.
            # If the name isn't a saved beneficiary, treat it as a new one.
            match = re.search(r"(?:pay|send|transfer)\s+(?:money\s+)?to\s+([a-z][a-z .'-]{1,50})$", text, re.I)
            if match:
                name = match.group(1).strip().title()
                amount = self._parse_amount(text)
                updates = {"beneficiary_name": name}
                if amount:
                    updates["amount"] = f"£{amount:,.2f}"
                update_workflow_data(phone_number, updates)
                set_workflow_step(phone_number, STEP_COLLECT_BENEFICIARY_ACCOUNT)
                return self._ask(
                    f"\U0001F44B Got it — you want to pay {name}. What is {name}'s account number?",
                    "Reply *Back* to choose a saved beneficiary or *Cancel* to stop.",
                )
        if selected:
            # A bare numeric selection (e.g. "1") must not also be read as an
            # amount — only look for a combined amount when the customer
            # wrote something beyond the plain menu digit (e.g. "Priya 50").
            amount = self._parse_amount(text) if not choice.isdigit() else None
            update_workflow_data(phone_number, {
                "beneficiary_name": selected["beneficiary_name"],
                "beneficiary_account": selected["account_number"],
            })
            if amount:
                update_workflow_data(phone_number, {"amount": f"£{amount:,.2f}"})
                set_workflow_step(phone_number, STEP_SELECT_SOURCE_ACCOUNT)
                return self._source_prompt(phone_number)
            set_workflow_step(phone_number, STEP_SELECT_AMOUNT)
            return self._amount_prompt()
        if choice in {str(len(beneficiaries) + 1), "new", "new beneficiary", "someone new", "add", "add beneficiary", "add new"}:
            set_workflow_step(phone_number, STEP_COLLECT_BENEFICIARY_NAME)
            return {"handled": True, "response": templates.render_new_beneficiary()}

        menu_options = [str(i) for i in range(1, len(beneficiaries) + 2)] + [b["beneficiary_name"] for b in beneficiaries]
        llm_choice = interpret_menu_choice(
            text, menu_options,
            llm_fallback=_llm_fallback(menu_options, "Choose a saved beneficiary by number/name, or 'new' to add one."),
        )
        if llm_choice and llm_choice.isdigit():
            return self._beneficiary(workflow, phone_number, llm_choice)
        if llm_choice:
            matched = next((b for b in beneficiaries if b["beneficiary_name"] == llm_choice), None)
            if matched:
                return self._beneficiary(workflow, phone_number, str(beneficiaries.index(matched) + 1))
        return self._beneficiary_prompt(phone_number, "Please choose one of the options below.")

    def _amount(self, workflow: dict, phone_number: str, text: str) -> dict:
        quick_amounts = {"1": Decimal("25"), "2": Decimal("50"), "3": Decimal("100"), "4": Decimal("250")}
        if text.strip() in quick_amounts:
            value = quick_amounts[text.strip()]
        elif text.strip() == "5":
            set_workflow_step(phone_number, STEP_COLLECT_AMOUNT)
            return self._ask("What amount would you like to send?", "For example: 125. Reply Back or Cancel.")
        else:
            value = self._parse_amount(text)
        if value is None:
            return self._amount_prompt(templates.render_invalid_amount())
        update_workflow_data(phone_number, {"amount": f"£{value:,.2f}"})
        set_workflow_step(phone_number, STEP_SELECT_SOURCE_ACCOUNT)
        return self._source_prompt(phone_number)

    def _source_account(self, workflow: dict, phone_number: str, text: str) -> dict:
        accounts = workflow.get("data", {}).get("source_accounts") or get_accounts_by_phone(phone_number)
        choice = text.strip().lower()
        account = None
        if choice.isdigit() and 1 <= int(choice) <= len(accounts):
            account = accounts[int(choice) - 1]
        if account is None:
            account = next((a for a in accounts if a["account_number"].lower() == choice), None)
        if account is None:
            # Free-form phrasing like "savings account" or "use my current
            # account" instead of the exact number/index — only resolves
            # unambiguously when exactly one account matches the type named.
            type_matches = [a for a in accounts if str(a["account_type"]).lower() in choice]
            if len(type_matches) == 1:
                account = type_matches[0]
        if not account:
            menu_options = [str(i) for i in range(1, len(accounts) + 1)] + [a["account_number"] for a in accounts]
            llm_choice = interpret_menu_choice(
                text, menu_options,
                llm_fallback=_llm_fallback(menu_options, "Choose a source account by number or account number."),
            )
            if llm_choice and llm_choice.isdigit():
                return self._source_account(workflow, phone_number, llm_choice)
            if llm_choice:
                return self._source_account(workflow, phone_number, llm_choice)
            return self._source_prompt(phone_number, "Please choose one of the numbered accounts.")

        data = workflow.get("data", {})
        balance = Decimal(str(account["balance"]))
        amount_value = self._parse_amount(data.get("amount", "")) or Decimal("0")

        if balance <= 0:
            return self._source_prompt(
                phone_number,
                templates.render_insufficient_balance(account_label=account["account_number"]),
            )
        if amount_value > balance:
            return self._source_prompt(
                phone_number,
                templates.render_insufficient_balance(
                    account_label=account["account_number"],
                    available_label=f"£{balance:,.2f}",
                    amount_label=data.get("amount"),
                ),
            )

        update_workflow_data(phone_number, {"source_account": account["account_number"]})
        set_workflow_step(phone_number, STEP_CONFIRM_TRANSFER)
        summary = templates.render_transfer_summary(
            beneficiary_name=data.get("beneficiary_name"),
            beneficiary_account_masked=data.get("beneficiary_account"),
            amount_label=data.get("amount"),
            source_account_label=account["account_number"],
        )
        return {"handled": True, "response": self._confirm_prompt(summary)}

    def _back(self, workflow: dict, phone_number: str) -> dict:
        previous = {
            STEP_COLLECT_BENEFICIARY_NAME: STEP_SELECT_BENEFICIARY,
            STEP_COLLECT_BENEFICIARY_ACCOUNT: STEP_COLLECT_BENEFICIARY_NAME,
            STEP_SELECT_AMOUNT: STEP_SELECT_BENEFICIARY,
            STEP_COLLECT_AMOUNT: STEP_SELECT_AMOUNT,
            STEP_SELECT_SOURCE_ACCOUNT: STEP_SELECT_AMOUNT,
            STEP_CONFIRM_TRANSFER: STEP_SELECT_SOURCE_ACCOUNT,
        }.get(workflow["step"])
        if not previous:
            return self._beneficiary_prompt(phone_number)
        set_workflow_step(phone_number, previous)
        if previous == STEP_SELECT_BENEFICIARY:
            return self._beneficiary_prompt(phone_number)
        if previous == STEP_SELECT_AMOUNT:
            return self._amount_prompt()
        if previous == STEP_SELECT_SOURCE_ACCOUNT:
            return self._source_prompt(phone_number)
        if previous == STEP_CONFIRM_TRANSFER:
            return self._confirm_prompt("Let's review this transfer again.")
        return self._ask("What is the beneficiary's name?", "Reply Back or Cancel.")

    @staticmethod
    def _confirm_prompt(summary: str) -> dict:
        """Ready-to-send confirmation with tap-to-reply Yes/Edit buttons —
        ids "1"/"2" match the digit fast-path STEP_CONFIRM_TRANSFER already
        checks (see handle()), so a typed "1"/"2" still works exactly as
        before if the interactive send falls back to plain text."""
        return StructuredResponse.buttons_of(
            summary,
            [InteractiveButton(id="1", title="Yes, send it"), InteractiveButton(id="2", title="Edit amount")],
        )

    @staticmethod
    def _parse_amount(text: str) -> Decimal | None:
        match = re.search(r"(?:£|GBP|Rs\.?|INR)?\s*([0-9]+(?:[.,][0-9]{1,2})?)", text, re.I)
        if not match:
            return None
        try:
            value = Decimal(match.group(1).replace(",", ""))
            return value if value > 0 and value <= 100000 else None
        except InvalidOperation:
            return None

    @staticmethod
    def _ask(question: str, footer: str) -> dict:
        return {"handled": True, "response": f"{question}\n\n{footer}"}

    @staticmethod
    def _beneficiary_prompt(phone_number: str, error: str | None = None) -> dict:
        beneficiaries = get_beneficiaries_by_phone(phone_number)
        if not beneficiaries:
            # Nothing saved yet — go straight to adding the first one instead
            # of showing an empty list with only a "someone new" option.
            set_workflow_step(phone_number, STEP_COLLECT_BENEFICIARY_NAME)
            return {"handled": True, "response": templates.render_beneficiary_selection(beneficiaries, error=error)}
        # 1 extra row for "Add new beneficiary" — beyond WhatsApp's 10-row
        # cap, fall back to the plain numbered text list rather than fail.
        if len(beneficiaries) + 1 > 10:
            return {"handled": True, "response": templates.render_beneficiary_selection(beneficiaries, error=error)}
        rows = [
            InteractiveListRow(
                id=str(index), title=b["beneficiary_name"][:24],
                description=_mask_account(b["account_number"]),
            )
            for index, b in enumerate(beneficiaries, 1)
        ]
        rows.append(InteractiveListRow(id="new", title="Add new beneficiary"))
        intro = f"{error}\n\n" if error else ""
        intro += "\U0001F464 Sure, who would you like to pay?"
        return {"handled": True, "response": StructuredResponse.list_of(
            intro, "Choose beneficiary", [InteractiveListSection(title="Beneficiaries", rows=rows)]
        )}

    @staticmethod
    def _amount_prompt(error: str | None = None) -> dict:
        quick_amounts = [("1", "£25"), ("2", "£50"), ("3", "£100"), ("4", "£250"), ("5", "A different amount")]
        rows = [InteractiveListRow(id=digit, title=label) for digit, label in quick_amounts]
        intro = f"{error}\n\n" if error else ""
        intro += "\U0001F4B0 How much would you like to send?"
        return {"handled": True, "response": StructuredResponse.list_of(
            intro, "Choose amount", [InteractiveListSection(title="Quick amounts", rows=rows)]
        )}

    @staticmethod
    def _source_prompt(phone_number: str, error: str | None = None) -> dict:
        accounts = get_accounts_by_phone(phone_number)
        if len(accounts) > 10:
            return {"handled": True, "response": templates.render_source_account_prompt(accounts, error=error)}
        rows = [
            InteractiveListRow(
                id=str(index), title=f"{str(a['account_type']).title()} · {a['account_number']}"[:24],
                description=f"Balance {format_currency(a['balance'], a.get('currency', 'GBP'))}",
            )
            for index, a in enumerate(accounts, 1)
        ]
        intro = f"{error}\n\n" if error else ""
        intro += "\U0001F3E6 Which account should this come from?"
        return {"handled": True, "response": StructuredResponse.list_of(
            intro, "Choose account", [InteractiveListSection(title="Your accounts", rows=rows)]
        )}

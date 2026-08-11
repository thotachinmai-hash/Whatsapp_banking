"""Formatting helpers + non-domain-specific response templates.

ARCHITECTURE RULE (see docs/current_architecture.md, "Response Template
Layer — Phase 4"): everything in app/conversation/responses/ formats
presentation text only. It never decides whether a transfer is valid,
whether a loan is approved, or queries a database — the workflow
processors / tools remain the sole source of that data and pass only the
already-decided values in.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from app.conversation.renderer import InteractiveListRow, InteractiveListSection, StructuredResponse

_CURRENCY_SYMBOLS = {"GBP": "£", "INR": "₹", "USD": "$", "EUR": "€"}


# ─── formatting helpers ─────────────────────────────────────────────────

def format_currency(amount: Any, currency: str = "GBP") -> str:
    """£500, £1,250, £50,000 — never a raw Python repr like 500.0."""
    symbol = _CURRENCY_SYMBOLS.get(str(currency or "GBP").upper(), "")
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        return f"{symbol}{amount}"
    return f"{symbol}{value:,.2f}"


def format_amount(amount: Any) -> str:
    """Currency-less numeric formatting (e.g. a tenure or a count)."""
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        return str(amount)
    if value == value.to_integral_value():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def mask_account_number(account_number: Any) -> str:
    """The one place account-number masking happens — every template that
    needs to show someone else's (e.g. a beneficiary's) account number
    calls this rather than re-implementing the masking rule."""
    digits = re.sub(r"\s", "", str(account_number or ""))
    return f"•••• {digits[-4:]}" if len(digits) >= 4 else digits


def format_account_label(account_number: Any, account_type: Optional[str] = None, masked: bool = True) -> str:
    shown = mask_account_number(account_number) if masked else str(account_number)
    if account_type:
        return f"{str(account_type).title()} Account {shown}"
    return f"Account {shown}"


def format_date(value: Any) -> str:
    text = str(value or "").strip()
    return text[:10] if text else "Not available"


def format_status(status: Any) -> str:
    return str(status or "").replace("_", " ").strip().title() or "Unknown"


def format_transaction(transaction: dict) -> str:
    """One line for a single transaction — used by status.py's transaction list."""
    amount = format_currency(transaction.get("amount"), transaction.get("currency", "GBP"))
    sign = "+" if transaction.get("type") == "credit" else "-"
    date = format_date(transaction.get("date"))
    description = transaction.get("description") or transaction.get("category") or "Transaction"
    return f"{date} · {description} · {sign}{amount}"


# ─── common templates ───────────────────────────────────────────────────

def render_greeting(name: Optional[str] = None) -> str:
    if name:
        return f"Hi {name}, welcome back! \U0001F44B"
    return "Hi there! \U0001F44B"


def render_help() -> str:
    return "I can help with balances, transfers, loans, cheques, KYC and transactions.\n\nWhat would you like to do?"


def render_main_menu(name: Optional[str] = None, greeting: bool = True) -> str:
    header = f"Hi {name}, welcome back to Finacle Banking! \U0001F44B\n\n" if (greeting and name) else ""
    return (
        header
        + "\U0001F4CB *What would you like to do?*\n"
        + "I can help with transfers, balances, transactions, cheques, loans and KYC.\n\n"
        + "1. \U0001F4B8 *Transfer money*\n"
        + "2. \U0001F4B0 *Check balance*\n"
        + "3. \U0001F4C4 *View transactions*\n"
        + "4. \U0001F9FE *Deposit a cheque*\n"
        + "5. \U0001F50E *Check cheque status*\n"
        + "6. \U0001F4DD *Apply for a loan*\n"
        + "7. \U0001F4C7 *Update KYC*\n\n"
        + "Reply with a number, type a request, or send a voice note.\n"
        + "Reply *Back* or *Cancel* anytime."
    )


_MAIN_MENU_ROWS = [
    ("1", "Transfer money", "\U0001F4B8 Send money to a beneficiary"),
    ("2", "Check balance", "\U0001F4B0 View your account balance"),
    ("3", "View transactions", "\U0001F4C4 See recent transactions"),
    ("4", "Deposit a cheque", "\U0001F9FE Upload a cheque image"),
    ("5", "Check cheque status", "\U0001F50E Track a submitted cheque"),
    ("6", "Apply for a loan", "\U0001F4DD Start a loan application"),
    ("7", "Update KYC", "\U0001F4C7 Update your identity details"),
]


def render_main_menu_list(name: Optional[str] = None, greeting: bool = True, prefix: str = "") -> StructuredResponse:
    """Tap-to-reply main menu — row ids "1".."7" match the digit lookup
    `menu_actions` already uses in app/workflows/manager.py::start_requested,
    so a typed digit still works exactly as before if the list send falls
    back to text. `prefix` lets a caller prepend other text (e.g. a
    cancellation notice) into the SAME message instead of concatenating a
    second render_main_menu() call, which isn't possible once this is a
    list instead of a plain string."""
    header = f"Hi {name}, welcome back to Finacle Banking! \U0001F44B\n\n" if (greeting and name) else ""
    body = f"{prefix}\n\n" if prefix else ""
    body += header + "\U0001F4CB *What would you like to do?*"
    rows = [InteractiveListRow(id=digit, title=title, description=desc) for digit, title, desc in _MAIN_MENU_ROWS]
    return StructuredResponse.list_of(body, "Menu", [InteractiveListSection(title="Services", rows=rows)])


def render_goodbye() -> str:
    return "Thank you for banking with Finacle! 🙏 Happy to help — just message us again anytime you need us."


def render_cancelled(item: str = "This request") -> str:
    return f"✅ {item} cancelled. Nothing was submitted or changed."


def render_back(prompt: str) -> str:
    return f"{prompt}\n\nReply *Back* or *Cancel* anytime."


def render_confirmation(question: str, footer: str = "Reply *Back* or *Cancel* anytime.") -> str:
    return f"{question}\n\n{footer}"


def render_yes_no_prompt(question: str) -> str:
    return f"{question}\n\n1. YES\n2. NO"


def render_retry(message: str = "Please try again.") -> str:
    return message


def render_invalid_input(hint: Optional[str] = None) -> str:
    base = "I didn't quite understand that."
    return f"{base} {hint}" if hint else f"{base} Please try again."


def render_processing(what: str = "your request") -> str:
    return f"One moment, I'm checking {what}..."


def render_success(message: str) -> str:
    return f"✅ {message}"


def render_failure(message: str = "That didn't go through.") -> str:
    return f"⚠️ {message}"


def render_service_unavailable() -> str:
    return "I'm sorry, the service is temporarily busy. Please try again in a few minutes."


def render_out_of_scope() -> str:
    return (
        "I'm here to help with your banking needs — accounts, balances, "
        "transfers, loans, cheques, KYC and transactions.\n\n"
        "What would you like to do?"
    )


def render_low_confidence() -> str:
    return (
        "I'm not quite sure I understood that. I can help with balances, "
        "transfers, loans, cheques, KYC and transactions — could you "
        "let me know what you'd like to do?"
    )


_CLARIFICATION_PROMPTS = {
    "transfer_request": (
        "Would you like to send money to someone? If so, just tell me who "
        "you'd like to pay and how much."
    ),
    "loan_application_request": (
        "Would you like to start a loan application? If so, let me know "
        "which type — personal, home, vehicle, or education."
    ),
    "cheque_deposit_request": (
        "Would you like to deposit a cheque? If so, please upload a clear "
        "photo of it and I'll get started."
    ),
    "kyc_update_request": (
        "Would you like to update your KYC details? If so, please upload "
        "your KYC document or let me know what needs to change."
    ),
    "registration_request": (
        "Would you like to register for an account? Just say \"register\" "
        "and I'll get you set up."
    ),
}
_DEFAULT_CLARIFICATION = (
    "I want to make sure I get this right — could you tell me a bit more "
    "about what you'd like to do?"
)


def render_clarification(intent: Optional[str] = None) -> str:
    return _CLARIFICATION_PROMPTS.get(intent, _DEFAULT_CLARIFICATION)


def render_workflow_boundary(label: str) -> str:
    """The reply when a message is outside the scope of the customer's
    active workflow (e.g. asking an unrelated question mid-transfer).
    Prefer render_workflow_boundary_with_step() when the current step is
    known — it gives the same protection with a friendlier, more useful
    reply (Task 10, Parts 9/10) instead of the rigid "I can answer
    questions only about this request here."."""
    return (
        f"You are currently working on a {label}. I can answer questions only about this request here.\n\n"
        "Please finish it, or say *Cancel* to stop and return to the main menu."
    )


# Static, presentation-only per-(workflow, step) hints — no live data (no
# beneficiary lists, no account numbers, no database access), matching the
# guidance layer's same rule. Describes what the current step is asking
# for, nothing more. Shared by workflows/manager.py's out-of-context
# boundary message (Task 10, Part 10) and app/conversation/guidance/responses.py's
# WORKFLOW_HELP renderer (Task 9.2/Phase 7) so the two never drift apart —
# see docs/current_architecture.md, "Response UX Architecture — Phase 8".
WORKFLOW_STEP_HINTS = {
    ("transfer", "SELECT_BENEFICIARY"): "First, choose who you'd like to pay — from your saved beneficiaries, or tell me you'd like to pay someone new.",
    ("transfer", "COLLECT_BENEFICIARY_NAME"): "Please tell me the name of the person you'd like to pay.",
    ("transfer", "COLLECT_BENEFICIARY_ACCOUNT"): "Please share the account number you're paying.",
    ("transfer", "SELECT_AMOUNT"): "Please tell me how much you'd like to send.",
    ("transfer", "COLLECT_AMOUNT"): "Please tell me how much you'd like to send.",
    ("transfer", "SELECT_SOURCE_ACCOUNT"): "Please choose which of your accounts to pay from.",
    ("transfer", "CONFIRM_TRANSFER"): "Please review the details above and reply YES to confirm or NO to cancel.",
    ("loan", "SELECT_LOAN_TYPE"): "Please choose the type of loan you'd like to apply for.",
    ("loan", "UPLOAD_LOAN_FORM"): "Please answer the next application question, or upload your loan form.",
    ("loan", "CONFIRM_LOAN"): "Please review your application above and reply YES to submit or NO to cancel.",
    ("cheque", "UPLOAD_CHEQUE"): "Please upload a clear image of the cheque so I can check the details.",
    ("cheque", "CORRECT_CHEQUE"): "Please correct the detail I flagged, either by re-uploading the cheque image or typing the correction.",
    ("kyc", "UPLOAD_KYC_FORM"): "Please upload your KYC document, or tell me what needs updating.",
    ("kyc", "CONFIRM_KYC"): "Please review your details above and reply YES to confirm or NO to cancel.",
    ("onboarding", "COLLECT_AADHAAR"): "We're currently waiting for your Aadhaar card — please upload a clear image of it.",
    ("onboarding", "COLLECT_PAN"): "We're currently waiting for your PAN card — please upload a clear image of it.",
    ("onboarding", "CONFIRM_REGISTRATION"): "Please review your details above and reply YES to confirm or NO to cancel.",
    ("onboarding", "SELECT_ACCOUNT_TYPE"): "Please choose the type of account you'd like to open.",
}

_WORKFLOW_INTRO_LABELS = {
    "transfer": "your money transfer",
    "cheque": "depositing a cheque",
    "loan": "your loan application",
    "kyc": "updating your KYC details",
    "onboarding": "your banking registration",
}


def render_workflow_step_hint(workflow_type: str, step: Optional[str]) -> Optional[str]:
    """The specific, presentation-only hint for one (workflow, step) pair,
    or None if this pair has no dedicated hint (caller should fall back to
    render_workflow_boundary())."""
    return WORKFLOW_STEP_HINTS.get((workflow_type, step or ""))


def render_workflow_boundary_with_step(workflow_type: str, step: Optional[str]) -> str:
    """A friendlier, step-aware alternative to render_workflow_boundary()
    (Task 10, Parts 9/10) — explains the CURRENT step instead of a rigid
    "I can answer questions only about this request here.", and never
    restarts the workflow or sends the customer to the main menu. Falls
    back to render_workflow_boundary() when the step has no dedicated hint."""
    hint = render_workflow_step_hint(workflow_type, step)
    if not hint:
        return render_workflow_boundary(_WORKFLOW_INTRO_LABELS.get(workflow_type, "current request"))
    intro = _WORKFLOW_INTRO_LABELS.get(workflow_type, "your current request")
    return (
        f"I'm here to help with {intro}. \U0001F60A\n\n"
        f"{hint}\n\n"
        "You can also say *Cancel* if you'd like to stop."
    )


def render_unsupported_request() -> str:
    return (
        "I'm not able to help with that specific request right now, but I "
        "can help with balances, transfers, loans, cheques, KYC and "
        "transactions — what would you like to do?"
    )

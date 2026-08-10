"""Banking Guidance response renderer — Task 9.2.

Turns a GuidanceResult (app/conversation/guidance/policy.py) into a
friendly, user-facing message plus a small set of stable action
identifiers the user can pick next. Reuses the existing centralized
response/template layer's formatting helpers
(app/conversation/responses/common.py) instead of re-implementing
currency/amount formatting — see that module's own architecture rule.

ARCHITECTURAL RULE — same boundary as policy.py, enforced by construction:
this module never imports app.database, app.workflows, or app.agent.tools.
It never calls a banking tool, queries the database, or changes workflow
state. It formats presentation text from the GuidanceResult it's handed —
nothing more. `render_guidance()` returns a RenderedGuidance whose
`actions` are only ever GuidanceAction identifiers (stable, internal,
never used as the identifier in place of the label a user sees) — see
GuidanceAction's docstring in models.py.

Clickable UI. app/services/whatsapp.py (inspected for this task) only
implements OpenWA's send-text API (`send_text_message`) — no interactive
button/list message endpoint exists anywhere in this codebase, and this
task explicitly says not to build a new OpenWA integration layer. Every
"option" below is therefore plain numbered text embedded in the message
itself (e.g. "1️⃣ ..."), which doubles as its own fallback — there is no
separate interactive payload to fall back *from*.
"""

from typing import Optional

from app.conversation.context import ConversationContext
from app.conversation.guidance.models import GuidanceAction, GuidanceResult, GuidanceType
from app.conversation.responses.common import format_currency, render_workflow_step_hint
from pydantic import BaseModel, Field


class RenderedGuidance(BaseModel):
    """One guidance turn's user-facing text plus the actions it offers.

    `primary_action` is what a bare "yes"/"go ahead" resolves to (see
    handoff.py) — the single action a user would reasonably mean without
    naming it, or None when the guidance is purely informational (no
    natural single next step, e.g. ACCOUNT_GUIDANCE).
    """

    text: str
    actions: list[GuidanceAction] = Field(default_factory=list)
    primary_action: Optional[GuidanceAction] = None


_LOAN_TYPE_LABELS = {
    "personal": "Personal loan",
    "home": "Home loan",
    "vehicle": "Vehicle loan",
    "education": "Education loan",
}


def render_guidance(guidance: GuidanceResult, context: Optional[ConversationContext] = None) -> RenderedGuidance:
    """Never raises: an unrecognized guidance_type falls back to the
    UNKNOWN renderer rather than erroring the turn."""
    if guidance.guidance_type == GuidanceType.WORKFLOW_HELP:
        return _render_workflow_help(guidance, context)
    renderer = _RENDERERS.get(guidance.guidance_type, _render_unknown)
    return renderer(guidance)


def _render_general_banking_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    return RenderedGuidance(
        text=(
            "\U0001F4A1 I can help with general banking questions — accounts, "
            "transfers, loans, cheques and KYC.\n\nWhat would you like to know?"
        )
    )


def _render_loan_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        "\U0001F4DD Loan details (like EMI, tenure and rate) depend on the loan type and "
        "the bank's current lending terms, so I can't quote exact numbers here.\n\n"
        "Would you like to see the requirements or start an application?"
    )
    return RenderedGuidance(
        text=text,
        actions=[GuidanceAction.SHOW_LOAN_REQUIREMENTS, GuidanceAction.START_LOAN_APPLICATION],
        primary_action=GuidanceAction.START_LOAN_APPLICATION,
    )


def _render_loan_eligibility_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    mentioned = []
    income = guidance.entities.get("monthly_income")
    if income is not None:
        currency = guidance.entities.get("currency", "INR")
        mentioned.append(f"• Monthly income: {format_currency(income, currency)}")
    loan_type = guidance.entities.get("loan_type")
    if loan_type:
        mentioned.append(f"• Loan type: {_LOAN_TYPE_LABELS.get(loan_type, loan_type.title())}")

    mentioned_block = ("\n\nYou mentioned:\n" + "\n".join(mentioned)) if mentioned else ""

    text = (
        "\U0001F4A1 I can help you explore a loan."
        + mentioned_block
        + "\n\nLoan eligibility can depend on factors such as income, existing "
        "obligations, credit history and the bank's lending criteria — I can't "
        "confirm approval or an exact amount here.\n\n"
        "What would you like to do?\n\n"
        "1️⃣ See requirements\n"
        "2️⃣ See documents\n"
        "3️⃣ Start application"
    )
    return RenderedGuidance(
        text=text,
        actions=[
            GuidanceAction.SHOW_LOAN_REQUIREMENTS,
            GuidanceAction.SHOW_LOAN_DOCUMENTS,
            GuidanceAction.START_LOAN_APPLICATION,
        ],
        primary_action=GuidanceAction.START_LOAN_APPLICATION,
    )


def _render_loan_document_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        "\U0001F4C4 I can help you start the application and show you the documents "
        "required at each step — the exact list depends on the loan type and "
        "your application.\n\n"
        "Would you like to:\n\n"
        "1️⃣ See requirements\n"
        "2️⃣ Start application"
    )
    return RenderedGuidance(
        text=text,
        actions=[GuidanceAction.SHOW_LOAN_REQUIREMENTS, GuidanceAction.START_LOAN_APPLICATION],
        primary_action=GuidanceAction.START_LOAN_APPLICATION,
    )


def _render_transfer_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        "\U0001F4B8 Of course — I'll guide you through it.\n\n"
        "We'll:\n"
        "1️⃣ Choose who you're paying\n"
        "2️⃣ Enter the amount\n"
        "3️⃣ Choose the account\n"
        "4️⃣ Review the details\n"
        "5️⃣ Confirm before anything is submitted\n\n"
        "What would you like to do?\n\n"
        "1. Start transfer\n"
        "2. Explain the process\n"
        "3. Cancel"
    )
    return RenderedGuidance(
        text=text,
        actions=[GuidanceAction.START_TRANSFER, GuidanceAction.SHOW_TRANSFER_INFORMATION, GuidanceAction.CANCEL],
        primary_action=GuidanceAction.START_TRANSFER,
    )


def _render_cheque_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        "\U0001F9FE I can help you deposit a cheque.\n\n"
        "You'll need a clear image of the cheque. I'll check the details and "
        "show you the request information before continuing.\n\n"
        "1. Deposit cheque\n"
        "2. Explain cheque deposit\n"
        "3. Back"
    )
    return RenderedGuidance(
        text=text,
        actions=[GuidanceAction.START_CHEQUE_DEPOSIT, GuidanceAction.SHOW_CHEQUE_INFORMATION, GuidanceAction.BACK],
        primary_action=GuidanceAction.START_CHEQUE_DEPOSIT,
    )


def _render_cheque_status_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        "\U0001F9FE Your cheque is still being processed.\n\n"
        "A pending status means the request hasn't reached a final decision yet.\n\n"
        "If you give me the cheque request ID, I can help you check its current status."
    )
    return RenderedGuidance(text=text)


def _render_kyc_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        '\U0001F4C7 KYC means "Know Your Customer".\n\n'
        "It's used to verify your identity and keep your account information "
        "up to date.\n\n"
        "I can also help you update your KYC details.\n\n"
        "1. Update KYC\n"
        "2. Ask another question"
    )
    return RenderedGuidance(
        text=text,
        actions=[GuidanceAction.START_KYC_UPDATE],
        primary_action=GuidanceAction.START_KYC_UPDATE,
    )


def _render_account_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    text = (
        "\U0001F3E6 A savings account is generally used for everyday banking and "
        "saving money.\n\n"
        "A current account is generally designed for frequent transactions, "
        "often for business use.\n\n"
        "Would you like help choosing an account?"
    )
    return RenderedGuidance(text=text)


def _render_transaction_guidance(guidance: GuidanceResult) -> RenderedGuidance:
    return RenderedGuidance(
        text=(
            "\U0001F4C4 I can show you your recent transactions, and you can ask me "
            "about spending in a specific category (like groceries or bills).\n\n"
            "Just ask, for example: \"Show my transactions\" or \"How much did I "
            "spend on groceries?\""
        )
    )


def _render_workflow_help(guidance: GuidanceResult, context: Optional[ConversationContext] = None) -> RenderedGuidance:
    if not context or not context.current_workflow:
        return RenderedGuidance(text="Let's continue from where we left off. You can also say Back or Cancel at any time.")
    hint = render_workflow_step_hint(context.current_workflow, context.current_step) or "Let's continue with your current request."
    return RenderedGuidance(
        text=f"{hint}\n\nYou can also say *Back* or *Cancel* at any time."
    )


def _render_unknown(guidance: GuidanceResult) -> RenderedGuidance:
    return RenderedGuidance(
        text=(
            "I want to make sure I understand — could you tell me a little more "
            "about what you'd like to do?"
        )
    )


_RENDERERS = {
    GuidanceType.GENERAL_BANKING_GUIDANCE: _render_general_banking_guidance,
    GuidanceType.LOAN_GUIDANCE: _render_loan_guidance,
    GuidanceType.LOAN_ELIGIBILITY_GUIDANCE: _render_loan_eligibility_guidance,
    GuidanceType.LOAN_DOCUMENT_GUIDANCE: _render_loan_document_guidance,
    GuidanceType.TRANSFER_GUIDANCE: _render_transfer_guidance,
    GuidanceType.CHEQUE_GUIDANCE: _render_cheque_guidance,
    GuidanceType.CHEQUE_STATUS_GUIDANCE: _render_cheque_status_guidance,
    GuidanceType.KYC_GUIDANCE: _render_kyc_guidance,
    GuidanceType.ACCOUNT_GUIDANCE: _render_account_guidance,
    GuidanceType.TRANSACTION_GUIDANCE: _render_transaction_guidance,
    GuidanceType.WORKFLOW_HELP: _render_workflow_help,
    GuidanceType.UNKNOWN: _render_unknown,
}


# ─── follow-up "show info" renderers (used by the handoff resolver) ─────

def render_action_info(action: GuidanceAction) -> str:
    """Short informational text for a SHOW_* action selection — e.g. the
    user replies "Tell me the requirements" after loan eligibility
    guidance. Never invents a number/policy; only ever restates the same
    non-committal factors the original guidance response already gave."""
    return _ACTION_INFO_TEXT.get(
        action,
        "I don't have more detail on that here, but I'm happy to help with something else.",
    )


_ACTION_INFO_TEXT = {
    GuidanceAction.SHOW_LOAN_REQUIREMENTS: (
        "\U0001F4CB Loan eligibility is normally considered based on things like your "
        "income, existing obligations, credit history and the bank's lending "
        "criteria. The exact requirements are confirmed during the application "
        "itself.\n\nWould you like to start the application?"
    ),
    GuidanceAction.SHOW_LOAN_DOCUMENTS: (
        "\U0001F4C4 Loan documents may include identity, address, income and "
        "employment-related documents. The exact list depends on the loan type "
        "and will be shown to you during the application.\n\n"
        "Would you like to start the application?"
    ),
    GuidanceAction.SHOW_KYC_INFORMATION: (
        '\U0001F4C7 KYC ("Know Your Customer") verifies your identity and keeps your '
        "account information up to date. Updating it usually means confirming "
        "your details or uploading a document.\n\nWould you like to update your KYC now?"
    ),
    GuidanceAction.SHOW_CHEQUE_INFORMATION: (
        "\U0001F9FE To deposit a cheque, upload a clear photo of it. I'll read the "
        "details, check them with you, and only submit the request once "
        "you're happy with them.\n\nWould you like to deposit a cheque now?"
    ),
    GuidanceAction.SHOW_TRANSFER_INFORMATION: (
        "\U0001F4B8 A transfer has a few steps: choose who you're paying, enter the "
        "amount, choose your account, review the details, then confirm. "
        "Nothing is sent until you confirm.\n\nWould you like to start a transfer now?"
    ),
}

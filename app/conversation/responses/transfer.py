"""Transfer response templates.

SENSITIVE-DATA RULE: the beneficiary-selection list
(render_beneficiary_selection()) always masks the account number via
mask_account_number() — this is the one place a beneficiary's account
number is ever displayed unprompted. render_transfer_summary()/
render_transfer_success() pass through whatever value the workflow already
collected for beneficiary_account (matching pre-Phase-4 behavior — the
customer just supplied or confirmed that exact number, so echoing it back
for their own review isn't a new exposure).

SUCCESS-CLAIM RULE: render_transfer_success() must only be called once
the transfer has actually been persisted with status INITIATED by the
workflow — the wording itself says "initiated", not "completed", since
completion is a separate bank-side event (see check_transfer_status).
"""

from typing import Optional

from app.conversation.responses.common import format_currency, mask_account_number


def render_beneficiary_selection(beneficiaries: list[dict], error: Optional[str] = None) -> str:
    prefix = f"{error}\n\n" if error else ""
    if not beneficiaries:
        return (
            prefix + "\U0001F464 Looks like you don't have any saved beneficiaries yet — "
            "let's add one. What's their full name?"
        )
    lines = [
        f"{index}. {b['beneficiary_name']} · {mask_account_number(b['account_number'])}"
        for index, b in enumerate(beneficiaries, 1)
    ]
    lines.append(f"{len(beneficiaries) + 1}. Someone new")
    return (
        prefix + "\U0001F464 Sure, who would you like to pay?\n\n"
        + "\n".join(lines)
        + f"\n\nJust reply with a number (1-{len(beneficiaries) + 1})."
    )


def render_new_beneficiary() -> str:
    return "\U0001F464 Sure, let's add them. What's their full name?"


def render_beneficiary_name_prompt() -> str:
    return "What's their full name?"


def render_beneficiary_account_prompt(beneficiary_name: str) -> str:
    return f"Thanks! And what's {beneficiary_name}'s account number?"


def render_beneficiary_saved(beneficiary_name: str) -> str:
    return f"✅ Great, I've saved {beneficiary_name} as a new beneficiary."


def render_transfer_amount_prompt(error: Optional[str] = None) -> str:
    prefix = f"{error}\n\n" if error else ""
    return (
        prefix + "\U0001F4B0 How much would you like to send?\n\n"
        "1. £25\n2. £50\n3. £100\n4. £250\n5. A different amount\n\n"
        "Pick a number or just type the amount."
    )


def render_source_account_prompt(accounts: list[dict], error: Optional[str] = None) -> str:
    lines = [
        f"{index}. {a['account_number']} · {str(a['account_type']).title()} · "
        f"balance {format_currency(a['balance'], a.get('currency', 'GBP'))}"
        for index, a in enumerate(accounts, 1)
    ]
    prefix = f"{error}\n\n" if error else ""
    return (
        prefix + "\U0001F3E6 Which account should this come from?\n\n"
        + "\n".join(lines)
        + "\n\nJust reply with a number."
    )


def render_transfer_summary(
    beneficiary_name: str,
    beneficiary_account_masked: str,
    amount_label: str,
    source_account_label: str,
) -> str:
    return (
        "\U0001F50E Almost there — take a quick look before I send this:\n\n"
        f"\U0001F464 To: {beneficiary_name} ({beneficiary_account_masked})\n"
        f"\U0001F4B0 Amount: {amount_label}\n"
        f"\U0001F3E6 From: {source_account_label}"
    )


def render_transfer_confirmation() -> str:
    return "1. Yes, send it\n2. Edit amount\n\nReply 1 or 2 — or *Back*/*Cancel* if you'd rather stop."


def render_transfer_success(
    reference: str,
    beneficiary_name: str,
    beneficiary_account_masked: str,
    amount_label: str,
    source_account_label: str,
) -> str:
    return (
        "✅ All set — your transfer is on its way!\n\n"
        f"\U0001F194 Transaction ID: {reference}\n"
        f"\U0001F464 To: {beneficiary_name} ({beneficiary_account_masked})\n"
        f"\U0001F4B0 Amount: {amount_label}\n"
        f"\U0001F3E6 From: {source_account_label}\n"
        "\U0001F4CC Status: INITIATED\n\n"
        "Your bank will process this shortly and I'll reflect the status here once it "
        f'settles. Feel free to check in anytime, e.g. "check status of {reference}" or '
        '"what was my last transfer".'
    )


def render_transfer_failed(reason: Optional[str] = None) -> str:
    detail = reason or "I couldn't initiate the transfer right now — please try again shortly."
    return f"⚠️ {detail}"


def render_transfer_cancelled() -> str:
    return "No problem — the transfer's cancelled and no money moved. Come back anytime you'd like to send some."


def render_insufficient_balance(account_label: Optional[str] = None, available_label: Optional[str] = None, amount_label: Optional[str] = None) -> str:
    if account_label and available_label and amount_label:
        return (
            f"⚠️ {account_label} only has {available_label} available, which isn't quite "
            f"enough to send {amount_label}. Want to choose a different account, or reply "
            "*Back* to change the amount?"
        )
    if account_label:
        return f"⚠️ {account_label} has a £0.00 balance, so I can't transfer from it. Could you pick a different account?"
    return (
        "⚠️ None of your accounts have enough balance for a transfer right now. "
        "Once you're topped up, just ask again and I'll get it sent."
    )


def render_invalid_amount() -> str:
    return "⚠️ I didn't quite get that — pick a number above, or type an amount like 125."


def render_invalid_account() -> str:
    return "Could you pick one of the numbered accounts above?"


def render_invalid_beneficiary_name() -> str:
    return "That doesn't look like a valid name — letters only, please."


def render_invalid_beneficiary_account() -> str:
    return "Hmm, that doesn't look like a valid account number — could you double-check it?"

"""Cheque response templates."""

from typing import Optional

FIELD_LABELS = {
    "payee": "Payee", "amount_in_figures": "Amount", "bank_name": "Bank",
    "branch": "Branch", "amount_in_words": "Amount (Words)", "numbers": "Cheque Number",
    "signatory_title": "Signatory", "date_written": "Date", "drawer_name": "Drawer",
}


def render_cheque_deposit_started() -> str:
    """The opening message when a cheque-deposit workflow is first created
    — distinct from render_cheque_upload_prompt() (also used later, e.g.
    on a re-prompt), matching pre-existing wording."""
    return "\U0001F9FE Let's get that cheque deposited! Go ahead and upload a clear photo of it whenever you're ready."


def render_cheque_upload_prompt() -> str:
    return "\U0001F9FE Whenever you're ready, upload a clear photo of the cheque and I'll take it from there."


def render_cheque_processing() -> str:
    return "One moment, I'm checking your cheque..."


def render_cheque_invalid(reason: Optional[str] = None) -> str:
    detail = f"\n\nReason: {reason}" if reason else ""
    return f"❌ Sorry, I wasn't able to process that cheque.{detail}\n\nCould you try uploading a clearer image?"


def render_cheque_not_an_image() -> str:
    return "\U0001F4F7 Could you upload the cheque as an image instead (JPG, PNG, or WEBP)?"


def render_cheque_missing_fields(fields: list[str]) -> str:
    labels = ", ".join(FIELD_LABELS.get(f, f) for f in fields)
    return (
        f"⚠️ I couldn't quite make out the following on your cheque: {labels}.\n\n"
        "Mind re-uploading a clearer image, or just typing the missing "
        "details, e.g.:\n\nPayee: John Smith\nAmount: 500.00"
    )


def render_cheque_payee_missing() -> str:
    return "⚠️ I couldn't spot a payee on the cheque."


def render_cheque_payee_mismatch(expected_name: str) -> str:
    return f"⚠️ This cheque needs to be made payable to {expected_name} for me to deposit it. Could you upload one made out to you?"


def render_cheque_correction_prompt() -> str:
    return (
        "I couldn't quite read that as cheque information. Could you either re-upload a "
        "clearer image, or reply using this format:\n\n"
        "Payee: John Smith\nAmount: 500.00"
    )


DEFAULT_VALIDATION_MESSAGE = "I’m checking the cheque details. Please provide the missing fields or upload a clearer image."
DEFAULT_PENDING_MESSAGE = "I still need a few more cheque details before I can submit your deposit."


def render_cheque_explanation(current_message: Optional[str] = None) -> str:
    base = current_message or DEFAULT_VALIDATION_MESSAGE
    return f"{base}\n\nYou can correct a field like `Payee: Your Name`, or upload the cheque again."


def render_cheque_pending_reminder(pending_message: Optional[str] = None) -> str:
    base = pending_message or DEFAULT_PENDING_MESSAGE
    return f"No problem! {base}\n\nWhenever you're ready, re-upload the cheque image or reply with the details."


def render_cheque_reupload_prompt() -> str:
    return "Please re-upload a clearer cheque image, or provide the missing details as text (e.g. `Payee: John Smith`)."


def render_cheque_summary(
    request_id: str,
    payee: str,
    amount_label: str,
    date_written: Optional[str] = None,
    drawer_name: Optional[str] = None,
    bank_name: Optional[str] = None,
) -> str:
    return (
        "✅ Nice, your cheque deposit is in!\n\n"
        f"\U0001F194 Request ID: {request_id}\n"
        f"\U0001F464 Payee: {payee}\n"
        f"\U0001F4B0 Amount: {amount_label}\n"
        f"\U0001F4C5 Date: {date_written or 'Not detected'}\n"
        f"\U0001F58A️ Drawer: {drawer_name or 'Not detected'}\n"
        f"\U0001F3E6 Bank: {bank_name or 'Not detected'}\n"
        "\U0001F4CC Status: PENDING\n\n"
        f'You can check on it anytime — just ask, e.g. "check status of {request_id}".'
    )


def render_cheque_confirmation() -> str:
    return "Does this all look right? Reply *YES* to confirm or *NO* to cancel."


def render_cheque_success(request_id: str) -> str:
    return f"✅ Cheque deposit request created!\n\n\U0001F194 Request ID: {request_id}\n\U0001F4CC Status: PENDING"


def render_cheque_failed() -> str:
    return "The cheque looked good, but something went wrong saving the request on my end. Please try again shortly."


def render_cheque_cancelled() -> str:
    return "No problem — the cheque deposit's cancelled, nothing was submitted."

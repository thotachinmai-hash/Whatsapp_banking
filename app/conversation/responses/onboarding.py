"""Onboarding/registration response templates.

SENSITIVE-DATA RULE: none of these functions accept an aadhaar_number or
pan_number value — only whether one was received/valid. The registration
summary confirms Aadhaar/PAN were *provided*, never their digits. This is
a deliberate, small behavior change from the pre-Phase-4 confirmation
screen (which echoed the digits back) — see docs/current_architecture.md.
"""

from typing import Optional


def render_onboarding_welcome() -> str:
    return (
        "\U0001F44B Welcome to Finacle Banking!\n\n"
        "You're not registered with us yet. Let's get you set up — "
        "it only takes a minute.\n\n"
        "First, please upload a clear image of your Aadhaar card. "
        "I'll pick up your name and other details from it."
    )


def render_ask_pan() -> str:
    return "Got it. Now please upload a clear image of your PAN card."


def render_document_received(document_name: str = "document") -> str:
    return f"I received your {document_name} document."


def render_document_expected_not_text(document_name: str = "document") -> str:
    """Task 10, Part 10: shown when the customer sends text (a question, a
    stray reply, anything) at a step that's waiting for a document image —
    stays in the registration context and says exactly what's needed,
    instead of trying to answer whatever the text said."""
    return (
        f"I'm still waiting on your {document_name} — "
        f"go ahead and upload a clear image whenever you're ready, or reply *Cancel* if you'd like to stop."
    )


def render_document_invalid(document_name: str = "document", reason: Optional[str] = None) -> str:
    base = f"I couldn't read a valid {document_name} from that image."
    detail = f" {reason}" if reason else ""
    return f"{base}{detail} Please upload a clearer image."


def render_document_processing(document_name: str = "document") -> str:
    return f"One moment, I'm checking your {document_name}..."


def render_profile_mismatch(fields: list[str]) -> str:
    labels = ", ".join(fields)
    return (
        f"These details don't match the information you already shared: {labels}. "
        "Please upload the correct document."
    )


def render_registration_summary(
    phone_number: str,
    full_name: str,
    date_of_birth: str = "",
    guardian_name: str = "",
    address: str = "",
) -> str:
    lines = [
        "Thanks! Take a quick look before I save these details:",
        f"Phone: {phone_number}",
        f"Full name: {full_name}",
    ]
    if date_of_birth:
        lines.append(f"Date of birth: {date_of_birth}")
    if guardian_name:
        lines.append(f"Guardian / father / spouse name: {guardian_name}")
    if address:
        lines.append(f"Address: {address}")
    lines.append("Aadhaar: Provided ✅")
    lines.append("PAN: Provided ✅")
    return "\n".join(lines)


def render_registration_confirmation() -> str:
    return "All good? Reply *YES* to save these details, or *NO* to cancel."


def render_registration_success() -> str:
    return "✅ You're all set — registration complete! Welcome aboard."


def render_registration_failed(reason: Optional[str] = None) -> str:
    detail = reason or "this Aadhaar or PAN may already be registered"
    return f"❌ Sorry, I couldn't complete your registration — {detail}. Please contact support and they'll sort it out."


def render_registration_cancelled() -> str:
    return (
        "Thank you for visiting Finacle Banking! 🙏 No problem at all — "
        "just message us again whenever you're ready to register. Happy to help!"
    )


def render_registration_declined() -> str:
    return "No problem — send any message to start registration again."


def render_account_type_prompt() -> str:
    return (
        "Which account would you like to open?\n"
        "1. Savings Account\n2. Current Account\n3. Salary Account\n\n"
        "Reply with 1, 2, or the account type."
    )


def render_account_created(account_number: str, account_type: str, currency_code: str = "GBP") -> str:
    return (
        "Account opened successfully!\n\n"
        f"Account Number: {account_number}\n"
        f"Account Type: {account_type.title()} Account\n"
        f"Balance: {currency_code} 0.00\n\n"
        "Your Relationship Manager will contact you via email. "
        "For any queries, contact finacle@infi.com."
    )


def render_account_type_invalid() -> str:
    return (
        "Please choose one of the available account types:\n"
        "1. Savings Account\n2. Current Account\n3. Salary Account"
    )


def render_account_creation_failed() -> str:
    return "We couldn't open the account right now. Please try again shortly."

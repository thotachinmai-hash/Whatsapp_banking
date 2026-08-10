"""KYC response templates.

SENSITIVE-DATA RULE: like onboarding, no function here accepts an
aadhaar_number or pan_number value — the confirmation summary states
which fields were received, never the digits.
"""

LABELS = {
    "full_name": "Full name", "date_of_birth": "Date of birth", "address": "Address",
    "aadhaar_number": "Aadhaar number", "pan_number": "PAN number",
}


def render_kyc_update_started() -> str:
    """The opening message when a KYC workflow is first created — distinct
    from render_kyc_upload_prompt() (also used later), matching
    pre-existing wording."""
    return "\U0001F4C4 Let's get your KYC updated! Whenever you're ready, upload your KYC document."


def render_kyc_upload_prompt() -> str:
    return "\U0001F4C4 Go ahead and upload your KYC document whenever you're ready."


def render_kyc_document_received() -> str:
    return "Got it, thanks — I received your KYC document."


def render_kyc_invalid() -> str:
    return "Sorry, I couldn't read that KYC document. Could you upload a clearer image or PDF?"


def render_kyc_processing() -> str:
    return "One moment, I'm checking your KYC document..."


def render_kyc_missing_fields(fields: list[str]) -> str:
    labels = ", ".join(LABELS.get(f, f) for f in fields)
    return f"I still need these details: {labels}. You can upload a clearer document, or just reply with `Field: value`."


def render_kyc_mismatch(fields: list[str]) -> str:
    """Task 10 catalog completeness — no KYC field-mismatch validation
    exists yet (unlike onboarding's Aadhaar/PAN vs typed-name check), so
    this has no caller today. Kept ready, matching onboarding's
    render_profile_mismatch(), for whenever that validation is added —
    this task does not add new validation logic."""
    labels = ", ".join(LABELS.get(f, f) for f in fields)
    return (
        f"The {labels} on your document doesn't match what's on file.\n\n"
        "You can upload the correct document, or reply *Cancel* if you'd like to stop."
    )


def render_kyc_summary(full_name: str, date_of_birth: str, address: str) -> str:
    return (
        "Take a quick look before I submit this:\n\n"
        f"Name: {full_name}\nDate of birth: {date_of_birth}\nAddress: {address}\n"
        "Aadhaar: Provided ✅\nPAN: Provided ✅"
    )


def render_kyc_confirmation() -> str:
    return "All correct? Reply *YES* to submit or *NO* to cancel."


def render_kyc_success(request_id: str) -> str:
    return (
        f"✅ Your KYC update is submitted!\n\nRequest ID: {request_id}\nStatus: PENDING\n\n"
        "Our team will verify your documents and reach out if anything else is needed."
    )


def render_kyc_failed() -> str:
    return "Sorry, something went wrong and I couldn't submit the KYC update. Please try again."


def render_kyc_cancelled() -> str:
    return "No problem — your KYC update's cancelled. Come back anytime you're ready."

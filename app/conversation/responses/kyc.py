"""KYC response templates.

SENSITIVE-DATA RULE: no function here accepts an id_number value — the
confirmation summary states that an ID number was received, never the
digits (matches app/conversation/context.py's SENSITIVE_WORKFLOW_KEYS,
which strips "id_number" before it ever reaches ConversationContext).
"""

ACCEPTED_ID_LABELS = {
    "aadhaar": "Aadhaar card",
    "pan": "PAN card",
    "passport": "Passport",
    "voter_id": "Voter ID",
    "driving_license": "Driving Licence",
}

_ACCEPTED_LIST = ", ".join(ACCEPTED_ID_LABELS.values())


def render_kyc_update_started() -> str:
    """The opening message when a KYC workflow is first created — distinct
    from render_kyc_upload_prompt() (also used later), matching
    pre-existing wording."""
    return (
        f"\U0001F4C4 Let's get your KYC updated! Please upload a clear photo "
        f"of one of these: {_ACCEPTED_LIST}."
    )


def render_kyc_upload_prompt() -> str:
    return f"\U0001F4C4 Please upload a clear photo of one of: {_ACCEPTED_LIST}."


def render_kyc_document_received() -> str:
    return "Got it, thanks — I received your KYC document."


def render_kyc_invalid() -> str:
    return f"Sorry, I couldn't read that document. Could you upload a clearer image or PDF of one of: {_ACCEPTED_LIST}?"


def render_kyc_processing() -> str:
    return "One moment, I'm checking your KYC document..."


def render_kyc_unsupported_document() -> str:
    """The upload was read fine, but it isn't one of the five accepted
    government IDs (or the model couldn't tell what it was)."""
    return (
        "That doesn't look like a supported ID document. I can only accept: "
        f"{_ACCEPTED_LIST}.\n\nPlease upload a clear photo of one of those."
    )


def render_kyc_could_not_read(id_type: str | None = None) -> str:
    """The document WAS one of the accepted types, but its ID number,
    name, or date of birth couldn't be read clearly enough to validate."""
    label = ACCEPTED_ID_LABELS.get(id_type or "", "document")
    return f"I couldn't clearly read the details on your {label}. Could you try a clearer, well-lit photo?"


def render_kyc_summary(id_type: str, full_name: str, date_of_birth: str, address: str) -> str:
    label = ACCEPTED_ID_LABELS.get(id_type, "ID document")
    address_line = f"Address: {address}\n" if address else ""
    return (
        "Take a quick look before I submit this:\n\n"
        f"Document: {label}\n"
        f"Name: {full_name}\nDate of birth: {date_of_birth}\n{address_line}"
        "ID number: Provided ✅"
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

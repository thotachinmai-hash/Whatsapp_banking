"""Loan response templates.

ELIGIBILITY RULE: no function here decides or claims eligibility.
render_loan_eligibility_guidance() states the factors eligibility depends
on and offers a next step — it never says the customer is or isn't
eligible, since that decision isn't made anywhere in this application.
"""

from typing import Optional

LOAN_LABELS = {"personal": "Personal Loan", "home": "Home Loan", "vehicle": "Vehicle Loan", "education": "Education Loan"}

FIELD_LABELS = {
    "account_number": "Account number", "applicant_name": "Applicant name",
    "monthly_income": "Monthly income", "employment_type": "Employment type",
    "requested_amount": "Requested amount", "tenure_months": "Tenure in months",
    "purpose": "Loan purpose",
}
FIELD_PROMPTS = {
    "account_number": "What is the account number this loan should be linked to?",
    "applicant_name": "What is the applicant's full name?",
    "monthly_income": "What is your monthly income?",
    "employment_type": "What is your employment type? (e.g. Salaried, Self-employed, Business)",
    "requested_amount": "How much would you like to borrow?",
    "tenure_months": "What repayment tenure would you like, in months? (e.g. 24)",
    "purpose": "What is the purpose of this loan?",
}


FIELD_ACKS = {
    "account_number": "Got it, thanks!",
    "applicant_name": "Nice to meet you!",
    "monthly_income": "Thanks for sharing that.",
    "employment_type": "Good to know.",
    "requested_amount": "Noted.",
    "tenure_months": "Perfect.",
    "purpose": "Makes sense.",
}


def render_loan_menu() -> str:
    return (
        "What kind of loan are you after?\n\n"
        "1. Personal\n2. Home\n3. Vehicle\n4. Education\n\n"
        "Just reply with a number or the name — or *Cancel* anytime if you change your mind."
    )


def render_loan_application_started() -> str:
    """The opening message when a loan workflow is first created —
    matching pre-existing wording, reusing render_loan_menu() rather than
    duplicating the type list."""
    return f"\U0001F4DD Let's get your loan application going!\n\n{render_loan_menu()}"


def render_loan_type_prompt() -> str:
    """The re-prompt shown when the customer's reply didn't match a loan
    type — distinct wording/footer from render_loan_menu() (the initial
    "loan application started" message), matching pre-Phase-4 behavior."""
    return "\U0001F4DD Sorry, I didn't catch that — which of these?\n\n1. Personal\n2. Home\n3. Vehicle\n4. Education\n\n(*Back* or *Cancel* work anytime.)"


def render_loan_type_selected(loan_type: str) -> str:
    label = LOAN_LABELS.get(loan_type, loan_type)
    return (
        f"✅ {label} it is! I'll just ask a few quick questions to get your "
        f"application together.\n\n{FIELD_PROMPTS['account_number']}"
    )


def render_loan_field_prompt(field: str, just_completed_field: Optional[str] = None) -> str:
    ack = f"{FIELD_ACKS[just_completed_field]} " if just_completed_field in FIELD_ACKS else ""
    return f"{ack}{FIELD_PROMPTS.get(field, 'And what about the next detail?')}"


def render_loan_field_invalid(field: str, reason: str) -> str:
    return f"{reason} {FIELD_PROMPTS.get(field, '')}"


def render_loan_field_explanation(field: str, definition: str, current_value: Optional[str] = None) -> str:
    value_note = f" You've already given me {current_value} for this, by the way." if current_value else " Still need this one from you."
    return f"ℹ️ Good question — {FIELD_LABELS.get(field, field).lower()} means {definition}.{value_note}"


def render_loan_summary(
    loan_type: str,
    account_number: str,
    applicant_name: str,
    monthly_income: str,
    requested_amount: str,
    tenure_months: str,
    employment_type: str,
    purpose: str,
) -> str:
    return (
        "Thanks — that's everything I need! Here's what I've got, take a quick look:\n\n"
        f"\U0001F4B3 {LOAN_LABELS.get(loan_type, loan_type)} · Account {account_number}\n"
        f"\U0001F464 {applicant_name} · {employment_type}\n"
        f"\U0001F4B0 Income {monthly_income}/mo · borrowing {requested_amount} over {tenure_months} months\n"
        f"\U0001F4DD Purpose: {purpose}"
    )


def render_loan_confirmation() -> str:
    return "Shall I go ahead and submit this? Reply *YES* to confirm or *NO* to cancel."


def render_loan_success(request_id: str) -> str:
    return (
        "✅ All done — your application is in!\n\n"
        f"\U0001F194 Loan ID: {request_id}\n"
        "\U0001F4CC Status: PENDING\n\n"
        "You can check on it anytime just by asking, e.g. \"check status of "
        f"{request_id}\". If our team needs anything else from you, they'll reach out."
    )


def render_loan_failed() -> str:
    return "Sorry, something went wrong on my end and I couldn't create the loan request. Please try again in a moment."


def render_loan_cancelled() -> str:
    return "No problem — your loan application has been cancelled. Come back anytime you'd like to apply."


def render_loan_document_unreadable() -> str:
    return "Sorry, I couldn't quite read that document. Feel free to try uploading it again, or just answer the question directly."


def render_loan_eligibility_guidance(loan_type: Optional[str] = None, monthly_income_label: Optional[str] = None) -> str:
    loan_phrase = LOAN_LABELS.get(loan_type, "a loan") if loan_type else "a loan"
    income_note = f"{monthly_income_label} monthly income may affect the loan amount and eligibility, but " if monthly_income_label else ""
    return (
        f"{income_note}I can't determine approval from income alone. Eligibility for {loan_phrase} "
        "can depend on your existing obligations, credit history and other bank requirements.\n\n"
        "Would you like me to explain the requirements, or help you start an application?"
    )

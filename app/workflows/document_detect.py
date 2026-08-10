"""Classify a document uploaded with NO active workflow — the customer
sent a cheque/KYC/loan-form image cold, without first saying "I want to
deposit a cheque". message_handler.py's no-workflow OCR prompt (see
build_document_prompt) asks the vision model to classify AND extract in
one call; this module reads that result so the already-parsed content can
feed straight into the right workflow instead of being discarded and
asked for again.
"""

from typing import Any, Optional

# Field-presence fallback, used only if the model didn't return a usable
# "document_type" (e.g. an older/other model response shape) — same idea
# as _missing_fields()/_extract() elsewhere in the workflow processors:
# never trust the model's own label alone when the raw fields disagree.
_CHEQUE_FIELDS = ("payee", "amount_in_figures", "numbers", "bank_name", "drawer_name")
_KYC_FIELDS = ("aadhaar_number", "pan_number", "guardian_name")
_LOAN_FIELDS = ("applicant_name", "monthly_income", "requested_amount", "tenure_months", "employment_type")

_VALID_TYPES = {"cheque", "kyc", "loan_form"}


def _present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "not detected", "none", "n/a")


def _count_present(content: dict, fields: tuple[str, ...]) -> int:
    return sum(1 for field in fields if _present(content.get(field)))


def detect_workflow_type(content: dict) -> Optional[str]:
    """Return "cheque", "kyc", "loan_form", or None (not confident enough).

    Trusts the model's own "document_type" field first; falls back to
    counting which field group actually has real values if that's missing
    or says "other", so a model that ignored the classification
    instruction but still extracted real cheque fields is still caught.
    """
    if not isinstance(content, dict):
        return None

    declared = str(content.get("document_type", "")).strip().lower()
    if declared in _VALID_TYPES:
        return declared

    scores = {
        "cheque": _count_present(content, _CHEQUE_FIELDS),
        "kyc": _count_present(content, _KYC_FIELDS),
        "loan_form": _count_present(content, _LOAN_FIELDS),
    }
    best_type, best_score = max(scores.items(), key=lambda pair: pair[1])
    # Require at least 2 real signals — one stray field (e.g. a name that
    # happens to also be a loan applicant field) isn't enough to guess a
    # whole workflow from.
    if best_score >= 2:
        return best_type
    return None

"""Deterministic intent rules — checked before any LLM call.

Mirrors the project's existing style of keeping transactional flows off
the LLM entirely (see app/workflows/manager.py's own keyword-based
start_requested()). Every function here returns an IntentResult or None;
None means "this layer has no opinion, try the next one" — see
classifier.py for the ordering.
"""

import re
from typing import Optional

from app.agent.tools import CATEGORY_SYNONYMS
from app.conversation.context import ConversationContext
from app.conversation.intent.models import IntentResult
from app.services.registration_gate import GREETING_KEYWORDS

# ─── shared helpers ─────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase and strip everything but letters/digits/spaces — used for
    exact-phrase membership checks (e.g. "cancel" vs "cancel it, please!")."""
    return re.sub(r"[^a-z0-9 ]", "", text.strip().lower()).strip()


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    normalized = stripped.lower()
    return normalized.startswith((
        "what", "how", "why", "can i", "could i", "is it", "does",
        "do i", "am i", "when", "where", "will i",
    ))


_CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "₹": "INR", "€": "EUR"}
_CURRENCY_WORDS = {
    "gbp": "GBP", "pounds": "GBP", "pound": "GBP",
    "usd": "USD", "dollars": "USD", "dollar": "USD",
    "inr": "INR", "rupees": "INR", "rupee": "INR", "rs": "INR", "rs.": "INR",
    "eur": "EUR", "euros": "EUR", "euro": "EUR",
}


def _to_number(raw: str):
    value = float(raw.replace(",", ""))
    return int(value) if value.is_integer() else value


def _extract_amount_and_currency(text: str) -> dict:
    """Only returns amount/currency when both are unambiguously present —
    never guesses a currency for a bare number."""
    match = re.search(r"([£$₹€])\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)
    if match:
        symbol, amount = match.groups()
        return {"amount": _to_number(amount), "currency": _CURRENCY_SYMBOLS[symbol]}
    match = re.search(
        r"\b([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(gbp|usd|inr|eur|pounds?|dollars?|rupees?|rs\.?|euros?)\b",
        text,
        re.I,
    )
    if match:
        amount, currency_word = match.groups()
        return {
            "amount": _to_number(amount),
            "currency": _CURRENCY_WORDS[currency_word.lower().rstrip(".")],
        }
    return {}


def _extract_beneficiary_name(text: str) -> Optional[str]:
    match = re.search(r"\bto\s+([A-Za-z][A-Za-z .'-]{1,40})\s*$", text.strip().rstrip(".!?"))
    if match:
        return match.group(1).strip().title()
    return None


def _extract_loan_type(text: str) -> Optional[str]:
    normalized = text.lower()
    for candidate in ("personal", "home", "vehicle", "education"):
        if candidate in normalized:
            return candidate
    return None


def _extract_spend_category(text: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z ]", " ", text.lower())
    for word in normalized.split():
        if word in CATEGORY_SYNONYMS:
            return CATEGORY_SYNONYMS[word]
    return None


_CHQ_ID_PATTERN = re.compile(r"\bCHQ-[A-Z0-9]{3,}\b", re.I)
_TRF_ID_PATTERN = re.compile(r"\bTRF-[A-Z0-9]{3,}\b", re.I)


# ─── 0. prompt-injection / role-override detection ─────────────────────

_INJECTION_PATTERN = re.compile(
    r"\bignore\s+(all\s+|your\s+|any\s+|the\s+|previous\s+)*instructions\b"
    r"|\bdisregard\s+(all\s+|your\s+|any\s+|the\s+|previous\s+)*instructions\b"
    r"|\bforget\s+(all\s+|your\s+|any\s+|the\s+|previous\s+)*instructions\b"
    r"|\byou\s+are\s+now\b"
    r"|\bact\s+as\s+(a|an)\b"
    r"|\bsystem\s+prompt\b"
    r"|\bnew\s+instructions\b",
    re.I,
)


def looks_like_injection(text: str) -> bool:
    """User text is untrusted input, never a new instruction to the
    classifier — see the module docstring in classifier.py."""
    return bool(_INJECTION_PATTERN.search(text))


# ─── A. navigation ───────────────────────────────────────────────────

_CANCEL_WORDS = {"cancel", "cancel it", "stop", "stop it", "quit", "exit", "end", "never mind", "nevermind", "forget it"}
_BACK_WORDS = {"back", "go back", "previous", "previous step", "b"}
_MENU_WORDS = {"menu", "main menu", "show menu", "take me to the main menu", "home"}
_REPEAT_WORDS = {"repeat", "repeat that", "say again", "say that again", "what did you say"}
_RESTART_WORDS = {"start over", "start again", "restart", "begin again"}
_HELP_WORDS = {
    "help", "can you help", "can you help me", "assist me",
    "what should i do", "what should i do?", "what do i do", "what next",
}


def classify_hard_navigation(text: str) -> Optional[IntentResult]:
    """Context-independent control commands — these mean the same thing
    no matter what workflow (if any) is active, matching how cancel/back
    already behave identically across every workflow today."""
    normalized = _normalize(text)
    if not normalized:
        return None
    if normalized in _CANCEL_WORDS or normalized.startswith(("cancel ", "stop ")):
        return IntentResult(intent="cancel", confidence=0.99, method="rule")
    if normalized in _BACK_WORDS:
        return IntentResult(intent="back", confidence=0.98, method="rule")
    if normalized in _RESTART_WORDS:
        return IntentResult(intent="start_over", confidence=0.95, method="rule")
    if normalized in _MENU_WORDS:
        return IntentResult(intent="main_menu", confidence=0.95, method="rule")
    if normalized in _REPEAT_WORDS:
        return IntentResult(intent="repeat", confidence=0.9, method="rule")
    if normalized in GREETING_KEYWORDS:
        return IntentResult(intent="greeting", confidence=0.97, method="rule")
    return None


def classify_soft_navigation(text: str) -> Optional[IntentResult]:
    """Generic 'help'-style phrases with no active workflow to give them a
    more precise meaning. Checked only after workflow-context resolution
    has had a chance to reinterpret them as workflow_help."""
    normalized = _normalize(text)
    if normalized in _HELP_WORDS or "can you help" in normalized:
        return IntentResult(intent="help", confidence=0.85, method="rule")
    return None


# ─── E. workflow conversation (context-aware) ──────────────────────────

_CONFIRM_ANSWERS = {"yes", "y", "confirm", "no", "n"}
_EXPLANATION_TRIGGERS = ("why",)
_HELP_TRIGGERS = ("what should i do", "what do i do", "what next", "help")
_CLARIFICATION_TRIGGERS = (
    "is this correct", "is that correct", "is my name correct", "what name",
    "what do you have on record", "correct?", "is this right", "is that right",
)
_CORRECTION_TRIGGERS = (
    "i can't", "i cant", "i can not", "that's wrong", "thats wrong",
    "that is wrong", "wrong", "mistake", "typo", "not correct",
)
_STATUS_TRIGGERS = ("has it completed", "what happened", "did it work", "is it done")


def _looks_like_bare_name(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 60:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", stripped):
        return False
    return len(stripped.split()) <= 4


_BENEFICIARY_SELECTION_STEPS = {"SELECT_BENEFICIARY", "COLLECT_BENEFICIARY_NAME"}


def classify_workflow_conversation(text: str, context: ConversationContext) -> Optional[IntentResult]:
    """Only meaningful with an active workflow — the active step is what
    turns an ambiguous "yes"/"why"/"what should I do" into a precise
    conversational intent. See docs/current_architecture.md for the
    worked examples this implements."""
    if not context or not context.current_workflow:
        return None

    normalized = text.strip().lower()
    step = (context.current_step or "").upper()

    if step.startswith("CONFIRM") and _normalize(text) in _CONFIRM_ANSWERS:
        return IntentResult(
            intent="workflow_confirmation",
            confidence=0.95,
            entities={"answer": _normalize(text)},
            method="context",
        )

    if any(normalized.startswith(t) or f" {t} " in f" {normalized} " for t in _EXPLANATION_TRIGGERS):
        return IntentResult(intent="workflow_explanation", confidence=0.88, method="context")

    if any(t in normalized for t in _HELP_TRIGGERS):
        return IntentResult(intent="workflow_help", confidence=0.9, method="context")

    if any(t in normalized for t in _CLARIFICATION_TRIGGERS):
        return IntentResult(intent="workflow_clarification", confidence=0.85, method="context")

    if any(t in normalized for t in _CORRECTION_TRIGGERS):
        return IntentResult(intent="workflow_correction", confidence=0.8, method="context")

    if any(t in normalized for t in _STATUS_TRIGGERS):
        return IntentResult(intent="workflow_status", confidence=0.8, method="context")

    if step in _BENEFICIARY_SELECTION_STEPS and _looks_like_bare_name(text):
        return IntentResult(
            intent="workflow_clarification",
            confidence=0.75,
            entities={"beneficiary_name": text.strip().title()},
            method="context",
        )

    return None


# ─── F. status / lookup requests ───────────────────────────────────────

def classify_status_request(text: str) -> Optional[IntentResult]:
    normalized = text.lower()

    chq_match = _CHQ_ID_PATTERN.search(text)
    if chq_match or ("cheque" in normalized and "status" in normalized) or (
        "cheque" in normalized and normalized.strip().startswith("check")
    ):
        entities = {"cheque_request_id": chq_match.group(0).upper()} if chq_match else {}
        # Deliberately the same intent name as the category-B request —
        # see docs/current_architecture.md for why "cheque_status" (the
        # category-F spelling) was not implemented as a separate value.
        return IntentResult(intent="cheque_status_request", confidence=0.92, entities=entities, method="rule")

    trf_match = _TRF_ID_PATTERN.search(text)
    if trf_match or ("transfer" in normalized and any(k in normalized for k in ("status", "completed", "has my transfer"))):
        entities = {"transfer_reference": trf_match.group(0).upper()} if trf_match else {}
        return IntentResult(intent="transfer_status", confidence=0.92, entities=entities, method="rule")

    if "loan" in normalized and any(k in normalized for k in ("status", "approved")):
        return IntentResult(intent="loan_status", confidence=0.88, method="rule")

    if "kyc" in normalized and any(k in normalized for k in ("status", "approved")):
        return IntentResult(intent="kyc_status", confidence=0.88, method="rule")

    if "account" in normalized and any(k in normalized for k in ("information", "details", "info")) and "balance" not in normalized:
        return IntentResult(intent="account_information", confidence=0.7, method="rule")

    return None


# ─── D. personalized guidance ──────────────────────────────────────────

_INCOME_PATTERN = re.compile(r"\bi\s+earn\b|\bmy\s+income\b|\bmy\s+salary\b|\bmonthly\s+income\b", re.I)
_AFFORD_PATTERN = re.compile(r"\bcan\s+i\s+afford\b|\bcan\s+i\s+get\s+a\s+loan\b|\beligib", re.I)
_SPEND_PATTERN = re.compile(
    r"\bhow\s+much\s+did\s+i\s+spend\b|\bwhat\s+did\s+i\s+spend\b|\bwhere\s+am\s+i\s+spending\b"
    r"|\bspending\s+(on|breakdown)\b|\bwhere.*money\s+going\b"
    # Broader real-world phrasings (e.g. "What my spend analysis of this
    # month", "I want to know abt my spend analysis") — any mention of
    # "spend"/"spending"/"expense"/"expenditure" alongside "my"/"i" as a
    # personal-finance reference, not just the two exact question templates
    # above. See docs/current_architecture.md, "Response UX Architecture —
    # Phase 8" for why the narrower version wasn't enough.
    r"|\bmy\s+spend(ing)?\b|\bspend(ing)?\s+analysis\b|\bspending\s+(pattern|habit)"
    r"|\bmy\s+expense|\bmy\s+expenditure",
    re.I,
)
_ADVICE_PHRASES = ("should i save", "should i invest", "financial advice", "money advice", "budget advice")


def classify_guidance(text: str) -> Optional[IntentResult]:
    normalized = text.lower()

    if _SPEND_PATTERN.search(normalized):
        entities = {}
        category = _extract_spend_category(text)
        if category:
            entities["category"] = category
        return IntentResult(intent="transaction_insight_question", confidence=0.9, entities=entities, method="rule")

    if _INCOME_PATTERN.search(normalized) and "loan" in normalized:
        entities = _extract_income_entities(text)
        return IntentResult(intent="loan_eligibility_question", confidence=0.92, entities=entities, method="rule")

    if _AFFORD_PATTERN.search(normalized):
        entities = _extract_income_entities(text)
        return IntentResult(intent="loan_eligibility_question", confidence=0.88, entities=entities, method="rule")

    if any(p in normalized for p in _ADVICE_PHRASES):
        return IntentResult(intent="financial_guidance", confidence=0.75, method="rule")

    return None


def _extract_income_entities(text: str) -> dict:
    entities: dict = {}
    amount_currency = _extract_amount_and_currency(text)
    if amount_currency:
        entities["monthly_income"] = amount_currency["amount"]
        entities["currency"] = amount_currency["currency"]
    loan_type = _extract_loan_type(text)
    if loan_type:
        entities["loan_type"] = loan_type
    return entities


# ─── B. direct banking workflow requests ───────────────────────────────

# Hedging language ("maybe", "I think", ...) means the customer hasn't
# committed to the request yet — a workflow-request match under one of
# these must not be treated as high-confidence, so a future confidence
# policy (Phase 3's router) won't auto-start a workflow from it. Added
# specifically to support that policy; the rest of Phase 2's rules are
# otherwise unchanged. See docs/current_architecture.md, "Intent-Based
# Routing — Phase 3" → Confidence Policy.
_HEDGE_WORDS = (
    "maybe", "perhaps", "possibly", "might", "may want",
    "not sure", "i think", "i guess", "thinking about", "considering",
)
_HEDGED_CONFIDENCE_CAP = 0.7


def _dampen_for_hedging(confidence: float, normalized_text: str) -> float:
    if any(h in normalized_text for h in _HEDGE_WORDS):
        return min(confidence, _HEDGED_CONFIDENCE_CAP)
    return confidence


def classify_workflow_request(text: str) -> Optional[IntentResult]:
    normalized = text.lower()

    if any(k in normalized for k in ("register", "sign up", "open an account", "create an account")):
        confidence = _dampen_for_hedging(0.85, normalized)
        return IntentResult(intent="registration_request", confidence=confidence, method="rule")

    amount_currency = _extract_amount_and_currency(text)
    beneficiary = _extract_beneficiary_name(text)
    looks_like_transfer = (
        "transfer" in normalized
        or ("send" in normalized and (amount_currency or "money" in normalized or beneficiary))
        or ("pay" in normalized and (" to " in normalized or beneficiary))
    )
    # Every branch below requires "not a question" — matching
    # kyc_update_request's existing guard (further down). Without it, a
    # question like "What's the loan interest amount charged" or "What's
    # the transfer limit?" was being read as a request to actually START
    # the workflow (high confidence, so the router would begin a real
    # loan application / transfer from a plain question). See
    # docs/current_architecture.md for the live-testing session that
    # surfaced this.
    if looks_like_transfer and not _is_question(text):
        entities = dict(amount_currency)
        if beneficiary:
            entities["beneficiary_name"] = beneficiary
        confidence = _dampen_for_hedging(0.9, normalized)
        return IntentResult(intent="transfer_request", confidence=confidence, entities=entities, method="rule")

    if "balance" in normalized or "how much money do i have" in normalized:
        return IntentResult(intent="balance_request", confidence=0.92, method="rule")

    if any(k in normalized for k in ("transaction", "transactions", "statement")):
        return IntentResult(intent="transaction_request", confidence=0.88, method="rule")

    if not _is_question(text) and "deposit" in normalized and ("cheque" in normalized or "check" in normalized):
        confidence = _dampen_for_hedging(0.9, normalized)
        return IntentResult(intent="cheque_deposit_request", confidence=confidence, method="rule")

    if not _is_question(text) and "loan" in normalized:
        entities = {}
        loan_type = _extract_loan_type(text)
        if loan_type:
            entities["loan_type"] = loan_type
        confidence = _dampen_for_hedging(0.85, normalized)
        return IntentResult(intent="loan_application_request", confidence=confidence, entities=entities, method="rule")

    if not _is_question(text) and any(k in normalized for k in ("kyc", "update my address", "change my address", "update my details")):
        confidence = _dampen_for_hedging(0.85, normalized)
        return IntentResult(intent="kyc_update_request", confidence=confidence, method="rule")

    return None


# ─── C. banking questions / knowledge ───────────────────────────────────

_QUESTION_DOMAIN_RULES = (
    (("kyc",), "kyc_question"),
    (("address",), "kyc_question"),
    (("cheque", "clear", "clearance"), "cheque_question"),
    (("transfer", "wire"), "transfer_question"),
    (("emi", "loan"), "loan_question"),
    (("savings account", "current account", "account"), "account_question"),
)

_GENERIC_BANKING_KEYWORDS = {
    "bank", "banking", "interest", "otp", "branch", "atm", "card",
    "overdraft", "withdraw", "withdrawal", "credit", "debit",
}


def classify_banking_question(text: str) -> Optional[IntentResult]:
    if not _is_question(text):
        return None
    normalized = text.lower()
    for keywords, intent in _QUESTION_DOMAIN_RULES:
        if any(k in normalized for k in keywords):
            return IntentResult(intent=intent, confidence=0.8, method="rule")
    if any(k in normalized for k in _GENERIC_BANKING_KEYWORDS):
        return IntentResult(intent="banking_question", confidence=0.7, method="rule")
    return None


# ─── G. out of scope ────────────────────────────────────────────────────

_DENY_REGEX = re.compile(
    r"\bwrite\s+(me\s+)?(a\s+|an\s+|some\s+)?(python\s+|javascript\s+)?(code|program|poem|story|song)\b"
    r"|\btell\s+me\s+a\s+joke\b"
    r"|\bcapital\s+of\b"
    r"|\bwho\s+won\b",
    re.I,
)

# Same domain-signal vocabulary the "generic banking question" fallback
# above uses, widened slightly for the out-of-scope gate.
BANKING_DOMAIN_KEYWORDS = {
    "bank", "account", "balance", "transfer", "loan", "cheque", "kyc",
    "transaction", "transactions", "deposit", "aadhaar", "aadhar", "pan",
    "register", "registration", "emi", "otp", "interest", "saving",
    "savings", "income", "salary", "spend", "spent", "spending", "money",
    "payment", "pay", "send", "beneficiary", "statement", "credit", "debit",
    "branch", "atm", "card", "overdraft", "withdraw", "withdrawal",
}


def _has_banking_keyword(normalized: str) -> bool:
    return any(k in normalized for k in BANKING_DOMAIN_KEYWORDS)


def classify_out_of_scope(text: str) -> Optional[IntentResult]:
    normalized = text.lower()
    if _DENY_REGEX.search(normalized):
        return IntentResult(intent="out_of_scope", confidence=0.95, method="rule")
    if not _has_banking_keyword(normalized):
        return IntentResult(intent="out_of_scope", confidence=0.85, method="rule")
    return None

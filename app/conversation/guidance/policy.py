"""Banking Guidance Policy — Task 9.1.

Converts an already-classified IntentResult (app/conversation/intent) +
ConversationContext into a GuidanceResult: a decision about *how* the
assistant should help — explain, guide, offer next actions, ask for
clarification, or redirect to the real workflow/status path.

ARCHITECTURAL RULE — enforced by construction, not just by convention:
this module never imports app.database, app.workflows, app.agent.tools,
or any Postgres/Redis client. It never mutates ConversationContext or
starts/advances a workflow. `build_guidance()` is a pure function: text +
IntentResult + ConversationContext in, GuidanceResult (or None) out.

Specifically, this layer NEVER:
  - executes a banking operation, calls a database function, or calls a
    banking tool (it has no import path to any of them)
  - creates or modifies workflow state
  - approves a loan, calculates real eligibility, or invents an interest
    rate / loan limit / fee / bank policy / approval decision — the
    `entities` on every GuidanceResult are always either a verbatim
    pass-through of `IntentResult.entities` (the intent classifier's own,
    already-tested extraction) or a value extracted by this module's own
    narrow, literal, "is this exact number/word in the text" helpers
    below. Nothing here is ever computed from those values — no interest
    math, no eligibility scoring, no threshold comparison.

Deliberately NOT wired into app/conversation/router.py or
ConversationManager by this task — it is a new, isolated, addressable
decision layer with its own tests. A future task can decide how (and
whether) a router/response-template layer consumes GuidanceResult; this
task only builds and tests the policy itself, per its explicit scope.

Guidance vs. action. The existing intent classifier (untouched by this
task) already separates most information/advice intents (banking
questions, guidance questions) from action intents (workflow-request
intents) by intent name alone — see `_ACTION_ONLY_INTENTS` below. Two
situations need a little more than the bare intent name, and this module
resolves them using ONLY the same untrusted user text the classifier
itself already had access to (never a new data source):
  1. `transfer_request` with no concrete details ("I don't know how to
     transfer money") vs. a real transfer ("Transfer £500 to Priya") —
     distinguished by whether the classifier's own entity extraction
     found an amount/beneficiary, plus a small fixed set of "I'm
     confused" phrases. A transfer_request WITH details, or without a
     confusion phrase, is left alone (returns None) — the existing
     workflow-start path still owns it.
  2. `cheque_status_request` with a concrete reference id ("Check
     CHQ-123") is a real lookup — left alone. Without one ("My cheque is
     still pending" — vague enough that the classifier itself falls
     through to "unknown") it becomes CHEQUE_STATUS_GUIDANCE,
     REDIRECT_TO_WORKFLOW: point the customer at the real status-check
     path rather than this layer guessing an answer.
"""

import re
from typing import Optional

from app.conversation.context import ConversationContext
from app.conversation.guidance.models import GuidanceResult, GuidanceType, ResponseMode, SuggestedAction
from app.conversation.intent.models import IntentResult

# Direct action requests — always continue to the existing deterministic
# workflow-start path unchanged. Never became guidance.
# loan_application_request and cheque_deposit_request are handled by their
# own functions below (not this blanket set) — Task 9.2 found that the
# classifier's own keyword rules for both (unchanged, out of scope to fix)
# match on a loose "loan"/"deposit"+"cheque" substring with no
# question-guard, so a genuine question like "What documents do I need for
# a personal loan?" or "How do I deposit a cheque?" is classified as the
# ACTION intent, not a question intent. See _guidance_for_loan_application_request
# / _guidance_for_cheque_deposit_request for the narrow, text-only carve-out.
_ACTION_ONLY_INTENTS = {
    "registration_request", "balance_request", "transaction_request", "kyc_update_request",
}

# Navigation and out-of-scope must never become a banking guidance
# response — mirrors Phase 3's "out_of_scope never reaches the banking
# LLM" rule; guidance is a banking-domain concept only.
_NEVER_GUIDANCE_INTENTS = {
    "out_of_scope", "greeting", "help", "back", "cancel", "main_menu",
    "repeat", "start_over",
}

_CONFUSION_PHRASES = (
    "don't know how", "dont know how", "not sure how", "no idea how",
    "how do i", "how does", "confused about", "don't understand",
    "dont understand",
)

_CHEQUE_STATUS_HINTS = ("pending", "still", "cleared", "clearing", "status", "processing", "rejected")
_DOCUMENT_HINTS = ("document", "documents", "paperwork", "papers")

# Narrow, literal fallback for "I earn 50000 a month" style phrasing where
# no currency symbol/word follows the number, so the classifier's own
# _extract_amount_and_currency() (which requires one) finds nothing. Only
# ever fires immediately after earn/income/salary, and only ever returns a
# number that is literally present in the text — never a computed value.
_INCOME_NUMBER_PATTERN = re.compile(
    r"\b(?:earn|income|salary)\D{0,15}?([0-9][0-9,]*(?:\.[0-9]{1,2})?)\b", re.I
)


def _extract_monthly_income_fallback(text: str) -> Optional[float]:
    match = _INCOME_NUMBER_PATTERN.search(text or "")
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return int(value) if value.is_integer() else value


def _looks_confused(normalized_text: str) -> bool:
    return any(p in normalized_text for p in _CONFUSION_PHRASES)


_GUIDANCE_MAP: dict[str, tuple] = {
    "financial_guidance": (GuidanceType.GENERAL_BANKING_GUIDANCE, ResponseMode.GUIDE, []),
    "banking_question": (GuidanceType.GENERAL_BANKING_GUIDANCE, ResponseMode.EXPLAIN, []),
    "loan_eligibility_question": (
        GuidanceType.LOAN_ELIGIBILITY_GUIDANCE,
        ResponseMode.OFFER_ACTIONS,
        [SuggestedAction.CHECK_ELIGIBILITY, SuggestedAction.START_APPLICATION, SuggestedAction.REQUIRED_DOCUMENTS],
    ),
    "loan_question": (GuidanceType.LOAN_GUIDANCE, ResponseMode.EXPLAIN, [SuggestedAction.REQUIRED_DOCUMENTS]),
    "transfer_question": (GuidanceType.TRANSFER_GUIDANCE, ResponseMode.EXPLAIN, [SuggestedAction.START_TRANSFER]),
    "cheque_question": (GuidanceType.CHEQUE_GUIDANCE, ResponseMode.EXPLAIN, [SuggestedAction.DEPOSIT_CHEQUE]),
    "kyc_question": (GuidanceType.KYC_GUIDANCE, ResponseMode.EXPLAIN, [SuggestedAction.START_KYC_UPDATE]),
    "account_question": (GuidanceType.ACCOUNT_GUIDANCE, ResponseMode.EXPLAIN, [SuggestedAction.VIEW_ACCOUNT_SUMMARY]),
    "transaction_insight_question": (
        GuidanceType.TRANSACTION_GUIDANCE,
        ResponseMode.OFFER_ACTIONS,
        [SuggestedAction.VIEW_TRANSACTIONS],
    ),
    "workflow_help": (GuidanceType.WORKFLOW_HELP, ResponseMode.GUIDE, [SuggestedAction.CONTINUE_WORKFLOW]),
}


def build_guidance(
    text: str,
    intent_result: IntentResult,
    context: Optional[ConversationContext] = None,
) -> Optional[GuidanceResult]:
    """Return a GuidanceResult, or None when this turn is not a guidance
    situation — callers must treat None as "defer to the existing
    routing/workflow path, unchanged."

    Never raises: this is a pure, deterministic mapping over already-safe
    inputs (IntentResult.entities has already been validated/limited by
    the classifier); any unexpected input simply falls through to None.
    """
    normalized = (text or "").lower()
    intent = intent_result.intent

    if intent in _NEVER_GUIDANCE_INTENTS or intent in _ACTION_ONLY_INTENTS:
        return None

    if intent == "transfer_request":
        return _guidance_for_transfer_request(normalized, intent_result)

    if intent == "cheque_status_request":
        return _guidance_for_cheque_status_request(intent_result)

    if intent == "loan_eligibility_question":
        return _guidance_for_loan_eligibility(text, normalized, intent_result)

    if intent == "loan_application_request":
        return _guidance_for_loan_application_request(normalized, intent_result)

    if intent == "cheque_deposit_request":
        return _guidance_for_cheque_deposit_request(normalized, intent_result)

    if intent == "unknown":
        return _guidance_for_unknown(normalized, intent_result)

    mapped = _GUIDANCE_MAP.get(intent)
    if not mapped:
        return None

    guidance_type, response_mode, suggested_actions = mapped
    if guidance_type == GuidanceType.LOAN_GUIDANCE and any(h in normalized for h in _DOCUMENT_HINTS):
        guidance_type = GuidanceType.LOAN_DOCUMENT_GUIDANCE

    return GuidanceResult(
        guidance_type=guidance_type,
        response_mode=response_mode,
        suggested_actions=list(suggested_actions),
        entities=dict(intent_result.entities),
        confidence=intent_result.confidence,
    )


def _guidance_for_transfer_request(normalized: str, intent_result: IntentResult) -> Optional[GuidanceResult]:
    # Deliberately only "amount" counts as a concrete detail here, not
    # beneficiary_name: the classifier's own beneficiary-name extraction
    # (rules._extract_beneficiary_name, unchanged by this task) matches
    # "to <the rest of the sentence>" and can false-positive on confused
    # phrasing like "...how to transfer money" (extracting "Transfer
    # Money" as a fake beneficiary). Amount has no such false-positive
    # mode — it only ever matches an explicit number+currency.
    has_details = bool(intent_result.entities.get("amount"))
    if has_details or not _looks_confused(normalized):
        return None  # a real transfer request — the existing workflow handles it
    return GuidanceResult(
        guidance_type=GuidanceType.TRANSFER_GUIDANCE,
        response_mode=ResponseMode.OFFER_ACTIONS,
        suggested_actions=[SuggestedAction.START_TRANSFER, SuggestedAction.VIEW_BENEFICIARIES],
        entities=dict(intent_result.entities),
        confidence=intent_result.confidence,
    )


def _guidance_for_cheque_status_request(intent_result: IntentResult) -> Optional[GuidanceResult]:
    if intent_result.entities.get("cheque_request_id"):
        return None  # a concrete lookup — the existing status/tool path handles it
    return GuidanceResult(
        guidance_type=GuidanceType.CHEQUE_STATUS_GUIDANCE,
        response_mode=ResponseMode.REDIRECT_TO_WORKFLOW,
        suggested_actions=[SuggestedAction.CHECK_CHEQUE_STATUS],
        entities=dict(intent_result.entities),
        confidence=intent_result.confidence,
    )


def _guidance_for_loan_application_request(normalized: str, intent_result: IntentResult) -> Optional[GuidanceResult]:
    # The classifier's own loan_application_request rule (rules.py,
    # unchanged) matches on a bare "loan" substring with no question-guard
    # (unlike its kyc_update_request sibling, which explicitly excludes
    # questions) — so "What documents do I need for a personal loan?" is
    # classified as the ACTION intent, not loan_question. Only override
    # when the wording is clearly asking about documents/requirements
    # rather than asking to apply.
    if any(h in normalized for h in _DOCUMENT_HINTS) or "requirement" in normalized:
        return GuidanceResult(
            guidance_type=GuidanceType.LOAN_DOCUMENT_GUIDANCE,
            response_mode=ResponseMode.EXPLAIN,
            suggested_actions=[SuggestedAction.REQUIRED_DOCUMENTS, SuggestedAction.START_APPLICATION],
            entities=dict(intent_result.entities),
            confidence=intent_result.confidence,
        )
    return None  # a real application request — the existing workflow handles it


def _guidance_for_cheque_deposit_request(normalized: str, intent_result: IntentResult) -> Optional[GuidanceResult]:
    # Same shape of pre-existing classifier gap as transfer_request above:
    # "deposit" + "cheque" matches regardless of whether the user is
    # asking how to do it or asking to do it right now.
    if _looks_confused(normalized):
        return GuidanceResult(
            guidance_type=GuidanceType.CHEQUE_GUIDANCE,
            response_mode=ResponseMode.OFFER_ACTIONS,
            suggested_actions=[SuggestedAction.DEPOSIT_CHEQUE],
            entities=dict(intent_result.entities),
            confidence=intent_result.confidence,
        )
    return None  # a real "deposit my cheque" request — the existing workflow handles it


def _guidance_for_loan_eligibility(text: str, normalized: str, intent_result: IntentResult) -> GuidanceResult:
    entities = dict(intent_result.entities)
    if "monthly_income" not in entities:
        fallback_income = _extract_monthly_income_fallback(text)
        if fallback_income is not None:
            entities["monthly_income"] = fallback_income
    return GuidanceResult(
        guidance_type=GuidanceType.LOAN_ELIGIBILITY_GUIDANCE,
        response_mode=ResponseMode.OFFER_ACTIONS,
        suggested_actions=[
            SuggestedAction.CHECK_ELIGIBILITY,
            SuggestedAction.START_APPLICATION,
            SuggestedAction.REQUIRED_DOCUMENTS,
        ],
        entities=entities,
        confidence=intent_result.confidence,
    )


def _guidance_for_unknown(normalized: str, intent_result: IntentResult) -> GuidanceResult:
    # The intent classifier itself sometimes falls through to "unknown"
    # for vague-but-clearly-banking phrasing its deterministic rules don't
    # have a specific home for (e.g. "My cheque is still pending" has no
    # question mark and no reference id). Only resolve the narrow case
    # this module has a confident, non-inventive answer for; everything
    # else stays a plain clarification request.
    if "cheque" in normalized and any(h in normalized for h in _CHEQUE_STATUS_HINTS):
        return GuidanceResult(
            guidance_type=GuidanceType.CHEQUE_STATUS_GUIDANCE,
            response_mode=ResponseMode.REDIRECT_TO_WORKFLOW,
            suggested_actions=[SuggestedAction.CHECK_CHEQUE_STATUS],
            entities=dict(intent_result.entities),
            confidence=max(intent_result.confidence, 0.5),
        )
    return GuidanceResult(
        guidance_type=GuidanceType.UNKNOWN,
        response_mode=ResponseMode.ASK_CLARIFICATION,
        entities=dict(intent_result.entities),
        confidence=intent_result.confidence,
    )

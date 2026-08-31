"""Deterministic pre-filter, checked before the LLM router.

ARCHITECTURAL RULE: this module only determines what the user is trying to
do, and only for the narrow deterministic slice documented in rules.py's
module docstring. It must never execute a banking transaction, modify
account balances, create/modify a cheque, loan, KYC, or customer record,
call a banking tool, or change workflow state.

Layered strategy (checked in this order, first confident match wins):
  0. prompt-injection / role-override detection            (rule)
  1. hard, literal navigation protocol                      (rule)
  2. workflow-confirmation shorthand (bare yes/no/confirm
     while a CONFIRM_* step is active)                       (context)
  3. unknown — everything else is the LLM router's job
     (app/conversation/intent/llm_routing.py::classify_and_route_llm),
     called directly by app/conversation/manager.py, not through this
     module. classify_intent() never calls an LLM itself: this keeps the
     deterministic pre-filter free (no Sarvam quota spent) and guarantees
     a message is never classified twice.
"""

from typing import Optional

from app.conversation.context import ConversationContext
from app.conversation.intent import rules
from app.conversation.intent.models import IntentResult, flags_for_intent
from app.logger import get_logger

logger = get_logger(__name__)


async def classify_intent(
    text: str,
    context: Optional[ConversationContext] = None,
    trace_id: str = "",
) -> IntentResult:
    """Classify one message against the deterministic layer only. Never
    raises — a failure here is a classification problem, not a banking
    problem, so callers get "unknown" back rather than an exception."""
    try:
        result = _classify(text, context)
    except Exception as e:
        logger.error(f"[{trace_id}] Intent classification raised | error={e}")
        result = IntentResult(intent="unknown", confidence=0.0, method="rule")

    result.requires_workflow, result.requires_llm = flags_for_intent(result.intent)
    _log_classification(result, context, trace_id)
    return result


def _classify(text: str, context: Optional[ConversationContext]) -> IntentResult:
    stripped = (text or "").strip()
    if not stripped:
        return IntentResult(intent="unknown", confidence=0.0, method="rule")

    # 0. Treat the message as untrusted data, not an instruction to us.
    if rules.looks_like_injection(stripped):
        return IntentResult(intent="out_of_scope", confidence=0.9, method="rule")

    # 1. Hard navigation — same meaning regardless of workflow state.
    result = rules.classify_hard_navigation(stripped)
    if result:
        return result

    # 2. Workflow-confirmation shorthand — a bare yes/no/confirm while a
    # CONFIRM_* step is awaiting exactly that answer.
    if context and context.current_workflow:
        result = rules.classify_workflow_conversation(stripped, context)
        if result:
            return result

    # 3. Everything else is the LLM router's job — see module docstring.
    return IntentResult(intent="unknown", confidence=0.0, method="rule")


def _log_classification(result: IntentResult, context: Optional[ConversationContext], trace_id: str) -> None:
    """Per the logging contract: trace_id, masked phone, intent,
    confidence, workflow, step. Never the raw message text or extracted
    entities."""
    phone_display = "unknown"
    if context and context.phone_number:
        phone_display = context.phone_number[-4:]
    workflow_display = (context.current_workflow if context else None) or "none"
    step_display = (context.current_step if context else None) or "none"
    logger.info(
        f"[{trace_id}] Intent classified | phone={phone_display} | intent={result.intent} | "
        f"confidence={result.confidence:.2f} | workflow={workflow_display} | step={step_display} | "
        f"method={result.method}"
    )

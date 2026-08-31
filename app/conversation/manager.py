"""ConversationManager — LLM-first orchestrator.

Owns the conversation turn lifecycle: load context, run the deterministic
pre-filter (injection / hard navigation / workflow-confirmation
shorthand), the registration gate, the active workflow, and the single LLM
routing decision, then persist context. See docs/current_architecture.md,
"Phase 13 — LLM-First Routing Migration".

ARCHITECTURAL RULE: this class orchestrates; it does not execute business
logic. It never queries the database directly, never validates a banking
operation, and never decides workflow state itself — all of that stays in
WorkflowManager, the workflow processors, and the workflow-start adapter.
The LLM+tools branch is injected as a callable (`llm_fallback`) rather than
imported directly, to avoid a circular import with app/agent/agent.py.

LLM call budget per turn: 0 calls for a deterministic pre-filter match
with no active workflow needing dispatch (injection, hard navigation, a
protocol/field input for an active workflow); exactly 1 call
(classify_and_route_llm[_sync]) for everything else the deterministic
layer has no opinion on; a 2nd call (the LLM+tools agent, `llm_fallback`)
only for a TOOL/RAG decision that needs real customer data or general
banking knowledge. No message is ever classified twice.
"""

import asyncio
import re
import time
from typing import Any, Awaitable, Callable, Optional

from app.conversation.builder import build_context
from app.conversation.context import ConversationContext, sanitize_workflow_data
from app.conversation.context_store import ConversationContextStore
from app.conversation.intent.classifier import classify_intent
from app.conversation.intent.llm_routing import LLMRoutingDecision, classify_and_route_llm
from app.conversation.intent.text_clean import clean_noisy_text
from app.services.language import (
    DEFAULT_LANGUAGE,
    MIN_DETECTABLE_LENGTH,
    detect_explicit_language_change,
    detect_language,
    should_attempt_detection,
    translate_text,
)
from app.conversation.renderer import ResponseLike, StructuredResponse, as_structured_response
from app.conversation.responses.common import (
    render_cancelled,
    render_clarification,
    render_main_menu_list,
    render_out_of_scope,
    render_service_unavailable,
)
from app.conversation.responses.errors import render_agent_error
from app.conversation.workflow_adapter import start_workflow_directly
from app.logger import get_logger
from app.memory import append_turn_to_session
from app.services.registration_gate import check_registration_gate
from app.workflows.manager import WorkflowManager
from app.workflows.memory import get_workflow
from app.workflows.processors.transactions import start_view_transactions

logger = get_logger(__name__)

# A small, safe cap on the stored retry_count. Nothing currently branches
# on this value; it exists so the persisted field can't grow unbounded
# across a long run of ambiguous messages.
MAX_CLARIFICATION_RETRIES = 3

# Hard-navigation/confirmation-shorthand intents the deterministic
# pre-filter (app/conversation/intent/classifier.py) can produce — these
# never need the LLM router to resolve them further.
_DETERMINISTIC_INTENTS = {"cancel", "back", "main_menu", "repeat", "start_over", "workflow_confirmation"}

# The row ids of app/conversation/responses/common.py's _MAIN_MENU_ROWS /
# app/workflows/manager.py's start_requested() menu_actions — a tapped
# WhatsApp list row arrives as this bare digit with no active workflow.
_MENU_DIGITS = {"1", "2", "3", "4", "5", "6", "7", "8"}

# A cheap, local, no-LLM signal that a message needs real reasoning
# (check a fact, then conditionally act — "transfer to my landlord if my
# balance is more than 20k") rather than a simple direct dispatch. A
# condition word alone is treated as sufficient (rather than also
# requiring a comparison word) because false positives here just mean an
# extra step through the LLM+tools agent, not a wrong answer.
_CONDITION_WORD_RE = re.compile(r"\b(if|when|unless|provided|as long as)\b", re.I)
_CHECK_VERB_RE = re.compile(r"\b(check|tell me|show|verify|find out|confirm)\b", re.I)
_ACT_VERB_RE = re.compile(r"\b(transfer|send|pay|apply|update|start)\b", re.I)


def _is_compound_or_conditional(query: str) -> bool:
    text = _as_text(query).strip()
    if not text:
        return False
    if _CONDITION_WORD_RE.search(text):
        return True
    return bool(_CHECK_VERB_RE.search(text) and _ACT_VERB_RE.search(text))


# A plain-ASCII text message with at least this many words is treated as
# an implicit "the customer is writing in English now" signal — see
# _update_language(). Below this, a short reply ("yes", "1", "ok") is too
# ambiguous to mean anything and the sticky non-English language is kept.
MIN_ENGLISH_SWITCH_WORDS = 3

LlmFallbackFn = Callable[[str, str, str, Optional[dict]], Awaitable[ResponseLike]]

_UNSET = object()


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


# Pattern-based redaction for the query/reply debug logs below — a
# key-based check (like sanitize_workflow_data(), app/conversation/
# context.py) can't help here since these are raw free-text messages, not
# a structured dict. A customer routinely TYPES their PAN/Aadhaar as a
# plain text reply during onboarding/KYC (not just uploads an image of
# it) — confirmed by tests/test_conversation_manager.py's
# test_15_sensitive_information_not_logged_or_stored, which sends a PAN
# number as the message itself. Deliberately shape-based, not
# context-aware (no reliance on knowing the current step): PAN and
# Aadhaar have distinctive enough shapes to redact unconditionally with
# low false-positive risk, unlike a bare short number (which could
# equally be a legitimate loan tenure/amount) — an OTP typed as free text
# is a known, accepted residual gap (documented in the report), not
# silently pretended away.
_PAN_RE = re.compile(r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b")
_AADHAAR_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")


def _redact_sensitive(text: str) -> str:
    text = _PAN_RE.sub("[REDACTED-PAN]", text)
    text = _AADHAAR_RE.sub("[REDACTED-ID]", text)
    return text


class ConversationManager:
    """Orchestrates one conversation turn using the existing components."""

    def __init__(
        self,
        workflow_manager: Optional[WorkflowManager] = None,
        context_store: Optional[ConversationContextStore] = None,
    ) -> None:
        self.workflow_manager = workflow_manager or WorkflowManager()
        self.context_store = context_store or ConversationContextStore()

    async def handle_message(
        self,
        phone_number: str,
        message: str,
        trace_id: str,
        llm_fallback: LlmFallbackFn,
        parsed_document: Optional[dict] = None,
        detected_language: Optional[str] = None,
        is_voice: bool = False,
    ) -> ResponseLike:
        """Run one full conversation turn and return the response text.

        Never raises — any failure anywhere in the turn is caught, logged
        with `trace_id`, and turned into an existing user-safe error
        response.
        """
        start = time.time()

        # Normalize non-string payloads before any .strip()/regex work.
        message = _as_text(message)

        # Strip laughter/filler noise ("check my balance ha ha ha") before
        # anything tries to classify or match this message.
        message = clean_noisy_text(message)

        # The customer's own message, once — everything below this point
        # only ever logs metadata (intent/action/duration), not text, so
        # this is the one place a turn's actual input is visible for
        # debugging. Never carries Aadhaar/PAN/OTP itself (those are
        # sent as images, not text), but IS the customer's raw words —
        # treat these logs with the same care as any other PII.
        logger.info(f"[{trace_id}] conversation.turn.started | phone={phone_number[-4:]} | query={_redact_sensitive(message)[:300]!r}")

        context = await asyncio.to_thread(self._load_context, phone_number, trace_id)
        if context is not None:
            context.last_user_message = message[:500]
            await self._update_language(context, message, detected_language, is_voice, trace_id)

        query = message

        try:
            pre_result = await self._classify_intent(context, query, trace_id)

            # Injection detection short-circuits everything else — the
            # message is never handed to a workflow or the LLM as if it
            # were a real request.
            if pre_result is not None and pre_result.intent == "out_of_scope":
                return await self._finish(
                    context, phone_number, query, render_out_of_scope(), trace_id, pending_action=None
                )

            has_active_workflow = bool(context and context.current_workflow)
            llm_decision: Optional[LLMRoutingDecision] = None
            forced_banking_llm = False
            is_bare_menu_digit = not has_active_workflow and query.strip() in _MENU_DIGITS

            if not has_active_workflow:
                # A tapped main-menu row (a bare digit "1".."8") carries no
                # words for any classifier to work with — it must never
                # cost an LLM call or risk being misread as OUT_OF_SCOPE/
                # CLARIFY. Deterministic digit/button protocol stays
                # deterministic: try WorkflowManager's digit map first,
                # with `decision=None` for the registration gate (its own
                # is_menu_tap check already handles this case for a
                # registered customer without needing one).
                if is_bare_menu_digit:
                    gate_result = await asyncio.to_thread(
                        check_registration_gate,
                        phone_number=phone_number,
                        query=query,
                        decision=None,
                        is_registered=bool(context and context.is_registered),
                        trace_id=trace_id,
                    )
                    if gate_result and gate_result["handled"]:
                        return await self._finish(context, phone_number, query, gate_result["response"], trace_id)

                    protocol = await asyncio.to_thread(
                        self.workflow_manager.start_requested, phone_number, query, trace_id=trace_id
                    )
                    if protocol["handled"]:
                        self._register_progress(context)
                        return await self._finish(
                            context, phone_number, query, protocol["response"], trace_id, pending_action=None
                        )
                    if protocol.get("reprocess_query"):
                        # A digit with no dedicated workflow of its own
                        # (balance, transactions, cheque status) resolves
                        # to a clear text query — answer it via the
                        # LLM+tools agent directly, no routing call needed.
                        query = protocol["reprocess_query"]
                        forced_banking_llm = True

                if not forced_banking_llm:
                    needs_llm = pre_result is None or pre_result.intent not in _DETERMINISTIC_INTENTS
                    if needs_llm:
                        llm_decision = await classify_and_route_llm(query, context=context, trace_id=trace_id)

                    gate_result = await asyncio.to_thread(
                        check_registration_gate,
                        phone_number=phone_number,
                        query=query,
                        decision=llm_decision,
                        is_registered=bool(context and context.is_registered),
                        trace_id=trace_id,
                    )
                    if gate_result and gate_result["handled"]:
                        return await self._finish(context, phone_number, query, gate_result["response"], trace_id)

            reprocess_query = None
            if not forced_banking_llm:
                workflow_result = await asyncio.to_thread(
                    self.workflow_manager.handle,
                    phone_number=phone_number,
                    query=query,
                    parsed_document=parsed_document,
                    trace_id=trace_id,
                    llm_decision=llm_decision,
                )
                reprocess_query = workflow_result.get("reprocess_query")
                if reprocess_query:
                    query = reprocess_query
                    if llm_decision is None:
                        # WorkflowManager already computed one internally
                        # to recognize this as a side question worth
                        # reprocessing — reuse it instead of discarding it
                        # and (eventually) falling back to the general
                        # agent with no idea what operation this actually
                        # is.
                        llm_decision = workflow_result.get("llm_decision")

                if workflow_result["handled"]:
                    logger.info(f"[{trace_id}] conversation.workflow.handled | phone={phone_number[-4:]}")
                    self._register_progress(context)
                    return await self._finish(context, phone_number, query, workflow_result["response"], trace_id)

            if not forced_banking_llm and pre_result is not None and pre_result.intent == "main_menu":
                self._register_progress(context)
                return await self._finish(
                    context, phone_number, query, render_main_menu_list(), trace_id, pending_action=None
                )

            is_compound = not forced_banking_llm and _is_compound_or_conditional(query)
            if is_compound or forced_banking_llm:
                # A compound/conditional request ("check my balance, and if
                # it's over 20k transfer 5000 to Priya") needs the real
                # LLM+tools agent's check-then-act reasoning, not a direct
                # workflow start or a static template. A forced-banking-LLM
                # digit reprocess (e.g. menu row "2" -> "check my balance")
                # likewise already knows its destination — no routing call
                # needed.
                if is_compound:
                    logger.info(f"[{trace_id}] conversation.route.compound_or_conditional | phone={phone_number[-4:]}")
                self._register_progress(context)
                response = await llm_fallback(query, phone_number, trace_id, parsed_document)
                return await self._finish(context, phone_number, query, response, trace_id, pending_action=None)

            if llm_decision is None and reprocess_query is None:
                # The deterministic pre-filter had no opinion (an
                # unresolved hard-nav intent like "repeat"/"start_over"
                # with no active workflow, or WorkflowManager declined a
                # protocol input without needing its own lazy call) — this
                # is the only remaining place a fresh routing call can be
                # needed, and it happens at most once.
                llm_decision = await classify_and_route_llm(query, context=context, trace_id=trace_id)

            if llm_decision is None:
                # The LLM call failed/unavailable — safe default: let the
                # general LLM+tools agent try, matching the app's existing
                # fail-safe pattern of never leaving a turn unanswered.
                self._register_progress(context)
                response = await llm_fallback(query, phone_number, trace_id, parsed_document)
                return await self._finish(context, phone_number, query, response, trace_id, pending_action=None)

            action = llm_decision.action
            if context is not None:
                # _classify_intent() only ever sets last_intent/confidence
                # from the deterministic pre-filter (injection/hard-nav/
                # confirm-shorthand), which is "unknown" for the vast
                # majority of turns now that the LLM router resolves them —
                # update it here from the actual decision so observability
                # reflects what really happened this turn.
                intent_result = llm_decision.to_intent_result()
                context.last_intent = intent_result.intent
                context.intent_confidence = intent_result.confidence
            logger.info(
                f"[{trace_id}] conversation.route.decided | phone={phone_number[-4:]} | "
                f"intent={llm_decision.intent} | action={action} | certainty={llm_decision.certainty} | "
                f"workflow={(context.current_workflow if context else None) or 'none'}"
            )

            if action == "GREETING":
                # WorkflowManager/registration_gate already absorb GREETING
                # whenever a workflow is active or none was yet started —
                # reaching here is a rare edge case (e.g. a registered
                # customer with history greeting mid-turn); fall back to
                # the same menu either would have shown.
                self._register_progress(context)
                return await self._finish(
                    context, phone_number, query, render_main_menu_list(), trace_id, pending_action=None
                )

            if action == "OUT_OF_SCOPE":
                return await self._finish(
                    context, phone_number, query, render_out_of_scope(), trace_id, pending_action=None
                )

            if action == "CLARIFY":
                response = render_clarification(llm_decision.intent)
                self._register_clarification(context)
                return await self._finish(
                    context, phone_number, query, response, trace_id, pending_action=f"clarify:{llm_decision.intent}"
                )

            if action == "CANCEL":
                # No active workflow reached this point (an active one is
                # always resolved inside WorkflowManager.handle() first) —
                # nothing to cancel.
                self._register_progress(context)
                return await self._finish(
                    context, phone_number, query, render_cancelled("That"), trace_id, pending_action=None
                )

            if action in ("START_WORKFLOW", "SWITCH") and llm_decision.certainty != "high":
                # Intent classification alone must never authorize a
                # financial action -- a workflow-starting decision only
                # begins one (still gated by its own STEP_CONFIRM_* before
                # anything is committed) at high certainty. Below that,
                # fall through to the LLM+tools agent rather than guessing
                # which real banking operation to start.
                action = "BANKING_LLM"

            if action in ("START_WORKFLOW", "SWITCH"):
                target = llm_decision.resolved_target_workflow()
                if target:
                    started = await asyncio.to_thread(
                        start_workflow_directly,
                        target,
                        phone_number,
                        transfer_handler=self.workflow_manager.transfer_handler,
                        query=query,
                        trace_id=trace_id,
                        entities=llm_decision.entities,
                    )
                    if started and started.get("handled"):
                        self._register_progress(context)
                        return await self._finish(
                            context, phone_number, query, started["response"], trace_id, pending_action=None
                        )
                # No resolvable target, or the adapter declined — fall
                # through to the deterministic menu-digit/list-tap starter,
                # then the LLM+tools agent, rather than dead-ending.
                protocol = await asyncio.to_thread(
                    self.workflow_manager.start_requested, phone_number, query, trace_id=trace_id
                )
                if protocol["handled"]:
                    self._register_progress(context)
                    return await self._finish(
                        context, phone_number, query, protocol["response"], trace_id, pending_action=None
                    )

            if action == "TOOL" and llm_decision.intent == "transaction_request" and not has_active_workflow:
                # The general LLM+tools agent has a documented reliability
                # gap here (see app/workflows/processors/transactions.py's
                # module docstring): it sometimes answers "no account
                # linked" without actually calling tool_get_last_transactions
                # at all. The LLM router has already told us definitively
                # this is a transaction_request — route straight to the
                # same deterministic handler the main-menu "View
                # transactions" row already uses (start_view_transactions),
                # instead of leaving execution up to the agent's own
                # tool-choice judgment. This is a dispatch/execution
                # change only — the LLM still does 100% of the intent
                # understanding.
                #
                # `not has_active_workflow` is deliberate: for a customer
                # with 2+ accounts, start_view_transactions() creates its
                # own WORKFLOW_VIEW_TRANSACTIONS record, and only one
                # workflow can be active per phone number — calling it
                # while a DIFFERENT workflow (e.g. an in-progress loan
                # application) is already active would silently clobber
                # it. Reaching here with an active workflow means this
                # came back as a mid-workflow side question instead (see
                # WorkflowManager's reprocess_query path), so it still
                # falls through to the read-only general agent below,
                # exactly as before this fix.
                result = await asyncio.to_thread(start_view_transactions, phone_number, trace_id)
                if result.get("handled"):
                    self._register_progress(context)
                    return await self._finish(
                        context, phone_number, query, result["response"], trace_id, pending_action=None
                    )

            # action in {"TOOL", "RAG", "CONTINUE", "CORRECT"}, or a
            # START_WORKFLOW/SWITCH nothing above could start — all
            # converge on the existing LLM+tools agent.
            self._register_progress(context)
            response = await llm_fallback(query, phone_number, trace_id, parsed_document)
            duration = (time.time() - start) * 1000
            logger.info(
                f"[{trace_id}] conversation.response.generated | phone={phone_number[-4:]} | "
                f"source=llm | duration={duration:.2f}ms"
            )
            return await self._finish(context, phone_number, query, response, trace_id, pending_action=None)

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(
                f"[{trace_id}] conversation.turn.failed | phone={phone_number[-4:]} | "
                f"error={e} | duration={duration:.2f}ms"
            )
            if context is not None:
                context.last_error = "turn_failed"
                await asyncio.to_thread(self._persist, context, phone_number, trace_id, None)
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str or "503" in error_str:
                return render_service_unavailable()
            return render_agent_error()

    # ─── internal helpers ────────────────────────────────────────────

    def _load_context(self, phone_number: str, trace_id: str) -> Optional[ConversationContext]:
        try:
            return build_context(phone_number, trace_id=trace_id)
        except Exception as e:
            logger.error(f"[{trace_id}] conversation.context.build_failed | phone={phone_number[-4:]} | error={e}")
            return None

    async def _update_language(
        self,
        context: ConversationContext,
        message: str,
        detected_language: Optional[str],
        is_voice: bool,
        trace_id: str,
    ) -> None:
        """Keep context.voice_language/text_language current, and set
        context.detected_language to whichever of the two applies to THIS
        turn — see app/services/language.py and the field docs on
        ConversationContext.

        Voice and text are tracked as two fully independent sticky
        languages — a language established on one channel never leaks
        into the other."""
        if is_voice:
            if detected_language:
                context.voice_language = detected_language
            context.detected_language = context.voice_language
            return

        stripped = (message or "").strip()
        if len(stripped) < MIN_DETECTABLE_LENGTH:
            context.detected_language = context.text_language
            return
        if should_attempt_detection(message):
            context.text_language = await detect_language(message, trace_id=trace_id)
            context.detected_language = context.text_language
            return
        if context.text_language and context.text_language != DEFAULT_LANGUAGE:
            explicit = detect_explicit_language_change(message)
            if explicit:
                context.text_language = explicit
            elif len(stripped.split()) >= MIN_ENGLISH_SWITCH_WORDS:
                context.text_language = DEFAULT_LANGUAGE
            # else: a short/low-signal reply — stays sticky, no change.
        else:
            context.text_language = DEFAULT_LANGUAGE
        context.detected_language = context.text_language

    async def _classify_intent(self, context: Optional[ConversationContext], message: str, trace_id: str):
        if context is None:
            return None
        try:
            result = await classify_intent(message, context=context, trace_id=trace_id)
            context.last_intent = result.intent
            context.intent_confidence = result.confidence
            return result
        except Exception as e:
            logger.error(
                f"[{trace_id}] conversation.intent.classification_failed | "
                f"phone={context.phone_number[-4:]} | error={e}"
            )
            return None

    def _register_clarification(self, context: Optional[ConversationContext]) -> None:
        """A clarification was needed this turn — track it without
        inventing any new routing behavior."""
        if context is None:
            return
        context.retry_count = min(context.retry_count + 1, MAX_CLARIFICATION_RETRIES)

    def _register_progress(self, context: Optional[ConversationContext]) -> None:
        """The turn moved the conversation forward (a workflow handled it,
        started, or the LLM answered) — clear any pending retry count."""
        if context is None:
            return
        context.retry_count = 0

    def _persist(
        self,
        context: Optional[ConversationContext],
        phone_number: str,
        trace_id: str,
        response_text: Optional[str],
    ) -> None:
        if context is None:
            return
        try:
            latest_workflow = get_workflow(phone_number)
            if latest_workflow:
                context.current_workflow = latest_workflow.get("type")
                context.current_step = latest_workflow.get("step")
                context.workflow_id = latest_workflow.get("workflow_id")
                context.workflow_data = sanitize_workflow_data(latest_workflow.get("data", {}))
            else:
                context.current_workflow = None
                context.current_step = None
                context.workflow_id = None
                context.workflow_data = {}
            if response_text is not None:
                context.last_assistant_message = response_text[:500]
            self.context_store.save(context, trace_id=trace_id)
        except Exception as e:
            logger.error(f"[{trace_id}] conversation.context.save_failed | phone={phone_number[-4:]} | error={e}")

    async def _finish(
        self,
        context: Optional[ConversationContext],
        phone_number: str,
        query: str,
        response: ResponseLike,
        trace_id: str,
        pending_action: Any = _UNSET,
    ) -> ResponseLike:
        if context is not None and pending_action is not _UNSET:
            context.pending_action = pending_action

        was_structured = isinstance(response, StructuredResponse)
        structured = as_structured_response(response)

        # Every response generated above this point is authored in
        # English (templates, RAG/LLM output, error text). Translate once,
        # here, right before it's sent/logged/persisted.
        if context is not None and context.detected_language != DEFAULT_LANGUAGE:
            structured.text = await translate_text(structured.text, context.detected_language, trace_id=trace_id)

        # Record the language `text` actually ends up in (English included)
        # so a voice reply can be spoken in the same language it was
        # translated into.
        structured.language = context.detected_language if context is not None else None

        await asyncio.gather(
            asyncio.to_thread(append_turn_to_session, phone_number, query, structured.text[:500]),
            asyncio.to_thread(self._persist, context, phone_number, trace_id, structured.text),
        )
        # The reply actually sent, paired with the query log at turn start
        # (same trace_id) — this is exactly the text the customer receives
        # over WhatsApp, so it carries nothing beyond what already went
        # through every existing template/masking safeguard (account
        # numbers masked, Aadhaar/PAN/OTP never echoed — see
        # app/conversation/responses/common.py's module docstring).
        logger.info(f"[{trace_id}] conversation.turn.completed | phone={phone_number[-4:]} | reply={_redact_sensitive(structured.text)[:300]!r}")
        return structured if was_structured else structured.text

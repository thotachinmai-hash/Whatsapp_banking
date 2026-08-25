"""ConversationManager — Phase 5 orchestrator.

Owns the conversation turn lifecycle: load context, classify intent, run
the registration gate, the active workflow, and the router, then persist
context — calling existing components for every step. See
docs/current_architecture.md, "Conversation Manager — Phase 5".

ARCHITECTURAL RULE: this class orchestrates; it does not execute business
logic. It never queries the database directly, never validates a banking
operation, and never decides workflow state itself — all of that stays in
WorkflowManager, the workflow processors, the intent classifier, and the
router, exactly as before this phase. The LLM+tools branch is injected as
a callable (`llm_fallback`) rather than imported directly: it is owned by
app/agent/agent.py, which in turn imports this module, so importing it
back here would be circular — dependency injection is the clean way
around that, not a workaround.
"""

import re
import time
from typing import Any, Awaitable, Callable, Optional

from app.conversation.builder import build_context
from app.conversation.context import ConversationContext, sanitize_workflow_data
from app.conversation.context_store import ConversationContextStore
from app.conversation.guidance.handoff import WORKFLOW_FOR_ACTION, resolve_pending_action
from app.conversation.guidance.models import GuidanceAction, GuidanceType
from app.conversation.guidance.policy import build_guidance
from app.conversation.guidance.responses import render_action_info, render_guidance
from app.conversation.intent import classify_intent
from app.conversation.intent.text_clean import clean_noisy_text
from app.services.language import (
    DEFAULT_LANGUAGE,
    MIN_DETECTABLE_LENGTH,
    detect_explicit_language_change,
    detect_language,
    should_attempt_detection,
    translate_text,
)
from app.services.llm_understanding import is_llm_fallback_enabled
from app.conversation.intent.classifier import default_llm_classify
from app.conversation.renderer import ResponseLike, StructuredResponse, as_structured_response
from app.conversation.responses.common import render_cancelled, render_clarification, render_low_confidence
from app.conversation.responses.common import render_main_menu_list, render_out_of_scope, render_service_unavailable
from app.conversation.responses.errors import render_agent_error
from app.conversation.router import route_intent
from app.conversation.workflow_adapter import start_workflow_directly
from app.logger import get_logger
from app.memory import append_to_session
from app.services.registration_gate import check_registration_gate
from app.workflows.manager import WorkflowManager
from app.workflows.memory import get_workflow

logger = get_logger(__name__)

# A small, safe cap on the stored retry_count — see docs, "Retry count".
# Nothing currently branches on this value; it exists so the persisted
# field can't grow unbounded across a long run of ambiguous messages.
MAX_CLARIFICATION_RETRIES = 3

# A cheap, local, no-LLM signal that a message needs real reasoning
# (check a fact, then conditionally act — "transfer to my landlord if my
# balance is more than 20k") rather than the deterministic
# keyword-triggered workflow starters (WorkflowManager.start_requested).
# Those blindly grab everything after "to"/a keyword via regex and have
# no concept of a condition — see the compound-request routing plan. A
# condition word alone is treated as sufficient (rather than also
# requiring a comparison word) because false positives here just mean an
# extra LLM round-trip, not a wrong answer, whereas requiring a
# comparison word missed real cases like "check whether I paid my
# landlord this month; if not, transfer ₹15,000". Deliberately narrow —
# a plain "send 500 to Priya please" must still take the fast,
# well-tested deterministic path.
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


def _looks_like_transfer_request_with_condition(query: str) -> bool:
    """Compound transfer intents with a balance condition still belong in the
    deterministic transfer flow (the same free-text parser that handles plain
    'send 2000 to Bhavitha' messages) rather than being sent to the generic
    LLM fallback. The condition check is a policy decision to be handled by
    the transfer workflow or the LLM tool layer, not a reason to drop the
    actual transfer intent entirely."""
    text = (query or "").strip()
    if not text:
        return False
    has_transfer_intent = bool(re.search(r"\b(?:transfer|send|pay)\b", text, re.I))
    has_beneficiary = bool(re.search(r"\bto\s+[A-Za-z][A-Za-z .'-]{1,50}\b", text, re.I))
    has_condition = bool(_CONDITION_WORD_RE.search(text))
    return has_transfer_intent and has_beneficiary and has_condition


def _looks_like_numeric_value(query: str) -> bool:
    """A bare number is not out of scope: it could be an amount the user is
    typing into a transfer or another numeric prompt. Keep menu taps on the
    dedicated digit path, but let true numeric values continue through to the
    regular LLM/tool logic instead of being rejected as a generic banking
    question."""
    text = (query or "").strip()
    return bool(text) and text not in {"1", "2", "3", "4", "5", "6", "7", "8"} and bool(re.fullmatch(r"\d+(?:[.,]\d+)?", text))

# A plain-ASCII text message with at least this many words is treated as
# an implicit "the customer is writing in English now" signal — see
# _update_language(). Below this, a short reply ("yes", "1", "ok") is too
# ambiguous to mean anything and the sticky non-English language is kept.
MIN_ENGLISH_SWITCH_WORDS = 3

# Guidance types that replace a turn's answer (skipping the LLM, or —
# for the two loan/cheque classifier gaps documented in
# app/conversation/guidance/policy.py — skipping an incorrect
# START_WORKFLOW) with a deterministic guidance response. Deliberately
# NOT every GuidanceType: GENERAL_BANKING_GUIDANCE and TRANSACTION_GUIDANCE
# stay on the existing LLM+tools path (the LLM/tools are the only place
# that can answer an open banking question or a real spending-insight
# question with actual data — see docs, "Banking Guidance & Response
# Handoff — Phase 7"); WORKFLOW_HELP stays on the existing path too (an
# active workflow's LLM-explained "what should I do" already correctly
# explains the current step without restarting it — replacing it here
# would risk drifting from the live workflow's real prompts); UNKNOWN
# stays on the router's own CLARIFICATION_REQUIRED text, which already
# exists for exactly this case. LOAN_GUIDANCE was moved off this list once
# app/agent/tools.py::tool_get_loan_product_info() gave the LLM a real
# rate/fee/tenure lookup — a plain "what's the interest rate" question now
# gets a real, tool-provided figure instead of the guidance layer's
# necessarily-generic "I can't quote exact numbers here". LOAN_DOCUMENT_GUIDANCE
# got the same treatment for the same reason once app/agent/tools.py::
# tool_search_bank_documents() (RAG over app/rag/documents/) gave the LLM a
# real, indexed answer for "what documents do I need" — confirmed via live
# testing that without this, that exact question never reached the RAG
# tool at all; it was answered by this layer's generic "the exact list
# depends..." text first. LOAN_ELIGIBILITY_GUIDANCE stays intercepted: it's
# about a *personal* decision (never invented/computed), not the bank's
# published rate card or document policy.
_INTERCEPT_GUIDANCE_TYPES = {
    GuidanceType.LOAN_ELIGIBILITY_GUIDANCE,
    GuidanceType.TRANSFER_GUIDANCE,
    GuidanceType.CHEQUE_GUIDANCE,
    GuidanceType.CHEQUE_STATUS_GUIDANCE,
    GuidanceType.KYC_GUIDANCE,
    GuidanceType.ACCOUNT_GUIDANCE,
}

LlmFallbackFn = Callable[[str, str, str, Optional[dict]], Awaitable[ResponseLike]]

_UNSET = object()


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


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
        response, matching the guarantee run_agent() made before this
        phase existed.
        """
        start = time.time()
        logger.info(f"[{trace_id}] conversation.turn.started | phone={phone_number[-4:]}")

        # Normalize non-string payloads before any .strip()/regex work.
        # Numeric menu taps or other unexpected types can reach this entry
        # point from downstream callers; turning them into strings here keeps
        # the rest of the pipeline consistent and prevents crashes like
        # 'int' object has no attribute 'strip'.
        message = _as_text(message)

        # Strip laughter/filler noise ("check my balance ha ha ha") before
        # anything tries to classify or match this message — see
        # app/conversation/intent/text_clean.py. Voice transcriptions and
        # typed text both flow through here, so this covers both.
        message = clean_noisy_text(message)

        context = self._load_context(phone_number, trace_id)
        if context is not None:
            context.last_user_message = message[:500]
            self._update_language(context, message, detected_language, is_voice, trace_id)

        intent_result = self._classify_intent(context, message, trace_id)
        query = message
        reprocess_query = None

        try:
            gate_result = await check_registration_gate(
                phone_number=phone_number, query=query, trace_id=trace_id
            )
            if gate_result and gate_result["handled"]:
                return self._finish(context, phone_number, query, gate_result["response"], trace_id)

            handoff_result = self._try_pending_action_handoff(context, phone_number, query, trace_id)
            if handoff_result is not None:
                return handoff_result

            workflow_result = await self.workflow_manager.handle(
                phone_number=phone_number,
                query=query,
                parsed_document=parsed_document,
                trace_id=trace_id,
            )
            reprocess_query = workflow_result.get("reprocess_query")
            if reprocess_query:
                query = reprocess_query

            if workflow_result["handled"]:
                logger.info(f"[{trace_id}] conversation.workflow.handled | phone={phone_number[-4:]}")
                self._register_progress(context)
                return self._finish(context, phone_number, query, workflow_result["response"], trace_id)

            routing_decision = None
            if intent_result is not None:
                if intent_result.intent == "main_menu":
                    self._register_progress(context)
                    return self._finish(
                        context, phone_number, query, render_main_menu_list(), trace_id, pending_action=None
                    )
                routing_decision = route_intent(intent_result, context=context)
                logger.info(
                    f"[{trace_id}] conversation.route.decided | phone={phone_number[-4:]} | "
                    f"intent={intent_result.intent} | confidence={intent_result.confidence:.2f} | "
                    f"routing_decision={routing_decision.action} | "
                    f"workflow={(context.current_workflow if context else None) or 'none'} | "
                    f"step={(context.current_step if context else None) or 'none'}"
                )
            action = routing_decision.action if routing_decision else "SAFE_FALLBACK"

            is_compound = _is_compound_or_conditional(query)
            if is_compound and _looks_like_transfer_request_with_condition(query):
                deterministic = self.workflow_manager.start_requested(phone_number, query, trace_id=trace_id)
                if deterministic["handled"]:
                    self._register_progress(context)
                    return self._finish(
                        context, phone_number, query, deterministic["response"], trace_id, pending_action=None
                    )
                logger.info(
                    f"[{trace_id}] conversation.route.compound_transfer_fallback | "
                    f"phone={phone_number[-4:]} | query={query[:80]!r}"
                )

            if is_compound:
                # Skip guidance interception and the deterministic
                # keyword-triggered workflow starters entirely — both are
                # single-intent shortcuts that would either mangle this
                # message (see start_requested()'s free-text regexes) or
                # answer only part of it. Go straight to the LLM+tools
                # agent, which has the check-then-act tools/instructions
                # needed to actually reason through it.
                logger.info(f"[{trace_id}] conversation.route.compound_or_conditional | phone={phone_number[-4:]}")
                action = "BANKING_LLM"

            guidance_response = None if is_compound else self._try_guidance(
                intent_result, context, phone_number, query, trace_id
            )
            if guidance_response is not None:
                return guidance_response

            if action in ("OUT_OF_SCOPE", "CLARIFICATION_REQUIRED") and query.strip() in {
                "1", "2", "3", "4", "5", "6", "7", "8",
            }:
                # A tapped main-menu row (or a typed bare digit) carries no
                # words for the intent classifier to work with, so it can
                # legitimately fall out as OUT_OF_SCOPE/CLARIFICATION_REQUIRED
                # even though WorkflowManager.start_requested()'s digit map
                # (see app/workflows/manager.py::menu_actions) knows exactly
                # what "2" means. Give that deterministic check one try
                # before giving up. Deliberately gated to an exact digit
                # match only (not the rest of start_requested()'s keyword
                # matching) — CLARIFICATION_REQUIRED also covers a
                # low-confidence WORKFLOW_EXECUTING_INTENTS guess (e.g. "I
                # might want a loan"), and the router's whole point there is
                # to ask rather than guess before starting a financial
                # workflow; a bare digit has no such ambiguity to protect
                # against.
                deterministic = self.workflow_manager.start_requested(phone_number, query, trace_id=trace_id)
                if deterministic["handled"]:
                    self._register_progress(context)
                    return self._finish(
                        context, phone_number, query, deterministic["response"], trace_id, pending_action=None
                    )
                if deterministic.get("reprocess_query"):
                    # Digit rows with no dedicated workflow of their own
                    # (balance, transactions, cheque status) resolve here to
                    # a clear text query instead — let the LLM+tools path
                    # below answer it rather than telling the customer their
                    # unambiguous menu tap was out of scope.
                    query = deterministic["reprocess_query"]
                    action = "BANKING_LLM"

            if action == "OUT_OF_SCOPE" and _looks_like_numeric_value(query):
                action = "BANKING_LLM"

            if action == "OUT_OF_SCOPE":
                return self._finish(
                    context, phone_number, query, render_out_of_scope(), trace_id, pending_action=None
                )

            if action == "CLARIFICATION_REQUIRED":
                reason = routing_decision.reason or "unknown"
                response = render_low_confidence() if reason == "unknown" else render_clarification(reason)
                self._register_clarification(context)
                return self._finish(
                    context, phone_number, query, response, trace_id, pending_action=f"clarify:{reason}"
                )

            if action in ("START_WORKFLOW", "SAFE_FALLBACK"):
                # START_WORKFLOW: high-confidence request to begin a new
                # workflow — reuse the existing deterministic starter
                # rather than duplicating "how a workflow begins" logic.
                # SAFE_FALLBACK: the router has no confident opinion (e.g.
                # a navigation intent that should already have been
                # handled upstream, or an unmapped/errored classification)
                # — behave exactly as before Phase 3 and give the legacy
                # keyword starter its normal chance.
                requested_workflow = self.workflow_manager.start_requested(phone_number, query, trace_id=trace_id)
                if (
                    not requested_workflow["handled"]
                    and action == "START_WORKFLOW"
                    and routing_decision.workflow
                ):
                    # The classifier recognized this as a workflow-start
                    # request but start_requested()'s own (narrower)
                    # keyword gate didn't — fall back to the
                    # smallest-necessary adapter rather than losing the
                    # router's confident decision to the general LLM.
                    requested_workflow = start_workflow_directly(
                        routing_decision.workflow,
                        phone_number,
                        transfer_handler=self.workflow_manager.transfer_handler,
                        query=query,
                        trace_id=trace_id,
                    ) or requested_workflow
                if requested_workflow["handled"]:
                    self._register_progress(context)
                    return self._finish(
                        context, phone_number, query, requested_workflow["response"], trace_id, pending_action=None
                    )
                # Router expected a workflow start (or had no opinion) but
                # neither the deterministic starter nor the adapter
                # recognized it — fall through to the LLM below rather
                # than dead-ending the turn.

            # action in {"BANKING_LLM", "WORKFLOW"}, or a START_WORKFLOW/
            # SAFE_FALLBACK the deterministic starter didn't pick up, all
            # converge on the existing LLM+tools agent — unchanged, and
            # owned by app/agent/agent.py (injected as `llm_fallback`).
            self._register_progress(context)
            response = await llm_fallback(query, phone_number, trace_id, parsed_document)
            duration = (time.time() - start) * 1000
            logger.info(
                f"[{trace_id}] conversation.response.generated | phone={phone_number[-4:]} | "
                f"source=llm | duration={duration:.2f}ms"
            )
            return self._finish(context, phone_number, query, response, trace_id, pending_action=None)

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(
                f"[{trace_id}] conversation.turn.failed | phone={phone_number[-4:]} | "
                f"error={e} | duration={duration:.2f}ms"
            )
            if context is not None:
                context.last_error = "turn_failed"
                self._persist(context, phone_number, trace_id, None)
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

    def _update_language(
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
        languages. A language established on a voice call must never
        leak into a text reply, and a language established over text must
        never leak into a voice reply — each channel is answered in
        whatever language THAT channel is currently using (see the
        module's "Language Separation" requirement). Without this split,
        a Hindi voice conversation followed by a short English text reply
        ("ok") would incorrectly keep answering in Hindi, since a bare
        ASCII reply carries no language signal of its own and used to
        fall back to one shared sticky field.

        Voice branch: every voice turn re-signals its own language
        (Sarvam STT detects it from the audio itself, not the transcript
        text — see app/services/transcription.py), so voice-to-voice
        never gets stuck: a customer who spoke Hindi and then English gets
        an English voice reply immediately, no special-casing needed here.
        `detected_language` is None when STT's own tag was dropped as
        unreliable (transcript too short to carry real signal) — the
        voice channel then simply stays on whatever it last established
        (voice continuity), same idea as the text branch's stickiness
        below but scoped to voice_language only.

        Text branch: a bare "yes"/"1" reply is too short to mean anything
        either way, so it keeps whatever language the TEXT channel already
        established — sticky ONLY for low-signal text. Once a non-English
        text language is established, a SHORT plain-ASCII reply does NOT
        silently reset the conversation back to English (the customer
        never asked for that), but a substantive plain-ASCII message (a
        real sentence, not a menu-tap) is itself a strong, unambiguous
        signal the customer has switched to English for this turn — it
        doesn't need "reply in English" phrasing
        (detect_explicit_language_change) to count. Language changes on: a
        genuine non-ASCII detection (should_attempt_detection), an
        explicit meta-request ("reply in English", "switch to Hindi") —
        see detect_explicit_language_change — or a plain-ASCII message
        with real sentence content. If no non-English language is active,
        the ASCII fast path is unchanged (no LLM call, stays English)."""
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
            context.text_language = detect_language(message, trace_id=trace_id)
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

    def _classify_intent(self, context: Optional[ConversationContext], message: str, trace_id: str):
        if context is None:
            return None
        try:
            llm_classify = default_llm_classify if is_llm_fallback_enabled() else None
            result = classify_intent(message, context=context, trace_id=trace_id, llm_classify=llm_classify)
            context.last_intent = result.intent
            context.intent_confidence = result.confidence
            logger.info(
                f"[{trace_id}] conversation.intent.classified | phone={context.phone_number[-4:]} | "
                f"intent={result.intent} | confidence={result.confidence:.2f}"
            )
            return result
        except Exception as e:
            logger.error(
                f"[{trace_id}] conversation.intent.classification_failed | "
                f"phone={context.phone_number[-4:]} | error={e}"
            )
            return None

    def _try_pending_action_handoff(
        self,
        context: Optional[ConversationContext],
        phone_number: str,
        query: str,
        trace_id: str,
    ) -> Optional[str]:
        """Task 9.2 — resolve a reply to a guidance offer from the previous
        turn ("Start application", "Show documents", "Cancel", "2", ...)
        into the existing workflow-start mechanism, BEFORE intent
        classification/routing sees this message. Necessary because
        short action-selection replies like "Start application" carry no
        banking keyword and would otherwise be misclassified as
        out_of_scope by the (unmodified) classifier — see
        docs/current_architecture.md, "Banking Guidance & Response
        Handoff — Phase 7".

        Only ever consulted when there is NO active workflow (a stale
        guidance offer must never hijack a numbered reply that's actually
        meant for a real in-progress workflow step, e.g. selecting a
        beneficiary by number). Returns the final response string if this
        turn was fully handled here, else None (continue the normal flow).
        """
        if context is None or context.current_workflow or not context.pending_action:
            return None
        if not context.pending_action.startswith("guidance:"):
            return None

        allowed_actions = context.allowed_actions
        selected = resolve_pending_action(query, allowed_actions)

        if selected is None:
            # Doesn't match the offer — treat as a fresh message, not a
            # dead end. Clear the stale offer so it can't leak into an
            # unrelated future turn.
            context.pending_action = None
            context.allowed_actions = []
            return None

        if selected == GuidanceAction.CANCEL:
            context.allowed_actions = []
            self._register_progress(context)
            return self._finish(
                context, phone_number, query, render_cancelled(), trace_id, pending_action=None
            )

        if selected == GuidanceAction.BACK:
            context.allowed_actions = []
            self._register_progress(context)
            return self._finish(
                context, phone_number, query, render_main_menu_list(), trace_id, pending_action=None
            )

        workflow_type = WORKFLOW_FOR_ACTION.get(selected)
        if workflow_type:
            started = start_workflow_directly(
                workflow_type, phone_number, transfer_handler=self.workflow_manager.transfer_handler,
                query=query, trace_id=trace_id,
            )
            if started and started["handled"]:
                logger.info(
                    f"[{trace_id}] conversation.guidance.action_selected | phone={phone_number[-4:]} | "
                    f"action={selected.value} | workflow={workflow_type}"
                )
                context.allowed_actions = []
                self._register_progress(context)
                return self._finish(
                    context, phone_number, query, started["response"], trace_id, pending_action=None
                )
            # Adapter declined (e.g. transfer with no transferable
            # balance already returns its own handled=True response
            # above) — fall through to a safe generic reply rather than
            # silently doing nothing.
            context.pending_action = None
            context.allowed_actions = []
            return self._finish(
                context, phone_number, query, render_agent_error(), trace_id, pending_action=None
            )

        # A SHOW_* info action — render the short explanation and keep the
        # SAME offer available (context.allowed_actions unchanged) so the
        # user can still say "start it" right after reading the info.
        info_text = render_action_info(selected)
        logger.info(
            f"[{trace_id}] conversation.guidance.action_selected | phone={phone_number[-4:]} | "
            f"action={selected.value} | workflow=none"
        )
        return self._finish(context, phone_number, query, info_text, trace_id, pending_action=_UNSET)

    def _try_guidance(
        self,
        intent_result,
        context: Optional[ConversationContext],
        phone_number: str,
        query: str,
        trace_id: str,
    ) -> Optional[str]:
        """Task 9.2 — deterministic guidance instead of the general LLM
        (or, for two documented classifier gaps, instead of an incorrect
        workflow start) for a curated set of guidance types — see
        `_INTERCEPT_GUIDANCE_TYPES`'s docstring above for exactly which,
        and why the rest stay on the existing path unchanged. Returns the
        final response string if handled, else None."""
        if intent_result is None:
            return None
        try:
            guidance_result = build_guidance(query, intent_result, context)
        except Exception as e:
            logger.error(f"[{trace_id}] conversation.guidance.failed | phone={phone_number[-4:]} | error={e}")
            return None
        if guidance_result is None or guidance_result.guidance_type not in _INTERCEPT_GUIDANCE_TYPES:
            return None

        rendered = render_guidance(guidance_result, context)
        logger.info(
            f"[{trace_id}] conversation.guidance.rendered | phone={phone_number[-4:]} | "
            f"guidance_type={guidance_result.guidance_type.value} | "
            f"response_mode={guidance_result.response_mode.value} | "
            f"actions={[a.value for a in rendered.actions]}"
        )
        if context is not None:
            context.allowed_actions = [a.value for a in rendered.actions]
        pending = f"guidance:{rendered.primary_action.value}" if rendered.primary_action else None
        self._register_progress(context)
        return self._finish(context, phone_number, query, rendered.text, trace_id, pending_action=pending)

    def _register_clarification(self, context: Optional[ConversationContext]) -> None:
        """A clarification was needed this turn — track it without
        inventing any new routing behavior (see module docstring)."""
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

    def _finish(
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

        # `response` may be a plain string or a StructuredResponse carrying
        # WhatsApp interactive buttons/list metadata (see
        # app/conversation/renderer.py) — either way, `.text` is the body
        # that gets translated/logged/persisted; the interactive metadata
        # (if any) passes through untouched to the eventual render_and_send
        # call. Button/list option titles are NOT translated in this
        # version — only the body text — a known limitation for non-English
        # interactive prompts.
        was_structured = isinstance(response, StructuredResponse)
        structured = as_structured_response(response)

        # Every response generated above this point is authored in
        # English (templates, RAG/LLM output, error text). Translate once,
        # here, right before it's sent/logged/persisted — a single choke
        # point instead of translating at each of the many call sites
        # above. See app/services/language.py; never blocks the turn on
        # failure — it falls back to the original English text.
        if context is not None and context.detected_language != DEFAULT_LANGUAGE:
            structured.text = translate_text(structured.text, context.detected_language, trace_id=trace_id)

        # Record the language `text` actually ends up in (English included)
        # so a voice reply can be spoken in the same language it was
        # translated into — see StructuredResponse.language's docstring.
        structured.language = context.detected_language if context is not None else None

        # Session history (app/memory.py) is a separate, unchanged
        # mechanism from ConversationContext — see docs/current_architecture.md,
        # "Conversation Context — Phase 1". Never logs the raw message —
        # only Redis stores it, under the same TTL/retention as before.
        append_to_session(phone_number, "user", query)
        append_to_session(phone_number, "assistant", structured.text[:500])
        self._persist(context, phone_number, trace_id, structured.text)
        logger.info(f"[{trace_id}] conversation.turn.completed | phone={phone_number[-4:]}")
        return structured if was_structured else structured.text

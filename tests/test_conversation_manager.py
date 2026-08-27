import unittest
from unittest.mock import patch

from app.conversation.context import ConversationContext
from app.conversation.intent.models import IntentResult
from app.conversation.manager import ConversationManager
from app.conversation.renderer import ResponseKind, as_structured_response
from app.conversation.responses.common import render_main_menu_list
from app.workflows.constants import STEP_CONFIRM_TRANSFER


def _fresh_context(phone_number="441111111111"):
    return ConversationContext(phone_number=phone_number)


class _FakeWorkflowManager:
    """A minimal stand-in for WorkflowManager — ConversationManager only
    ever calls .handle()/.start_requested()/.transfer_handler on it, so a
    real WorkflowManager (with real Postgres/Redis calls) isn't needed to
    test orchestration."""

    def __init__(self, handle_result=None, start_requested_result=None):
        self.handle_result = handle_result or {"handled": False, "response": None}
        self.start_requested_result = start_requested_result or {"handled": False, "response": None}
        self.handle_calls = []
        self.start_requested_calls = []
        self.transfer_handler = object()

    def handle(self, phone_number, query, parsed_document=None, trace_id=""):
        self.handle_calls.append((phone_number, query))
        return self.handle_result

    def start_requested(self, phone_number, query, trace_id=""):
        self.start_requested_calls.append((phone_number, query))
        return self.start_requested_result


async def _fake_llm_fallback(query, phone_number, trace_id, parsed_document=None):
    return f"llm-answer:{query}"


def _manager_with(handle_result=None, start_requested_result=None):
    wf = _FakeWorkflowManager(handle_result, start_requested_result)
    return ConversationManager(workflow_manager=wf), wf


class ConversationManagerTests(unittest.IsolatedAsyncioTestCase):
    """Every test patches app.conversation.manager.* — that's where
    ConversationManager actually calls these collaborators from (see
    tests/test_conversation_router.py for the same pattern applied to
    run_agent())."""

    def _patches(self, context=None, registration_gate_result=None):
        context = context if context is not None else _fresh_context()
        return (
            patch("app.conversation.manager.build_context", return_value=context),
            patch("app.conversation.manager.check_registration_gate", return_value=registration_gate_result),
            patch("app.conversation.manager.get_workflow", return_value=None),
            patch("app.conversation.manager.append_to_session"),
        )

    # 1. New banking question
    async def test_01_new_banking_question_reaches_llm(self):
        # "What is an overdraft?" -> banking_question -> GENERAL_BANKING_GUIDANCE,
        # which Task 9.2 deliberately does NOT intercept (see manager.py's
        # _INTERCEPT_GUIDANCE_TYPES docstring) — a broad/open banking
        # question still goes to the LLM, unlike the curated guidance
        # types (loan/transfer/cheque/kyc/account) that now short-circuit it.
        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "What is an overdraft?", "t1", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "llm-answer:What is an overdraft?")

    # 2. Out-of-scope request
    async def test_02_out_of_scope_never_reaches_llm(self):
        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        llm_calls = []

        async def tracking_llm(*args, **kwargs):
            llm_calls.append(args)
            return "should not be called"

        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "Why is the sky blue?", "t2", llm_fallback=tracking_llm
            )
        self.assertEqual(llm_calls, [])
        self.assertIn("banking", response.lower())

    async def test_02a_non_string_message_is_normalized_before_stripping(self):
        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", 2000, "t2a", llm_fallback=_fake_llm_fallback
            )
        self.assertIsInstance(response, str)
        self.assertIn("2000", response)

    async def test_02aa_display_menu_returns_list(self):
        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch.object(ConversationManager, "_classify_intent", return_value=IntentResult(intent="main_menu", confidence=0.95)):
            response = await manager.handle_message(
                "441111111111", "Display menu", "t2aa", llm_fallback=_fake_llm_fallback
            )
        structured = as_structured_response(response)
        self.assertEqual(structured.kind, ResponseKind.LIST)
        self.assertIn("What would you like to do?", structured.text)
        self.assertEqual(wf.start_requested_calls, [])
        self.assertEqual(structured.text, render_main_menu_list().text)
        self.assertEqual(len(structured.list_sections), 1)
        self.assertEqual(len(structured.list_sections[0].rows), 8)

    async def test_02b_conditional_transfer_request_starts_transfer_workflow(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "Who would you like to pay?"}
        )
        p1, p2, p3, p4 = self._patches()
        query = (
            "I want to transfer 2000 to Bhavitha if my balance is greater than 10000. "
            "Proceed with transfer if balance is greater than 10000"
        )
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message("441111111111", query, "t2b", llm_fallback=_fake_llm_fallback)
        self.assertEqual(response, "Who would you like to pay?")
        self.assertEqual(len(wf.start_requested_calls), 1)

    # 3. Low-confidence request
    async def test_03_low_confidence_request_asks_for_clarification(self):
        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "Maybe send some money to Rahul", "t3", llm_fallback=_fake_llm_fallback
            )
        self.assertIn("who", response.lower())
        self.assertNotIn("llm-answer", response)

    # 4. New transfer request
    async def test_04_new_transfer_request_starts_workflow(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "Who would you like to pay?"}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "Send £500 to Priya", "t4", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "Who would you like to pay?")
        self.assertEqual(len(wf.start_requested_calls), 1)

    # 5. New loan request
    async def test_05_new_loan_request_starts_workflow(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "Loan application started."}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "I want a personal loan", "t5", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "Loan application started.")

    # 6. New cheque request
    async def test_06_new_cheque_request_starts_workflow(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "Cheque deposit started."}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "Deposit this cheque", "t6", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "Cheque deposit started.")

    # 7. New KYC request
    async def test_07_new_kyc_request_starts_workflow(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "KYC update started."}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "I want to update my KYC", "t7", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "KYC update started.")

    # 8. Active workflow message — WorkflowManager stays authoritative
    async def test_08_active_workflow_message_is_authoritative(self):
        manager, wf = _manager_with(
            handle_result={"handled": True, "response": "Which account should we use?"}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "500", "t8", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "Which account should we use?")
        self.assertEqual(len(wf.handle_calls), 1)
        self.assertEqual(len(wf.start_requested_calls), 0)

    # 9. Clarification flow — pending_action persists across the turn
    async def test_09_clarification_sets_pending_action(self):
        manager, wf = _manager_with()
        context = _fresh_context()
        p1, p2, p3, p4 = self._patches(context=context)
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            await manager.handle_message(
                "441111111111", "Maybe send some money to Rahul", "t9", llm_fallback=_fake_llm_fallback
            )
        self.assertIsNotNone(context.pending_action)
        self.assertTrue(context.pending_action.startswith("clarify:"))

    # 10. Context persistence — save() is called with the context
    async def test_10_context_is_persisted(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "Loan application started."}
        )
        context = _fresh_context()
        p1, p2, p3, p4 = self._patches(context=context)
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True) as mock_save:
            await manager.handle_message(
                "441111111111", "I want a personal loan", "t10", llm_fallback=_fake_llm_fallback
            )
        mock_save.assert_called()
        saved_context = mock_save.call_args[0][0]
        self.assertIs(saved_context, context)
        self.assertEqual(context.last_assistant_message, "Loan application started.")

    # 11. Context sanitization — workflow_data never carries sensitive fields
    async def test_11_context_sanitizes_workflow_data(self):
        manager, wf = _manager_with()
        context = _fresh_context()
        p1, p2, _p3, p4 = self._patches(context=context)
        with p1, p2, p4, \
             patch("app.conversation.manager.get_workflow", return_value={
                 "type": "onboarding", "step": "COLLECT_AADHAAR", "workflow_id": "wf-1",
                 "data": {"full_name": "John Smith", "aadhaar_number": "123456789012"},
             }), \
             patch.object(manager.context_store, "save", return_value=True):
            await manager.handle_message(
                "441111111111", "What should I do?", "t11", llm_fallback=_fake_llm_fallback
            )
        self.assertNotIn("aadhaar_number", context.workflow_data)
        self.assertEqual(context.workflow_data.get("full_name"), "John Smith")

    # 12. Retry count increments on clarification, resets on progress
    async def test_12_retry_count_increments_then_resets(self):
        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "Loan application started."}
        )
        context = _fresh_context()
        p1, p2, p3, p4 = self._patches(context=context)
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            await manager.handle_message(
                "441111111111", "Maybe send some money to Rahul", "t12a", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(context.retry_count, 1)

        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            await manager.handle_message(
                "441111111111", "I want a personal loan", "t12b", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(context.retry_count, 0)

    async def test_12b_retry_count_is_capped(self):
        from app.conversation.manager import MAX_CLARIFICATION_RETRIES

        manager, wf = _manager_with()
        context = _fresh_context()
        p1, p2, p3, p4 = self._patches(context=context)
        for i in range(MAX_CLARIFICATION_RETRIES + 5):
            with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
                await manager.handle_message(
                    "441111111111", "Maybe send some money", f"t12c-{i}", llm_fallback=_fake_llm_fallback
                )
        self.assertLessEqual(context.retry_count, MAX_CLARIFICATION_RETRIES)

    # 13. Error handling — never exposes exceptions
    async def test_13_error_handling_never_exposes_exceptions(self):
        manager, wf = _manager_with()

        async def broken_llm(*args, **kwargs):
            raise RuntimeError("psycopg2.errors.UniqueViolation: duplicate key at 0x7f")

        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "What is an overdraft?", "t13", llm_fallback=broken_llm
            )
        self.assertNotIn("psycopg2", response)
        self.assertNotIn("Traceback", response)
        self.assertIn("try again", response.lower())

    async def test_13b_error_sets_last_error_on_context(self):
        manager, wf = _manager_with()
        context = _fresh_context()

        async def broken_llm(*args, **kwargs):
            raise RuntimeError("boom")

        p1, p2, p3, p4 = self._patches(context=context)
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True) as mock_save:
            await manager.handle_message(
                "441111111111", "What is an overdraft?", "t13b", llm_fallback=broken_llm
            )
        self.assertEqual(context.last_error, "turn_failed")
        mock_save.assert_called()

    # 14. Trace ID propagation
    async def test_14_trace_id_appears_in_logs(self):
        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             self.assertLogs("app.conversation.manager", level="INFO") as logs:
            await manager.handle_message(
                "441111111111", "What is KYC?", "trace-xyz-123", llm_fallback=_fake_llm_fallback
            )
        joined = "\n".join(logs.output)
        self.assertIn("trace-xyz-123", joined)
        self.assertIn("conversation.turn.started", joined)
        self.assertIn("conversation.turn.completed", joined)

    # 15. Sensitive information is not logged or stored
    async def test_15_sensitive_information_not_logged_or_stored(self):
        manager, wf = _manager_with()
        context = _fresh_context()
        p1, p2, _p3, p4 = self._patches(context=context)
        with p1, p2, p4, \
             patch("app.conversation.manager.get_workflow", return_value={
                 "type": "onboarding", "step": "COLLECT_PAN", "workflow_id": "wf-2",
                 "data": {"pan_number": "ABCDE1234F", "otp": "123456"},
             }), \
             patch.object(manager.context_store, "save", return_value=True), \
             self.assertLogs("app.conversation.manager", level="INFO") as logs:
            await manager.handle_message(
                "441111111111", "ABCDE1234F", "t15", llm_fallback=_fake_llm_fallback
            )
        self.assertNotIn("pan_number", context.workflow_data)
        self.assertNotIn("otp", context.workflow_data)
        joined = "\n".join(logs.output)
        self.assertNotIn("ABCDE1234F", joined)
        self.assertNotIn("123456", joined)


class LanguageStickinessTests(unittest.IsolatedAsyncioTestCase):
    """Language changes only on a real non-ASCII detection or an explicit
    meta-request — a later plain-ASCII message (a bare "yes", a number)
    must NOT silently reset the conversation back to English. See
    ConversationManager._update_language."""

    def setUp(self):
        self.manager, _ = _manager_with()

    async def test_hindi_stays_sticky_through_a_bare_ascii_reply(self):
        context = _fresh_context()
        with patch("app.conversation.manager.detect_language", return_value="hi"):
            await self.manager._update_language(context, "मेरा बैलेंस क्या है", None, False, "t")
        self.assertEqual(context.detected_language, "hi")

        # A later plain-ASCII, non-trivial-length message must not reset it.
        with patch("app.conversation.manager.detect_language") as mock_detect:
            await self.manager._update_language(context, "5000", None, False, "t")
        mock_detect.assert_not_called()
        self.assertEqual(context.detected_language, "hi")

    async def test_explicit_request_switches_language(self):
        context = _fresh_context()
        context.text_language = "hi"
        await self.manager._update_language(context, "reply in English please", None, False, "t")
        self.assertEqual(context.detected_language, "en")
        self.assertEqual(context.text_language, "en")

    async def test_plain_english_conversation_never_calls_detection(self):
        context = _fresh_context()
        with patch("app.conversation.manager.detect_language") as mock_detect:
            await self.manager._update_language(context, "check my balance please", None, False, "t")
        mock_detect.assert_not_called()
        self.assertEqual(context.detected_language, "en")

    async def test_voice_hint_always_wins_on_a_voice_turn(self):
        context = _fresh_context()
        context.voice_language = "hi"
        await self.manager._update_language(context, "5000", "es", True, "t")
        self.assertEqual(context.detected_language, "es")
        self.assertEqual(context.voice_language, "es")

    async def test_short_voice_reply_with_no_own_signal_stays_on_voice_language(self):
        # Sarvam drops its own per-utterance tag for very short audio (see
        # transcribe_audio) — detected_language arrives as None even
        # though this is still a voice turn. It must stay on the voice
        # channel's own sticky language, not fall through to text.
        context = _fresh_context()
        context.voice_language = "hi"
        context.text_language = "en"
        await self.manager._update_language(context, "yes", None, True, "t")
        self.assertEqual(context.detected_language, "hi")
        self.assertEqual(context.voice_language, "hi")
        self.assertEqual(context.text_language, "en")

    async def test_voice_language_never_leaks_into_a_later_text_reply(self):
        # Requirement: "Language Separation" — a language established on
        # a voice call must not carry over to a text message, even a
        # short one that carries no signal of its own.
        context = _fresh_context()
        await self.manager._update_language(context, "मेरा बैलेंस बताओ", "hi", True, "t")
        self.assertEqual(context.detected_language, "hi")

        await self.manager._update_language(context, "ok", None, False, "t")
        self.assertEqual(context.detected_language, "en")
        self.assertEqual(context.text_language, "en")
        self.assertEqual(context.voice_language, "hi")  # untouched

    async def test_text_language_never_leaks_into_a_later_voice_reply(self):
        context = _fresh_context()
        with patch("app.conversation.manager.detect_language", return_value="ta"):
            await self.manager._update_language(context, "என் இருப்பு என்ன", None, False, "t")
        self.assertEqual(context.detected_language, "ta")

        # A voice turn with its own STT-detected language must win
        # immediately, ignoring the text channel's Tamil entirely.
        await self.manager._update_language(context, "how much do I have", "en", True, "t")
        self.assertEqual(context.detected_language, "en")
        self.assertEqual(context.voice_language, "en")
        self.assertEqual(context.text_language, "ta")  # untouched


class ConversationManagerArchitectureBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Verifies the orchestrator/business-logic boundary the task requires."""

    def _patches(self, context=None):
        context = context if context is not None else _fresh_context()
        return (
            patch("app.conversation.manager.build_context", return_value=context),
            patch("app.conversation.manager.check_registration_gate", return_value=None),
            patch("app.conversation.manager.get_workflow", return_value=None),
            patch("app.conversation.manager.append_to_session"),
        )

    async def test_active_workflow_remains_authoritative_over_router(self):
        """Even an intent that would start a *different* workflow must not
        override an already-active one — WorkflowManager.handle() runs
        first and, if it handles the turn, the router is never consulted."""
        manager, wf = _manager_with(handle_result={"handled": True, "response": "continuing transfer"})
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.route_intent") as mock_route:
            response = await manager.handle_message(
                "441111111111", "I want a personal loan", "tb1", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "continuing transfer")
        mock_route.assert_not_called()

    async def test_manager_has_no_direct_database_import(self):
        """ConversationManager must orchestrate, not execute — it should
        never import app.database directly."""
        import inspect

        import app.conversation.manager as manager_module

        source = inspect.getsource(manager_module)
        self.assertNotIn("import app.database", source)
        self.assertNotIn("from app.database", source)
        self.assertNotIn("psycopg2", source)

    async def test_router_decision_carries_no_execution_capability(self):
        """RoutingDecision is a plain data object — it cannot itself call a
        banking tool or touch the database (see app/conversation/router.py)."""
        from app.conversation.router import RoutingDecision

        decision = RoutingDecision(action="START_WORKFLOW", workflow="transfer")
        for forbidden in ("execute", "commit", "run", "call"):
            self.assertFalse(hasattr(decision, forbidden))

    async def test_response_templates_are_used_for_out_of_scope(self):
        """The out-of-scope response text must come from the centralized
        template, not an inline string in the manager."""
        from app.conversation.responses.common import render_out_of_scope

        manager, wf = _manager_with()
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "441111111111", "Tell me a joke", "tb2", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, render_out_of_scope())


class ConversationManagerMenuTapTests(unittest.IsolatedAsyncioTestCase):
    """A tapped main-menu row (e.g. WhatsApp list_reply id "2") arrives as a
    bare digit with no active workflow — the intent classifier can easily
    misread that as OUT_OF_SCOPE or CLARIFICATION_REQUIRED since a lone
    digit carries no banking vocabulary. WorkflowManager.start_requested()'s
    digit map already knows exactly what "2" means; the manager must give it
    a chance before giving up on an unambiguous menu tap."""

    def _patches(self, context=None):
        context = context if context is not None else _fresh_context()
        return (
            patch("app.conversation.manager.build_context", return_value=context),
            patch("app.conversation.manager.check_registration_gate", return_value=None),
            patch("app.conversation.manager.get_workflow", return_value=None),
            patch("app.conversation.manager.append_to_session"),
        )

    async def test_out_of_scope_digit_falls_back_to_start_requested(self):
        """Row "1" (Transfer money) tapped while the classifier reports
        OUT_OF_SCOPE must still start the transfer workflow, not tell the
        customer their menu tap was out of scope."""
        from app.conversation.router import RoutingDecision

        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "transfer-started"}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.route_intent", return_value=RoutingDecision(action="OUT_OF_SCOPE")), \
             patch.object(ConversationManager, "_classify_intent", return_value=IntentResult(intent="unknown", confidence=0.0)):
            response = await manager.handle_message(
                "441111111111", "1", "tm1", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "transfer-started")
        self.assertIn(("441111111111", "1"), wf.start_requested_calls)

    async def test_clarification_required_digit_falls_back_to_start_requested(self):
        """Same protection for CLARIFICATION_REQUIRED (a low-confidence
        guess), which a bare menu-row digit can also land on."""
        from app.conversation.router import RoutingDecision

        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "kyc-started"}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.route_intent", return_value=RoutingDecision(action="CLARIFICATION_REQUIRED", reason="kyc_update_request")), \
             patch.object(ConversationManager, "_classify_intent", return_value=IntentResult(intent="unknown", confidence=0.0)):
            response = await manager.handle_message(
                "441111111111", "7", "tm2", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "kyc-started")

    async def test_out_of_scope_digit_with_no_dedicated_workflow_reaches_llm(self):
        """Rows like "2" (Check balance) have no dedicated workflow of their
        own — start_requested() resolves them to a reprocess_query instead
        of handled=True. That query must still reach the LLM+tools path
        rather than being told it's out of scope."""
        from app.conversation.router import RoutingDecision

        manager, wf = _manager_with(
            start_requested_result={"handled": False, "response": None, "reprocess_query": "check my balance"}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.route_intent", return_value=RoutingDecision(action="OUT_OF_SCOPE")), \
             patch.object(ConversationManager, "_classify_intent", return_value=IntentResult(intent="unknown", confidence=0.0)):
            response = await manager.handle_message(
                "441111111111", "2", "tm3", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "llm-answer:check my balance")

    async def test_out_of_scope_non_digit_is_unaffected(self):
        """The deterministic fallback must only trigger for an exact menu
        digit — an ordinary out-of-scope message keeps its existing
        behavior untouched."""
        from app.conversation.responses.common import render_out_of_scope
        from app.conversation.router import RoutingDecision

        manager, wf = _manager_with(
            start_requested_result={"handled": True, "response": "should-not-be-used"}
        )
        p1, p2, p3, p4 = self._patches()
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.route_intent", return_value=RoutingDecision(action="OUT_OF_SCOPE")), \
             patch.object(ConversationManager, "_classify_intent", return_value=IntentResult(intent="unknown", confidence=0.0)):
            response = await manager.handle_message(
                "441111111111", "Tell me a joke", "tm4", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, render_out_of_scope())
        self.assertEqual(wf.start_requested_calls, [])


if __name__ == "__main__":
    unittest.main()

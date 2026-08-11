"""Tests for Task 10 — Response Renderer, templates, and response UX.

Follows the project's unittest convention (no pytest installed here).
"""

import unittest
from unittest.mock import AsyncMock, patch

from app.conversation.renderer import (
    InteractiveButton,
    InteractiveListRow,
    InteractiveListSection,
    ResponseKind,
    StructuredResponse,
    as_structured_response,
    render_and_send,
)  # as_structured_response also used by LlmFallbackWorkflowTests below
from app.conversation.responses.common import (
    WORKFLOW_STEP_HINTS,
    render_main_menu_list,
    render_workflow_boundary,
    render_workflow_boundary_with_step,
    render_workflow_step_hint,
)
from app.workflows.processors.onboarding import account_type_list_prompt
from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_UPLOAD_CHEQUE,
    WORKFLOW_CHEQUE,
    WORKFLOW_ONBOARDING,
)
from app.workflows.manager import WorkflowManager
from app.workflows.memory import create_workflow, create_workflow_model


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis, matching the pattern
    already used in tests/test_idempotency.py — no live Redis required."""

    def __init__(self):
        self._store = {}

    def setex(self, key, ttl, value):
        self._store[key] = value
        return True

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._store:
            return None
        self._store[key] = value
        return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, key):
        return 1 if self._store.pop(key, None) is not None else 0


# ─── Part 1: Response Renderer ──────────────────────────────────────────

class ResponseRendererTests(unittest.IsolatedAsyncioTestCase):
    def test_plain_string_normalizes_to_text_kind(self):
        structured = as_structured_response("Hello")
        self.assertEqual(structured.kind, ResponseKind.TEXT)
        self.assertEqual(structured.text, "Hello")

    def test_structured_response_passthrough(self):
        original = StructuredResponse.template("Hi there", template_name="render_greeting")
        self.assertIs(as_structured_response(original), original)

    async def test_render_and_send_text(self):
        with patch("app.conversation.renderer.send_text_message", new=AsyncMock(return_value=True)) as mock_send:
            result = await render_and_send("Hello", "447700900000", "trace-1")
        self.assertTrue(result)
        mock_send.assert_awaited_once_with("447700900000", "Hello", "trace-1")

    async def test_render_and_send_template(self):
        response = StructuredResponse.template("Hi Alex!", template_name="render_greeting")
        with patch("app.conversation.renderer.send_text_message", new=AsyncMock(return_value=True)) as mock_send:
            result = await render_and_send(response, "447700900000", "trace-2")
        self.assertTrue(result)
        mock_send.assert_awaited_once_with("447700900000", "Hi Alex!", "trace-2")

    async def test_render_and_send_failure_returns_false_not_raise(self):
        with patch("app.conversation.renderer.send_text_message", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await render_and_send("Hello", "447700900000", "trace-3")
        self.assertFalse(result)

    async def test_render_and_send_buttons(self):
        response = StructuredResponse.buttons_of(
            "Ready to send this?",
            [InteractiveButton(id="1", title="Yes, send it"), InteractiveButton(id="2", title="Edit amount")],
        )
        with patch("app.conversation.renderer.send_button_message", new=AsyncMock(return_value=True)) as mock_send:
            result = await render_and_send(response, "447700900000", "trace-5")
        self.assertTrue(result)
        mock_send.assert_awaited_once_with(
            "447700900000", "Ready to send this?",
            [{"id": "1", "title": "Yes, send it"}, {"id": "2", "title": "Edit amount"}],
            "trace-5",
        )

    async def test_render_and_send_list(self):
        section = InteractiveListSection(title="Loan types", rows=[
            InteractiveListRow(id="1", title="Personal Loan"),
            InteractiveListRow(id="2", title="Home Loan"),
        ])
        response = StructuredResponse.list_of("Choose a loan type", "Choose", [section])
        with patch("app.conversation.renderer.send_list_message", new=AsyncMock(return_value=True)) as mock_send:
            result = await render_and_send(response, "447700900000", "trace-6")
        self.assertTrue(result)
        mock_send.assert_awaited_once()
        args = mock_send.call_args.args
        self.assertEqual(args[0], "447700900000")
        self.assertEqual(args[1], "Choose a loan type")
        self.assertEqual(args[2], "Choose")

    async def test_too_many_buttons_falls_back_to_text(self):
        response = StructuredResponse.buttons_of(
            "Pick one",
            [InteractiveButton(id=str(i), title=f"Option {i}") for i in range(1, 5)],
        )
        with patch("app.conversation.renderer.send_button_message", new=AsyncMock(return_value=True)) as mock_buttons, \
             patch("app.conversation.renderer.send_text_message", new=AsyncMock(return_value=True)) as mock_text:
            result = await render_and_send(response, "447700900000", "trace-7")
        self.assertTrue(result)
        mock_buttons.assert_not_called()
        mock_text.assert_awaited_once()
        sent_text = mock_text.call_args.args[1]
        self.assertIn("Pick one", sent_text)
        self.assertIn("Option 1", sent_text)

    async def test_too_many_list_rows_falls_back_to_text(self):
        section = InteractiveListSection(
            title="Accounts", rows=[InteractiveListRow(id=str(i), title=f"Account {i}") for i in range(1, 12)]
        )
        response = StructuredResponse.list_of("Choose an account", "Choose", [section])
        with patch("app.conversation.renderer.send_list_message", new=AsyncMock(return_value=True)) as mock_list, \
             patch("app.conversation.renderer.send_text_message", new=AsyncMock(return_value=True)) as mock_text:
            result = await render_and_send(response, "447700900000", "trace-8")
        self.assertTrue(result)
        mock_list.assert_not_called()
        mock_text.assert_awaited_once()

    async def test_interactive_send_exception_falls_back_to_text(self):
        response = StructuredResponse.buttons_of(
            "Confirm?", [InteractiveButton(id="yes", title="Yes"), InteractiveButton(id="no", title="No")]
        )
        with patch("app.conversation.renderer.send_button_message", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.conversation.renderer.send_text_message", new=AsyncMock(return_value=True)) as mock_text:
            result = await render_and_send(response, "447700900000", "trace-9")
        self.assertTrue(result)
        mock_text.assert_awaited_once()

    async def test_render_and_send_delivery_declined(self):
        with patch("app.conversation.renderer.send_text_message", new=AsyncMock(return_value=False)):
            result = await render_and_send("Hello", "447700900000", "trace-4")
        self.assertFalse(result)

    def test_renderer_hides_openwa_details(self):
        # StructuredResponse never carries a chat id, session id, or any
        # OpenWA-shaped payload — only text + provenance metadata, plus
        # the interactive button/list shape (still just ids/titles, no
        # transport-level detail).
        fields = set(StructuredResponse.model_fields.keys())
        self.assertEqual(
            fields,
            {"kind", "text", "template_name", "buttons", "list_button_label", "list_sections"},
        )


# ─── Part 2/3: reusable templates ───────────────────────────────────────

class WorkflowBoundaryTemplateTests(unittest.TestCase):
    def test_known_step_produces_specific_hint_not_generic_boundary(self):
        text = render_workflow_boundary_with_step(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        self.assertIn("upload a clear image of the cheque", text.lower())
        self.assertIn("cancel", text.lower())
        # The old rigid phrasing must be gone.
        self.assertNotIn("i can answer questions only about this request", text.lower())

    def test_onboarding_aadhaar_step_hint(self):
        text = render_workflow_boundary_with_step(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        self.assertIn("aadhaar", text.lower())
        self.assertIn("cancel", text.lower())

    def test_unknown_step_falls_back_to_generic_boundary(self):
        text = render_workflow_boundary_with_step(WORKFLOW_CHEQUE, "NOT_A_REAL_STEP")
        self.assertEqual(text, render_workflow_boundary("depositing a cheque"))

    def test_every_hint_is_short_and_senior_friendly(self):
        for (_workflow, _step), hint in WORKFLOW_STEP_HINTS.items():
            self.assertLess(len(hint), 200)
            self.assertNotIn("null", hint.lower())
            self.assertNotIn("error", hint.lower())

    def test_step_hint_never_contains_sensitive_terms(self):
        forbidden = ("aadhaar number", "pan number", "otp", "password", "cvv")
        for hint in WORKFLOW_STEP_HINTS.values():
            lowered = hint.lower()
            for term in forbidden:
                self.assertNotIn(term, lowered)

    def test_missing_hint_returns_none_not_a_crash(self):
        self.assertIsNone(render_workflow_step_hint("transfer", "NOT_A_STEP"))


# ─── Part 9/10: active-workflow guidance + out-of-context handling ─────

class WorkflowManagerBoundaryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Drives the real WorkflowManager.handle() (no mocking of the
    boundary-message logic itself) against a real Redis-backed workflow,
    matching how test_cheque_processor.py etc. already test this layer."""

    def setUp(self):
        self.manager = WorkflowManager()
        self.phone = "447700900099"
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_out_of_context_question_during_onboarding_stays_on_topic(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        result = await self.manager.handle(self.phone, "Why is the sky blue?", trace_id="t1")

        self.assertTrue(result["handled"])
        response = result["response"].lower()
        self.assertNotIn("rayleigh", response)  # never answers the science question
        self.assertIn("aadhaar", response)
        self.assertIn("cancel", response)

    async def test_active_cheque_workflow_what_should_i_do_explains_current_step(self):
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        result = await self.manager.handle(self.phone, "What should I do?", trace_id="t2")

        self.assertTrue(result["handled"])
        response = result["response"].lower()
        self.assertIn("upload a clear image of the cheque", response)
        # Must not restart the workflow or dump the main menu.
        self.assertNotIn("what would you like to do?", response)
        self.assertNotIn("1. \U0001f4b8", result["response"])

    async def test_workflow_is_not_restarted_by_out_of_context_question(self):
        from app.workflows.memory import get_workflow

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)
        workflow_id_before = get_workflow(self.phone)["workflow_id"]

        await self.manager.handle(self.phone, "What should I do?", trace_id="t3")

        workflow_after = get_workflow(self.phone)
        self.assertIsNotNone(workflow_after)
        self.assertEqual(workflow_after["workflow_id"], workflow_id_before)
        self.assertEqual(workflow_after["step"], STEP_UPLOAD_CHEQUE)

    async def test_cross_topic_banking_question_mid_transfer_is_reprocessed_not_blocked(self):
        # A genuine banking question about a DIFFERENT topic than the
        # active workflow must be answered, not rejected with the rigid
        # boundary message — this is what let a real loan-interest
        # question during a transfer get stuck before this fix.
        from app.workflows.constants import STEP_SELECT_BENEFICIARY, WORKFLOW_TRANSFER

        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY)
        create_workflow(self.phone, workflow)

        result = await self.manager.handle(
            self.phone, "What's the interest rate on a personal loan?", trace_id="t4"
        )

        # handled=False + reprocess_query means "let the router/LLM answer
        # this", not the workflow processor and not the boundary message.
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), "What's the interest rate on a personal loan?")

    async def test_cross_topic_question_does_not_disturb_the_active_workflow(self):
        from app.workflows.constants import STEP_SELECT_BENEFICIARY, WORKFLOW_TRANSFER
        from app.workflows.memory import get_workflow

        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY)
        create_workflow(self.phone, workflow)
        workflow_id_before = get_workflow(self.phone)["workflow_id"]

        await self.manager.handle(self.phone, "What's the interest rate on a personal loan?", trace_id="t5")

        workflow_after = get_workflow(self.phone)
        self.assertIsNotNone(workflow_after)
        self.assertEqual(workflow_after["workflow_id"], workflow_id_before)
        self.assertEqual(workflow_after["step"], STEP_SELECT_BENEFICIARY)

    async def test_real_workflow_field_input_is_not_diverted_by_widened_rule(self):
        # Guard against the widening above accidentally catching ordinary
        # field input that happens to contain a banking word ("send 500"
        # is not a question, so it must still reach the transfer
        # processor, not get reprocessed as if it were a question).
        from app.workflows.constants import STEP_SELECT_BENEFICIARY, WORKFLOW_TRANSFER

        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY)
        create_workflow(self.phone, workflow)

        with patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result = await self.manager.handle(self.phone, "send 500", trace_id="t6")

        self.assertNotEqual(result.get("reprocess_query"), "send 500")


# ─── LLM-fallback behaviors (side-question resume, workflow jump) ───────

class LlmFallbackWorkflowTests(unittest.IsolatedAsyncioTestCase):
    """These behaviors only activate when LLM_FALLBACK_ENABLED is set —
    with it off (the default, exercised by every other test in this file),
    behavior is unchanged from before these features existed."""

    def setUp(self):
        self.manager = WorkflowManager()
        self.phone = "447700900097"
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        env_patcher = patch.dict("os.environ", {"LLM_FALLBACK_ENABLED": "true"})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    async def test_side_question_answered_and_step_resumed_in_one_turn(self):
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch(
            "app.workflows.manager.answer_side_question",
            return_value="Your interest rate depends on the loan type.",
        ):
            result = await self.manager.handle(
                self.phone, "What's the interest rate on a personal loan?", trace_id="t1"
            )

        self.assertTrue(result["handled"])
        response = result["response"].lower()
        self.assertIn("interest rate", response)
        self.assertIn("upload a clear image of the cheque", response)

    async def test_side_question_resume_does_not_change_workflow_state(self):
        from app.workflows.memory import get_workflow

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)
        workflow_id_before = get_workflow(self.phone)["workflow_id"]

        with patch("app.workflows.manager.answer_side_question", return_value="Some answer."):
            await self.manager.handle(self.phone, "What's the interest rate on a loan?", trace_id="t2")

        after = get_workflow(self.phone)
        self.assertEqual(after["workflow_id"], workflow_id_before)
        self.assertEqual(after["step"], STEP_UPLOAD_CHEQUE)

    async def test_llm_answer_failure_falls_back_to_reprocess_query(self):
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch("app.workflows.manager.answer_side_question", return_value=None):
            result = await self.manager.handle(
                self.phone, "What's the interest rate on a personal loan?", trace_id="t3"
            )

        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), "What's the interest rate on a personal loan?")

    async def test_confident_workflow_jump_asks_confirmation_first(self):
        from app.services.llm_understanding import JumpTarget

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch(
            "app.workflows.manager.detect_step_or_workflow_jump",
            return_value=JumpTarget(target_workflow="loan", confidence=0.9),
        ):
            result = await self.manager.handle(self.phone, "actually let me apply for a loan instead", trace_id="t4")

        self.assertTrue(result["handled"])
        # The confirmation prompt is now a StructuredResponse with tap-to-
        # reply Continue/Switch buttons (see app/conversation/renderer.py).
        response = as_structured_response(result["response"])
        self.assertIn("switch", response.text.lower())
        self.assertEqual({b.id for b in response.buttons}, {"continue", "switch"})
        from app.workflows.memory import get_workflow
        # Not jumped yet — still the original workflow, now awaiting confirmation.
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_CHEQUE)
        self.assertTrue(get_workflow(self.phone)["data"].get("pending_stop_confirmation"))

    async def test_confirming_the_jump_starts_the_target_workflow(self):
        from app.services.llm_understanding import JumpTarget
        from app.workflows.memory import get_workflow

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch(
            "app.workflows.manager.detect_step_or_workflow_jump",
            return_value=JumpTarget(target_workflow="loan", confidence=0.9),
        ):
            await self.manager.handle(self.phone, "actually let me apply for a loan instead", trace_id="t5")

        result = await self.manager.handle(self.phone, "switch", trace_id="t6")

        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], "loan")

    async def test_declining_the_jump_resumes_the_original_workflow(self):
        from app.services.llm_understanding import JumpTarget
        from app.workflows.memory import get_workflow

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch(
            "app.workflows.manager.detect_step_or_workflow_jump",
            return_value=JumpTarget(target_workflow="loan", confidence=0.9),
        ):
            await self.manager.handle(self.phone, "actually let me apply for a loan instead", trace_id="t7")

        result = await self.manager.handle(self.phone, "continue", trace_id="t8")

        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_CHEQUE)
        self.assertEqual(get_workflow(self.phone)["step"], STEP_UPLOAD_CHEQUE)


# ─── Interactive list conversions (menu-style prompts) ──────────────────

class InteractiveListConversionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = WorkflowManager()
        self.phone = "447700900096"
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_loan_start_returns_list_with_four_types(self):
        result = self.manager.start_requested(self.phone, "I want a loan", trace_id="t1")
        self.assertTrue(result["handled"])
        response = as_structured_response(result["response"])
        self.assertEqual(response.kind, ResponseKind.LIST)
        rows = [row for section in response.list_sections for row in section.rows]
        self.assertEqual({row.id for row in rows}, {"1", "2", "3", "4"})
        self.assertEqual({row.title for row in rows}, {"Personal Loan", "Home Loan", "Vehicle Loan", "Education Loan"})

    async def test_tapped_loan_type_row_id_advances_the_workflow(self):
        from app.workflows.constants import STEP_UPLOAD_LOAN_FORM

        self.manager.start_requested(self.phone, "I want a loan", trace_id="t2")
        result = await self.manager.handle(self.phone, "2", trace_id="t3")
        self.assertTrue(result["handled"])
        from app.workflows.memory import get_workflow
        workflow = get_workflow(self.phone)
        self.assertEqual(workflow["step"], STEP_UPLOAD_LOAN_FORM)
        self.assertEqual(workflow["data"]["loan_type"], "home")

    async def test_transfer_beneficiary_list_row_ids_are_digits_plus_new(self):
        from app.workflows.constants import STEP_SELECT_BENEFICIARY, WORKFLOW_TRANSFER
        from app.workflows.memory import create_workflow, create_workflow_model

        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY)
        create_workflow(self.phone, workflow)
        beneficiaries = [
            {"beneficiary_name": "Priya", "account_number": "GB12FNCL00010001234567"},
            {"beneficiary_name": "Amit", "account_number": "GB12FNCL00010009999999"},
        ]
        with patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=beneficiaries):
            result = self.manager.transfer_handler._beneficiary_prompt(self.phone)
        response = as_structured_response(result["response"])
        self.assertEqual(response.kind, ResponseKind.LIST)
        rows = [row for section in response.list_sections for row in section.rows]
        self.assertEqual([row.id for row in rows], ["1", "2", "new"])

    async def test_beneficiary_list_falls_back_to_text_beyond_ten_rows(self):
        beneficiaries = [
            {"beneficiary_name": f"Person {i}", "account_number": f"GB12FNCL0001000{i:07d}"} for i in range(10)
        ]
        with patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=beneficiaries):
            result = self.manager.transfer_handler._beneficiary_prompt(self.phone)
        # 10 beneficiaries + 1 "add new" row = 11 > WhatsApp's 10-row cap.
        self.assertIsInstance(result["response"], str)

    async def test_main_menu_list_has_seven_rows_with_expected_ids(self):
        response = render_main_menu_list("Alex", greeting=False)
        self.assertEqual(response.kind, ResponseKind.LIST)
        rows = [row for section in response.list_sections for row in section.rows]
        self.assertEqual([row.id for row in rows], ["1", "2", "3", "4", "5", "6", "7"])

    async def test_onboarding_account_type_list_row_ids_match_aliases(self):
        response = account_type_list_prompt("Which account?")
        rows = [row for section in response.list_sections for row in section.rows]
        self.assertEqual([row.id for row in rows], ["1", "2", "3"])
        self.assertEqual([row.title for row in rows], ["Savings Account", "Current Account", "Salary Account"])


# ─── Part 13: Back / Cancel ─────────────────────────────────────────────

class NavigationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = WorkflowManager()
        self.phone = "447700900098"
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_cancel_confirms_nothing_was_submitted(self):
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result = await self.manager.handle(self.phone, "Cancel", trace_id="t1")

        self.assertTrue(result["handled"])
        self.assertIn("cancelled", result["response"].lower())
        self.assertIn("nothing was submitted", result["response"].lower())

    async def test_cancel_clears_the_workflow(self):
        from app.workflows.memory import get_workflow

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)

        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            await self.manager.handle(self.phone, "Cancel", trace_id="t2")

        self.assertIsNone(get_workflow(self.phone))

    async def test_back_moves_to_previous_step_without_losing_workflow(self):
        from app.workflows.memory import get_workflow
        from app.workflows.constants import STEP_UPLOAD_LOAN_FORM, STEP_SELECT_LOAN_TYPE

        workflow = create_workflow_model("loan", STEP_UPLOAD_LOAN_FORM)
        create_workflow(self.phone, workflow)

        result = await self.manager.handle(self.phone, "Back", trace_id="t3")

        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["step"], STEP_SELECT_LOAN_TYPE)


# ─── Part 4: natural language must remain primary ──────────────────────

class NaturalLanguageTests(unittest.TestCase):
    """These exact phrasings, from Task 10 Part 4, must resolve to the
    right intent/workflow without forcing a menu reply."""

    def test_transfer_request(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("I want to transfer money.")
        self.assertEqual(result.intent, "transfer_request")

    def test_transfer_with_amount_and_beneficiary(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("Transfer 500 to Priya.")
        self.assertEqual(result.intent, "transfer_request")

    def test_balance_request(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("Show my balance.")
        self.assertEqual(result.intent, "balance_request")

    def test_transaction_insight_request(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("What did I spend this month?")
        self.assertEqual(result.intent, "transaction_insight_question")

    def test_spend_question_is_not_misread_as_cancel(self):
        # Regression: "spend" contains "end", and "this" is a common word —
        # workflows/manager.py's cancel-detection used to substring-match
        # both and treat this ordinary question as a cancel command.
        from app.workflows.manager import _is_cancel_command
        self.assertFalse(_is_cancel_command("What did I spend this month?"))

    def test_cheque_deposit_request(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("I want to deposit a cheque.")
        self.assertEqual(result.intent, "cheque_deposit_request")

    def test_loan_interest_question_never_starts_a_loan_application(self):
        # Regression: a real user asked "What's the loan intrest amount
        # charged" and the classifier's loan branch (unlike its sibling
        # kyc_update_request) had no question-guard, so it was classified
        # as loan_application_request at high confidence and the router
        # actually STARTED a real loan workflow from a plain question.
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.router import route_intent

        result = classify_intent("What's the loan intrest amount charged")
        self.assertNotEqual(result.intent, "loan_application_request")
        decision = route_intent(result)
        self.assertNotEqual(decision.action, "START_WORKFLOW")

    def test_transfer_limit_question_never_starts_a_transfer(self):
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.router import route_intent

        result = classify_intent("What is the transfer limit?")
        self.assertNotEqual(result.intent, "transfer_request")
        decision = route_intent(result)
        self.assertNotEqual(decision.action, "START_WORKFLOW")

    def test_cheque_deposit_question_never_starts_a_cheque_workflow(self):
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.router import route_intent

        result = classify_intent("Can I deposit a cheque online?")
        self.assertNotEqual(result.intent, "cheque_deposit_request")
        decision = route_intent(result)
        self.assertNotEqual(decision.action, "START_WORKFLOW")

    def test_genuine_loan_request_still_starts_workflow(self):
        # The question-guard fix above must not affect real, non-question
        # action requests.
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.router import route_intent

        result = classify_intent("I want a personal loan")
        self.assertEqual(result.intent, "loan_application_request")
        decision = route_intent(result)
        self.assertEqual(decision.action, "START_WORKFLOW")

    def test_genuine_transfer_request_still_starts_workflow(self):
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.router import route_intent

        result = classify_intent("Transfer 500 to Priya")
        self.assertEqual(result.intent, "transfer_request")
        decision = route_intent(result)
        self.assertEqual(decision.action, "START_WORKFLOW")

    def test_cheque_status_request(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("Check my cheque.")
        self.assertEqual(result.intent, "cheque_status_request")

    def test_loan_eligibility_natural_language(self):
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.guidance.policy import build_guidance

        text = "I earn 50000 per month and want a personal loan."
        result = classify_intent(text)
        self.assertEqual(result.intent, "loan_eligibility_question")
        guidance = build_guidance(text, result)
        self.assertEqual(guidance.entities.get("monthly_income"), 50000)
        self.assertEqual(guidance.entities.get("loan_type"), "personal")

    def test_kyc_update_request(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("I want to update my KYC.")
        self.assertEqual(result.intent, "kyc_update_request")

    def test_kyc_question(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("What is KYC?")
        self.assertEqual(result.intent, "kyc_question")

    def test_cancel_natural_language(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("Cancel.")
        self.assertEqual(result.intent, "cancel")

    def test_back_natural_language(self):
        from app.conversation.intent.classifier import classify_intent
        result = classify_intent("Go back.")
        self.assertEqual(result.intent, "back")

    def test_no_eligibility_or_approval_claim_in_loan_guidance(self):
        from app.conversation.intent.classifier import classify_intent
        from app.conversation.guidance.policy import build_guidance
        from app.conversation.guidance.responses import render_guidance

        text = "I earn 50000 per month and want a personal loan."
        intent_result = classify_intent(text)
        guidance = build_guidance(text, intent_result)
        rendered = render_guidance(guidance)
        lowered = rendered.text.lower()
        self.assertNotIn("you are eligible", lowered)
        self.assertNotIn("you will get", lowered)
        self.assertNotIn("your loan is approved", lowered)


if __name__ == "__main__":
    unittest.main()

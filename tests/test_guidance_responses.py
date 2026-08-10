"""Tests for Task 9.2 — Banking Guidance responses and action handoffs.

Three levels, matching the layers introduced this task:
  1. Rendering — classify real text -> build_guidance() -> render_guidance()
     and check the user-facing text/actions (no manager involved).
  2. Handoff resolution — resolve_pending_action() on a raw reply against
     an offered action list (no manager involved).
  3. ConversationManager integration — a full turn through
     ConversationManager.handle_message(), proving the guidance
     interception and action-handoff wiring actually work end to end,
     with WorkflowManager/start_workflow_directly mocked (no live
     Redis/Postgres needed) — same pattern as test_conversation_manager.py.

Follows the project's unittest convention (no pytest installed here).
"""

import unittest
from unittest.mock import AsyncMock, patch

from app.conversation.context import ConversationContext
from app.conversation.guidance.handoff import resolve_pending_action
from app.conversation.guidance.models import GuidanceAction, GuidanceType
from app.conversation.guidance.policy import build_guidance
from app.conversation.guidance.responses import render_guidance
from app.conversation.intent.classifier import classify_intent
from app.conversation.manager import ConversationManager
from app.workflows.constants import STEP_UPLOAD_CHEQUE, WORKFLOW_CHEQUE


def _render(text, context=None):
    intent_result = classify_intent(text, context=context)
    guidance = build_guidance(text, intent_result, context)
    rendered = render_guidance(guidance, context) if guidance else None
    return intent_result, guidance, rendered


# ─── 1-8: rendering ──────────────────────────────────────────────────────

class GuidanceRenderingTests(unittest.TestCase):
    def test_01_loan_eligibility_guidance_text(self):
        _, guidance, rendered = _render("I earn 50000 a month and want a personal loan")

        self.assertEqual(guidance.guidance_type, GuidanceType.LOAN_ELIGIBILITY_GUIDANCE)
        self.assertIn("50,000", rendered.text)
        self.assertIn("Personal", rendered.text)
        self.assertIn(GuidanceAction.START_LOAN_APPLICATION, rendered.actions)
        self.assertEqual(rendered.primary_action, GuidanceAction.START_LOAN_APPLICATION)
        lowered = rendered.text.lower()
        self.assertNotIn("you are eligible", lowered)
        self.assertNotIn("you're eligible", lowered)
        self.assertNotIn("approved", lowered)

    def test_02_loan_document_question_guidance_not_workflow_start(self):
        intent_result, guidance, rendered = _render("What documents do I need for a personal loan?")

        # A question-phrased loan message now classifies directly as
        # loan_question (the classifier itself gained a question-guard —
        # see docs/current_architecture.md — matching kyc_update_request's
        # existing one), rather than needing the guidance-layer carve-out
        # to redirect a mis-tagged loan_application_request. Either way,
        # the guidance result itself is unchanged and correct.
        self.assertEqual(intent_result.intent, "loan_question")
        self.assertEqual(guidance.guidance_type, GuidanceType.LOAN_DOCUMENT_GUIDANCE)
        self.assertIn(GuidanceAction.START_LOAN_APPLICATION, rendered.actions)

    def test_03_transfer_guidance_how_do_i(self):
        _, guidance, rendered = _render("How do I transfer money?")

        self.assertEqual(guidance.guidance_type, GuidanceType.TRANSFER_GUIDANCE)
        self.assertIn(GuidanceAction.START_TRANSFER, rendered.actions)
        self.assertIn("confirm", rendered.text.lower())

    def test_04_cheque_deposit_guidance_how_do_i(self):
        _, guidance, rendered = _render("How do I deposit a cheque?")

        self.assertEqual(guidance.guidance_type, GuidanceType.CHEQUE_GUIDANCE)
        self.assertIn(GuidanceAction.START_CHEQUE_DEPOSIT, rendered.actions)

    def test_05_kyc_guidance(self):
        _, guidance, rendered = _render("What is KYC?")

        self.assertEqual(guidance.guidance_type, GuidanceType.KYC_GUIDANCE)
        self.assertIn("know your customer", rendered.text.lower())

    def test_06_account_guidance(self):
        _, guidance, rendered = _render("What is a savings account?")

        self.assertEqual(guidance.guidance_type, GuidanceType.ACCOUNT_GUIDANCE)
        self.assertIn("savings", rendered.text.lower())

    def test_07_cheque_status_guidance_pending(self):
        intent_result, guidance, rendered = _render("My cheque is pending")

        self.assertEqual(intent_result.intent, "unknown")
        self.assertEqual(guidance.guidance_type, GuidanceType.CHEQUE_STATUS_GUIDANCE)
        self.assertIn("pending", rendered.text.lower())
        self.assertEqual(rendered.actions, [])  # no numbered menu — just asks for the reference id

    def test_08_transaction_insight_not_intercepted_stays_on_llm_path(self):
        from app.conversation.manager import _INTERCEPT_GUIDANCE_TYPES

        _, guidance, _rendered = _render("Where am I spending most of my money?")

        # A guidance mapping DOES exist (for standalone testability, per
        # Task 9.1/9.2 section 2), but the manager's interception
        # whitelist deliberately excludes it — real spend data can only
        # come from the existing tool-using LLM path (see manager.py's
        # _INTERCEPT_GUIDANCE_TYPES docstring, and section 10 of the task).
        self.assertEqual(guidance.guidance_type, GuidanceType.TRANSACTION_GUIDANCE)
        self.assertNotIn(GuidanceType.TRANSACTION_GUIDANCE, _INTERCEPT_GUIDANCE_TYPES)


# ─── 14: natural-language + numbered + cancel/back resolution ──────────

class GuidanceHandoffResolutionTests(unittest.TestCase):
    _LOAN_ACTIONS = ["SHOW_LOAN_REQUIREMENTS", "SHOW_LOAN_DOCUMENTS", "START_LOAN_APPLICATION"]
    _TRANSFER_ACTIONS = ["START_TRANSFER", "SHOW_TRANSFER_INFORMATION", "CANCEL"]
    _CHEQUE_ACTIONS = ["START_CHEQUE_DEPOSIT", "SHOW_CHEQUE_INFORMATION", "BACK"]

    def test_numbered_reply(self):
        self.assertEqual(resolve_pending_action("3", self._LOAN_ACTIONS), GuidanceAction.START_LOAN_APPLICATION)
        self.assertEqual(resolve_pending_action("1", self._LOAN_ACTIONS), GuidanceAction.SHOW_LOAN_REQUIREMENTS)

    def test_start_application_phrase(self):
        self.assertEqual(resolve_pending_action("Start application", self._LOAN_ACTIONS), GuidanceAction.START_LOAN_APPLICATION)

    def test_yes_apply_phrase(self):
        self.assertEqual(resolve_pending_action("Yes apply", self._LOAN_ACTIONS), GuidanceAction.START_LOAN_APPLICATION)

    def test_i_want_to_apply_phrase(self):
        self.assertEqual(resolve_pending_action("I want to apply", self._LOAN_ACTIONS), GuidanceAction.START_LOAN_APPLICATION)

    def test_show_documents_phrase(self):
        self.assertEqual(resolve_pending_action("Show me the documents", self._LOAN_ACTIONS), GuidanceAction.SHOW_LOAN_DOCUMENTS)

    def test_tell_me_requirements_phrase(self):
        self.assertEqual(resolve_pending_action("Tell me the requirements", self._LOAN_ACTIONS), GuidanceAction.SHOW_LOAN_REQUIREMENTS)

    def test_deposit_the_cheque_phrase(self):
        self.assertEqual(resolve_pending_action("Deposit the cheque", self._CHEQUE_ACTIONS), GuidanceAction.START_CHEQUE_DEPOSIT)

    def test_lets_transfer_phrase(self):
        self.assertEqual(resolve_pending_action("Let's transfer", self._TRANSFER_ACTIONS), GuidanceAction.START_TRANSFER)

    def test_start_it_is_contextual_not_global(self):
        # The same "start it" resolves to a DIFFERENT action depending on
        # what was actually offered — never a hardcoded global mapping.
        self.assertEqual(resolve_pending_action("Start it", self._LOAN_ACTIONS), GuidanceAction.START_LOAN_APPLICATION)
        self.assertEqual(resolve_pending_action("Start it", self._TRANSFER_ACTIONS), GuidanceAction.START_TRANSFER)
        self.assertEqual(resolve_pending_action("Start it", self._CHEQUE_ACTIONS), GuidanceAction.START_CHEQUE_DEPOSIT)

    def test_cancel(self):
        self.assertEqual(resolve_pending_action("Cancel", self._LOAN_ACTIONS), GuidanceAction.CANCEL)

    def test_back(self):
        self.assertEqual(resolve_pending_action("Back", self._CHEQUE_ACTIONS), GuidanceAction.BACK)

    def test_unrelated_reply_resolves_to_none(self):
        self.assertIsNone(resolve_pending_action("What's the weather today?", self._LOAN_ACTIONS))

    def test_no_offered_actions_resolves_to_none(self):
        self.assertIsNone(resolve_pending_action("yes", []))

    def test_out_of_range_number_resolves_to_none(self):
        self.assertIsNone(resolve_pending_action("9", self._LOAN_ACTIONS))


# ─── ConversationManager integration: guidance turn + handoff turn ─────

def _fresh_context(phone_number="447818658034"):
    return ConversationContext(phone_number=phone_number)


class _FakeWorkflowManager:
    def __init__(self, handle_result=None):
        self.handle_result = handle_result or {"handled": False, "response": None}
        self.start_requested_result = {"handled": False, "response": None}
        self.handle_calls = []
        self.start_requested_calls = []
        self.transfer_handler = object()

    async def handle(self, phone_number, query, parsed_document=None, trace_id=""):
        self.handle_calls.append((phone_number, query))
        return self.handle_result

    def start_requested(self, phone_number, query, trace_id=""):
        self.start_requested_calls.append((phone_number, query))
        return self.start_requested_result


async def _fake_llm_fallback(query, phone_number, trace_id, parsed_document=None):
    return f"llm-answer:{query}"


def _manager():
    wf = _FakeWorkflowManager()
    return ConversationManager(workflow_manager=wf), wf


def _patches(context):
    return (
        patch("app.conversation.manager.build_context", return_value=context),
        patch("app.conversation.manager.check_registration_gate", new=AsyncMock(return_value=None)),
        patch("app.conversation.manager.get_workflow", return_value=None),
        patch("app.conversation.manager.append_to_session"),
    )


class ConversationManagerGuidanceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_09_active_workflow_help_still_uses_existing_llm_path(self):
        """Section 11/Scenario 6: WORKFLOW_HELP is deliberately NOT
        intercepted by guidance (see manager.py) — the existing
        LLM-with-workflow-context path already explains the current step
        without restarting it, so this proves that path is still reached
        for a "what should I do?" message during an active workflow."""
        manager, wf = _manager()
        context = _fresh_context()
        context.current_workflow = WORKFLOW_CHEQUE
        context.current_step = STEP_UPLOAD_CHEQUE
        wf.handle_result = {"handled": False, "response": None, "reprocess_query": "What should I do?"}

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "447818658034", "What should I do?", "t9", llm_fallback=_fake_llm_fallback
            )

        self.assertEqual(response, "llm-answer:What should I do?")

    async def test_10_start_application_after_loan_eligibility_guidance(self):
        manager, wf = _manager()
        context = _fresh_context()
        context.pending_action = "guidance:START_LOAN_APPLICATION"
        context.allowed_actions = ["SHOW_LOAN_REQUIREMENTS", "SHOW_LOAN_DOCUMENTS", "START_LOAN_APPLICATION"]

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True), \
             patch(
                 "app.conversation.manager.start_workflow_directly",
                 return_value={"handled": True, "response": "Loan application started."},
             ) as mock_start:
            response = await manager.handle_message(
                "447818658034", "Start application", "t10", llm_fallback=_fake_llm_fallback
            )

        mock_start.assert_called_once()
        self.assertEqual(mock_start.call_args.args[0], "loan")
        self.assertEqual(response, "Loan application started.")
        self.assertIsNone(context.pending_action)
        self.assertEqual(len(wf.handle_calls), 0)  # never reached WorkflowManager.handle()

    async def test_11_show_documents_does_not_start_workflow(self):
        manager, wf = _manager()
        context = _fresh_context()
        context.pending_action = "guidance:START_LOAN_APPLICATION"
        context.allowed_actions = ["SHOW_LOAN_REQUIREMENTS", "SHOW_LOAN_DOCUMENTS", "START_LOAN_APPLICATION"]

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.start_workflow_directly") as mock_start:
            response = await manager.handle_message(
                "447818658034", "Show documents", "t11", llm_fallback=_fake_llm_fallback
            )

        mock_start.assert_not_called()
        self.assertIn("document", response.lower())
        # The offer stays available — pending_action untouched — so a
        # follow-up "start it" still works.
        self.assertEqual(context.pending_action, "guidance:START_LOAN_APPLICATION")

    async def test_12_cancel_after_guidance_does_not_start_workflow(self):
        manager, wf = _manager()
        context = _fresh_context()
        context.pending_action = "guidance:START_LOAN_APPLICATION"
        context.allowed_actions = ["SHOW_LOAN_REQUIREMENTS", "SHOW_LOAN_DOCUMENTS", "START_LOAN_APPLICATION"]

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.start_workflow_directly") as mock_start:
            response = await manager.handle_message(
                "447818658034", "Cancel", "t12", llm_fallback=_fake_llm_fallback
            )

        mock_start.assert_not_called()
        self.assertIn("cancelled", response.lower())
        self.assertIsNone(context.pending_action)
        self.assertEqual(context.allowed_actions, [])

    async def test_13_back_after_guidance_returns_to_menu_without_starting_workflow(self):
        manager, wf = _manager()
        context = _fresh_context()
        context.pending_action = "guidance:START_CHEQUE_DEPOSIT"
        context.allowed_actions = ["START_CHEQUE_DEPOSIT", "SHOW_CHEQUE_INFORMATION", "BACK"]

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.start_workflow_directly") as mock_start:
            response = await manager.handle_message(
                "447818658034", "Back", "t13", llm_fallback=_fake_llm_fallback
            )

        mock_start.assert_not_called()
        self.assertIsNone(context.pending_action)

    async def test_14_stale_pending_action_ignored_during_active_workflow(self):
        """A leftover guidance offer must never hijack a numbered reply
        that's actually meant for a real, currently-active workflow step
        (e.g. selecting a beneficiary by number)."""
        manager, wf = _manager()
        context = _fresh_context()
        context.current_workflow = "transfer"
        context.current_step = "SELECT_BENEFICIARY"
        context.pending_action = "guidance:START_LOAN_APPLICATION"
        context.allowed_actions = ["SHOW_LOAN_REQUIREMENTS", "SHOW_LOAN_DOCUMENTS", "START_LOAN_APPLICATION"]
        wf.handle_result = {"handled": True, "response": "Beneficiary selected."}

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.start_workflow_directly") as mock_start:
            response = await manager.handle_message(
                "447818658034", "3", "t14", llm_fallback=_fake_llm_fallback
            )

        mock_start.assert_not_called()
        self.assertEqual(response, "Beneficiary selected.")
        self.assertEqual(len(wf.handle_calls), 1)

    async def test_loan_eligibility_message_produces_guidance_not_llm_call(self):
        manager, wf = _manager()
        context = _fresh_context()

        with _patches(context)[0], _patches(context)[1], _patches(context)[2], _patches(context)[3], \
             patch.object(manager.context_store, "save", return_value=True):
            response = await manager.handle_message(
                "447818658034",
                "I earn 50000 a month and want a personal loan",
                "t15",
                llm_fallback=_fake_llm_fallback,
            )

        self.assertNotIn("llm-answer", response)
        self.assertEqual(context.pending_action, "guidance:START_LOAN_APPLICATION")
        self.assertIn("START_LOAN_APPLICATION", context.allowed_actions)


# ─── 15/16: safety ──────────────────────────────────────────────────────

class GuidanceSafetyTests(unittest.TestCase):
    _CASES = [
        "I earn 50000 a month and want a personal loan",
        "What documents do I need for a personal loan?",
        "How do I transfer money?",
        "How do I deposit a cheque?",
        "What is KYC?",
        "What is a savings account?",
        "My cheque is pending",
    ]

    _SENSITIVE_TERMS = ("aadhaar", "pan number", "otp", "password", "cvv", " pin ", "account number:")
    _ELIGIBILITY_CLAIMS = (
        "you are eligible", "you're eligible", "your loan will be approved",
        "you can get", "you qualify for", "guaranteed approval",
    )

    def test_15_no_sensitive_data_in_any_guidance_response(self):
        for text in self._CASES:
            _, guidance, rendered = _render(text)
            if not rendered:
                continue
            lowered = rendered.text.lower()
            for term in self._SENSITIVE_TERMS:
                self.assertNotIn(term, lowered, f"{term!r} leaked for {text!r}")

    def test_16_no_unsupported_eligibility_or_approval_claims(self):
        for text in self._CASES:
            _, guidance, rendered = _render(text)
            if not rendered:
                continue
            lowered = rendered.text.lower()
            for claim in self._ELIGIBILITY_CLAIMS:
                self.assertNotIn(claim, lowered, f"{claim!r} leaked for {text!r}")


class GuidanceBoundaryTests(unittest.TestCase):
    def test_responses_module_has_no_database_or_workflow_imports(self):
        import ast
        import inspect

        from app.conversation.guidance import handoff as handoff_module
        from app.conversation.guidance import responses as responses_module

        forbidden_prefixes = ("app.database", "app.workflows", "app.agent.tools", "psycopg2", "redis")
        for module in (responses_module, handoff_module):
            tree = ast.parse(inspect.getsource(module))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            for name in imported:
                for forbidden in forbidden_prefixes:
                    self.assertFalse(
                        name == forbidden or name.startswith(forbidden + "."),
                        f"{module.__name__} must not import {name}",
                    )

    def test_guidance_cannot_create_a_transaction_loan_cheque_or_kyc_record(self):
        # No function anywhere in the guidance package accepts a
        # database connection/cursor, nor imports anything that could
        # persist a record — structurally, not just by convention.
        import inspect

        from app.conversation.guidance import handoff, policy, responses

        for module in (policy, responses, handoff):
            for name, obj in vars(module).items():
                if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                    params = list(inspect.signature(obj).parameters.keys())
                    for forbidden in ("conn", "cursor", "db", "connection"):
                        self.assertNotIn(forbidden, params, f"{module.__name__}.{name} must not accept {forbidden}")


if __name__ == "__main__":
    unittest.main()

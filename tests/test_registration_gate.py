"""Tests for deferring registration until an actual service is requested —
an unregistered customer's general question should be answered (via the
RAG/LLM path further down the pipeline) instead of being force-started
into onboarding on every message.

The gate is now driven by the single LLM routing decision (an
LLMRoutingDecision) computed once per turn by
app/conversation/manager.py — never by keyword-sniffing the raw text
itself — so these tests pass a fake decision object directly rather than
relying on any classification happening inside the gate."""

import unittest
from unittest.mock import patch

from app.conversation.intent.llm_routing import LLMRoutingDecision
from app.services.registration_gate import check_registration_gate
from app.workflows.constants import STEP_COLLECT_AADHAAR, WORKFLOW_ONBOARDING
from app.workflows.manager import WorkflowManager
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow


def _decision(action: str, intent: str = "unknown", target_workflow=None, certainty: str = "high") -> LLMRoutingDecision:
    return LLMRoutingDecision(intent=intent, action=action, certainty=certainty, target_workflow=target_workflow)


class FakeRedis:
    def __init__(self):
        self._store = {}

    def setex(self, key, ttl, value):
        self._store[key] = value
        return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, key):
        return 1 if self._store.pop(key, None) is not None else 0


class RegistrationGateDeferralTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_unregistered_question_is_not_onboarded(self):
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = check_registration_gate(
                phone_number="441111111111",
                query="What documents do I need for a home loan?",
                decision=_decision("RAG", intent="loan_question"),
                is_registered=False,
                trace_id="t1",
            )
        # None means "let this fall through" — the gate must not have
        # started onboarding for a plain question.
        self.assertIsNone(result)
        self.assertIsNone(get_workflow("441111111111"))

    async def test_unregistered_service_request_starts_onboarding_with_pending_query(self):
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = check_registration_gate(
                phone_number="441111111111",
                query="I want to deposit a cheque",
                decision=_decision("START_WORKFLOW", intent="cheque_deposit_request", target_workflow="cheque"),
                is_registered=False,
                trace_id="t2",
            )
        self.assertTrue(result["handled"])
        workflow = get_workflow("441111111111")
        self.assertEqual(workflow["type"], WORKFLOW_ONBOARDING)
        self.assertEqual(workflow["data"]["pending_service_query"], "I want to deposit a cheque")

    async def test_unregistered_greeting_still_starts_onboarding(self):
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = check_registration_gate(
                phone_number="441111111111",
                query="hi",
                decision=_decision("GREETING", intent="greeting"),
                is_registered=False,
                trace_id="t3",
            )
        self.assertTrue(result["handled"])
        workflow = get_workflow("441111111111")
        self.assertEqual(workflow["type"], WORKFLOW_ONBOARDING)
        self.assertNotIn("pending_service_query", workflow["data"])

    async def test_unregistered_out_of_scope_is_not_onboarded(self):
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = check_registration_gate(
                phone_number="441111111111",
                query="tell me a joke",
                decision=_decision("OUT_OF_SCOPE", intent="out_of_scope"),
                is_registered=False,
                trace_id="t3b",
            )
        self.assertIsNone(result)
        self.assertIsNone(get_workflow("441111111111"))

    async def test_no_decision_computed_defers_normally(self):
        # A None decision (e.g. the deterministic pre-filter already
        # resolved the turn to something the gate has no opinion on)
        # must never crash and must default to deferring.
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = check_registration_gate(
                phone_number="441111111111",
                query="back",
                decision=None,
                is_registered=False,
                trace_id="t3c",
            )
        self.assertIsNone(result)


class RegisteredCustomerMenuTapTests(unittest.IsolatedAsyncioTestCase):
    """A tapped main-menu row (WhatsApp list_reply id "1".."8") for a
    REGISTERED customer with no session history yet (their very first
    message ever, or any time after the session history TTL expires)
    must not be swallowed by the "first message -> show the menu again"
    fallback — otherwise every menu option looks unwired the first time
    it's actually used. See registration_gate.py's MENU_DIGITS."""

    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        memory_patcher = patch("app.memory.redis_client", self.fake_redis)
        memory_patcher.start()
        self.addCleanup(memory_patcher.stop)

    async def test_menu_digit_is_not_swallowed_on_first_ever_message(self):
        with patch(
            "app.services.registration_gate.get_customer_by_phone",
            return_value={"full_name": "John Smith"},
        ):
            for digit in ("1", "2", "3", "4", "5", "6", "7", "8"):
                with self.subTest(digit=digit):
                    result = check_registration_gate(
                        phone_number=f"44770090020{digit}",
                        query=digit,
                        decision=None,
                        is_registered=True,
                        trace_id=f"t{digit}",
                    )
                    # None means "let this fall through to
                    # WorkflowManager.start_requested()", which is what
                    # actually knows what each digit means — the gate
                    # must not intercept it with the greeting/menu again.
                    self.assertIsNone(result)

    async def test_non_menu_first_message_still_shows_the_menu(self):
        # A genuinely ambiguous first message ("hey there") is correctly
        # still shown the welcome menu — this fix is scoped to the known
        # menu digits only, not a blanket bypass.
        with patch(
            "app.services.registration_gate.get_customer_by_phone",
            return_value={"full_name": "John Smith"},
        ), patch("app.services.registration_gate.get_accounts_by_phone", return_value=[]):
            result = check_registration_gate(
                phone_number="441111111111",
                query="hey there",
                decision=_decision("CLARIFY", intent="unknown"),
                is_registered=True,
                trace_id="t8",
            )
        self.assertTrue(result["handled"])

    async def test_greeting_shown_even_with_existing_history(self):
        with patch(
            "app.services.registration_gate.get_customer_by_phone",
            return_value={"full_name": "John Smith"},
        ), patch("app.services.registration_gate.get_accounts_by_phone", return_value=[]), \
           patch("app.services.registration_gate.get_session_history", return_value=[{"role": "user", "content": "hi"}]):
            result = check_registration_gate(
                phone_number="441111111111",
                query="hi",
                decision=_decision("GREETING", intent="greeting"),
                is_registered=True,
                trace_id="t9",
            )
        self.assertTrue(result["handled"])


class ResumePendingServiceTests(unittest.IsolatedAsyncioTestCase):
    """WorkflowManager.resume_pending_request() should resume a stashed
    pending_service_query once onboarding actually completes (a real
    customer record now exists), but never after a cancelled/declined
    registration."""

    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = WorkflowManager()

    async def test_resumes_cheque_deposit_after_successful_registration(self):
        workflow = create_workflow_model(
            WORKFLOW_ONBOARDING,
            STEP_COLLECT_AADHAAR,
            data={"pending_service_query": "I want to deposit a cheque"},
        )
        create_workflow("441111111111", workflow)

        with patch("app.workflows.processors.onboarding.OnboardingWorkflowHandler.handle") as mock_handle, \
             patch("app.database.get_customer_by_phone", return_value={"full_name": "Test User"}), \
             patch(
                 "app.workflows.manager.classify_and_route_llm_sync",
                 return_value=_decision("START_WORKFLOW", intent="cheque_deposit_request", target_workflow="cheque"),
             ):
            def _complete(*args, **kwargs):
                from app.workflows.memory import complete_workflow
                complete_workflow("441111111111")
                return {"handled": True, "response": "Account created!"}
            mock_handle.side_effect = _complete

            result = self.manager.handle(
                phone_number="441111111111",
                query="savings",
                trace_id="t4",
            )

        self.assertTrue(result["handled"])
        self.assertIn("Account created!", result["response"])
        self.assertIn("cheque", result["response"].lower())
        resumed_workflow = get_workflow("441111111111")
        self.assertIsNotNone(resumed_workflow)
        self.assertEqual(resumed_workflow["type"], "cheque")

    async def test_does_not_resume_after_cancelled_registration(self):
        workflow = create_workflow_model(
            WORKFLOW_ONBOARDING,
            STEP_COLLECT_AADHAAR,
            data={"pending_service_query": "I want to deposit a cheque"},
        )
        create_workflow("441111111111", workflow)

        with patch("app.workflows.processors.onboarding.OnboardingWorkflowHandler.handle") as mock_handle, \
             patch("app.database.get_customer_by_phone", return_value=None):
            def _cancel(*args, **kwargs):
                from app.workflows.memory import complete_workflow
                complete_workflow("441111111111")
                return {"handled": True, "response": "Registration cancelled."}
            mock_handle.side_effect = _cancel

            result = self.manager.handle(
                phone_number="441111111111",
                query="no",
                trace_id="t5",
            )

        self.assertEqual(result["response"], "Registration cancelled.")
        self.assertIsNone(get_workflow("441111111111"))


if __name__ == "__main__":
    unittest.main()

"""Tests for deferring registration until an actual service is requested —
an unregistered customer's general question should be answered (via the
RAG/LLM path further down the pipeline) instead of being force-started
into onboarding on every message. Follows this project's unittest +
FakeRedis convention (see tests/test_response_ux.py)."""

import unittest
from unittest.mock import patch

from app.services.registration_gate import check_registration_gate
from app.workflows.constants import STEP_COLLECT_AADHAAR, WORKFLOW_ONBOARDING
from app.workflows.manager import WorkflowManager
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow


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
            result = await check_registration_gate(
                phone_number="447700900111",
                query="What documents do I need for a home loan?",
                trace_id="t1",
            )
        # None means "let this fall through" — the gate must not have
        # started onboarding for a plain question.
        self.assertIsNone(result)
        self.assertIsNone(get_workflow("447700900111"))

    async def test_unregistered_service_request_starts_onboarding_with_pending_query(self):
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = await check_registration_gate(
                phone_number="447700900112",
                query="I want to deposit a cheque",
                trace_id="t2",
            )
        self.assertTrue(result["handled"])
        workflow = get_workflow("447700900112")
        self.assertEqual(workflow["type"], WORKFLOW_ONBOARDING)
        self.assertEqual(workflow["data"]["pending_service_query"], "I want to deposit a cheque")

    async def test_unregistered_greeting_still_starts_onboarding(self):
        with patch("app.services.registration_gate.get_customer_by_phone", return_value=None):
            result = await check_registration_gate(
                phone_number="447700900113",
                query="hi",
                trace_id="t3",
            )
        self.assertTrue(result["handled"])
        workflow = get_workflow("447700900113")
        self.assertEqual(workflow["type"], WORKFLOW_ONBOARDING)
        self.assertNotIn("pending_service_query", workflow["data"])


class ResumePendingServiceTests(unittest.IsolatedAsyncioTestCase):
    """WorkflowManager.handle() should resume a stashed pending_service_query
    once onboarding actually completes (a real customer record now exists),
    but never after a cancelled/declined registration."""

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
        create_workflow("447700900114", workflow)

        with patch("app.workflows.processors.onboarding.OnboardingWorkflowHandler.handle") as mock_handle, \
             patch("app.database.get_customer_by_phone", return_value={"full_name": "Test User"}):
            async def _complete(*args, **kwargs):
                from app.workflows.memory import complete_workflow
                complete_workflow("447700900114")
                return {"handled": True, "response": "Account created!"}
            mock_handle.side_effect = _complete

            result = await self.manager.handle(
                phone_number="447700900114",
                query="savings",
                trace_id="t4",
            )

        self.assertTrue(result["handled"])
        self.assertIn("Account created!", result["response"])
        self.assertIn("Cheque deposit started", result["response"])
        resumed_workflow = get_workflow("447700900114")
        self.assertIsNotNone(resumed_workflow)
        self.assertEqual(resumed_workflow["type"], "cheque")

    async def test_does_not_resume_after_cancelled_registration(self):
        workflow = create_workflow_model(
            WORKFLOW_ONBOARDING,
            STEP_COLLECT_AADHAAR,
            data={"pending_service_query": "I want to deposit a cheque"},
        )
        create_workflow("447700900115", workflow)

        with patch("app.workflows.processors.onboarding.OnboardingWorkflowHandler.handle") as mock_handle, \
             patch("app.database.get_customer_by_phone", return_value=None):
            async def _cancel(*args, **kwargs):
                from app.workflows.memory import complete_workflow
                complete_workflow("447700900115")
                return {"handled": True, "response": "Registration cancelled."}
            mock_handle.side_effect = _cancel

            result = await self.manager.handle(
                phone_number="447700900115",
                query="no",
                trace_id="t5",
            )

        self.assertEqual(result["response"], "Registration cancelled.")
        self.assertIsNone(get_workflow("447700900115"))


if __name__ == "__main__":
    unittest.main()

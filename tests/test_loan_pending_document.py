"""A loan form image uploaded before the customer said which loan type it
was for (see message_handler.py's bare-upload auto-detect) must be applied
once they answer, not discarded. Follows this project's FakeRedis
convention (see tests/test_response_ux.py)."""

import unittest
from unittest.mock import patch

from app.workflows.constants import STEP_SELECT_LOAN_TYPE, STEP_UPLOAD_LOAN_FORM
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow
from app.workflows.processors.loan import LoanWorkflowHandler


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


class LoanPendingDocumentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.handler = LoanWorkflowHandler()

    async def test_applies_pending_form_content_after_type_selection(self):
        pending_content = {
            "applicant_name": "Jane Doe",
            "monthly_income": "50000",
            "employment_type": "Salaried",
        }
        workflow = create_workflow_model(
            "loan", STEP_SELECT_LOAN_TYPE,
            data={"pending_document_content": pending_content},
        )
        create_workflow("447700900200", workflow)

        result = await self.handler.handle(
            workflow=get_workflow("447700900200"),
            phone_number="447700900200",
            query="home loan",
            trace_id="t1",
        )

        self.assertTrue(result["handled"])
        stored = get_workflow("447700900200")
        self.assertEqual(stored["step"], STEP_UPLOAD_LOAN_FORM)
        self.assertEqual(stored["data"]["applicant_name"], "Jane Doe")
        self.assertEqual(stored["data"]["loan_type"], "home")
        self.assertNotIn("pending_document_content", stored["data"])
        # Already have applicant_name/monthly_income/employment_type — the
        # next thing asked for should be a still-missing field, not those.
        self.assertNotIn("applicant name", result["response"].lower())

    async def test_no_pending_content_behaves_as_before(self):
        workflow = create_workflow_model("loan", STEP_SELECT_LOAN_TYPE)
        create_workflow("447700900201", workflow)

        result = await self.handler.handle(
            workflow=get_workflow("447700900201"),
            phone_number="447700900201",
            query="vehicle",
            trace_id="t2",
        )

        self.assertTrue(result["handled"])
        stored = get_workflow("447700900201")
        self.assertEqual(stored["step"], STEP_UPLOAD_LOAN_FORM)
        self.assertEqual(stored["data"]["loan_type"], "vehicle")


if __name__ == "__main__":
    unittest.main()

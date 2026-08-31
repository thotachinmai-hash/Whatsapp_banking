"""The customer may upload Aadhaar and PAN in either order during
registration/add-account — see app/workflows/processors/onboarding.py's
_handle_collect_identity_document/_detect_identity_document_type. Previously
the flow was strictly Aadhaar-then-PAN: a PAN image uploaded while the step
was still STEP_COLLECT_AADHAAR was read as if it were an Aadhaar card and
rejected."""

import unittest
from unittest.mock import patch

from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_COLLECT_PAN,
    STEP_CONFIRM_REGISTRATION,
    WORKFLOW_ONBOARDING,
)
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow
from app.workflows.processors.onboarding import OnboardingWorkflowHandler


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


_AADHAAR_CONTENT = {
    "id_type": "aadhaar",
    "aadhaar_number": "123456789012",
    "full_name": "Jordan Smith",
}
_PAN_CONTENT = {
    "id_type": "pan",
    "pan_number": "ABCDE1234F",
    "full_name": "Jordan Smith",
}


class OnboardingFlexibleOrderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.handler = OnboardingWorkflowHandler()

    async def test_pan_uploaded_first_is_accepted_and_asks_for_aadhaar_next(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)
        parsed_document = {"mime_type": "image/jpeg", "content": _PAN_CONTENT}

        result = self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "", parsed_document, "t1"
        )

        self.assertTrue(result["handled"])
        stored = get_workflow(self.phone)
        self.assertEqual(stored["data"].get("pan_number"), "ABCDE1234F")
        self.assertNotIn("aadhaar_number", stored["data"])
        # Still needs Aadhaar -- stays on the "Aadhaar still needed" step,
        # not an error.
        self.assertEqual(stored["step"], STEP_COLLECT_AADHAAR)
        self.assertIn("Aadhaar", result["response"])

    async def test_aadhaar_then_pan_completes_registration(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        first = self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "",
            {"mime_type": "image/jpeg", "content": _AADHAAR_CONTENT}, "t1",
        )
        self.assertTrue(first["handled"])
        after_first = get_workflow(self.phone)
        self.assertEqual(after_first["step"], STEP_COLLECT_PAN)

        second = self.handler.handle(
            {"step": STEP_COLLECT_PAN}, self.phone, "",
            {"mime_type": "image/jpeg", "content": _PAN_CONTENT}, "t2",
        )
        self.assertTrue(second["handled"])
        after_second = get_workflow(self.phone)
        self.assertEqual(after_second["step"], STEP_CONFIRM_REGISTRATION)
        self.assertEqual(after_second["data"].get("aadhaar_number"), "123456789012")
        self.assertEqual(after_second["data"].get("pan_number"), "ABCDE1234F")

    async def test_pan_then_aadhaar_completes_registration(self):
        """The reverse order -- PAN first, then Aadhaar -- must reach the
        same confirmation step, which was impossible before this fix."""
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        first = self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "",
            {"mime_type": "image/jpeg", "content": _PAN_CONTENT}, "t1",
        )
        self.assertTrue(first["handled"])
        after_first = get_workflow(self.phone)
        # PAN is already provided; still on this step only because
        # Aadhaar is what's now outstanding.
        self.assertEqual(after_first["step"], STEP_COLLECT_AADHAAR)
        self.assertEqual(after_first["data"].get("pan_number"), "ABCDE1234F")

        second = self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "",
            {"mime_type": "image/jpeg", "content": _AADHAAR_CONTENT}, "t2",
        )
        self.assertTrue(second["handled"])
        after_second = get_workflow(self.phone)
        self.assertEqual(after_second["step"], STEP_CONFIRM_REGISTRATION)
        self.assertEqual(after_second["data"].get("aadhaar_number"), "123456789012")
        self.assertEqual(after_second["data"].get("pan_number"), "ABCDE1234F")

    async def test_unreadable_document_still_rejected_with_expected_step_wording(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)
        parsed_document = {"mime_type": "image/jpeg", "content": {"full_name": "No ID Fields Here"}}

        result = self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "", parsed_document, "t1"
        )

        self.assertTrue(result["handled"])
        self.assertIn("Aadhaar", result["response"])
        stored = get_workflow(self.phone)
        self.assertEqual(stored["step"], STEP_COLLECT_AADHAAR)

    async def test_detect_identity_document_type_shape_fallback_with_no_id_type_field(self):
        """The exact older content shape (no "id_type" key at all) that
        pre-existing tests/production data may still send must keep
        resolving correctly via shape detection alone."""
        self.assertEqual(
            self.handler._detect_identity_document_type({"aadhaar_number": "123456789012"}),
            "aadhaar",
        )
        self.assertEqual(
            self.handler._detect_identity_document_type({"pan_number": "ABCDE1234F"}),
            "pan",
        )
        self.assertIsNone(self.handler._detect_identity_document_type({"full_name": "No ID Here"}))


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from unittest.mock import patch

from app.services.message_handler import build_document_prompt
from app.workflows.constants import STEP_COLLECT_AADHAAR, STEP_COLLECT_PAN, WORKFLOW_ONBOARDING
from app.workflows.processors.onboarding import OnboardingWorkflowHandler
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


class OnboardingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = OnboardingWorkflowHandler()

    def test_reports_name_and_dob_mismatches(self) -> None:
        user_profile = {
            "full_name": "John Doe",
            "date_of_birth": "01/01/1990",
        }
        document_data = {
            "full_name": "Jane Doe",
            "date_of_birth": "02/02/1990",
        }

        errors = self.handler._validate_profile_data(user_profile, document_data)

        self.assertIn("full_name", errors)
        self.assertIn("date_of_birth", errors)

    def test_accepts_matching_name_and_dob(self) -> None:
        user_profile = {
            "full_name": "John Doe",
            "date_of_birth": "01/01/1990",
        }
        document_data = {
            "full_name": "John Doe",
            "date_of_birth": "01/01/1990",
        }

        errors = self.handler._validate_profile_data(user_profile, document_data)

        self.assertEqual(errors, [])

    def test_pan_only_document_does_not_bleed_into_other_profile_fields(self) -> None:
        # _document_value's deep-search fallback used to apply its
        # PAN-shape regex unconditionally for ANY field lookup — so a PAN
        # card whose OCR content only had a bare "pan_number" key (no
        # separate name/DOB/address — a realistic OCR outcome, PAN cards
        # print far less than Aadhaar) had its own PAN number string
        # wrongly extracted AS the person's full_name/date_of_birth/
        # address/guardian_name, producing a false "details don't match"
        # rejection on a completely valid PAN upload.
        profile = self.handler._extract_profile_fields({"pan_number": "ABCDE1234F"})
        self.assertEqual(profile, {})

        stored = {"full_name": "John Smith", "date_of_birth": "1990-01-01", "address": "1 Test St"}
        self.assertEqual(self.handler._validate_profile_data(stored, profile), [])

    def test_pan_number_itself_is_still_found_via_deep_search(self) -> None:
        # The fix above must not break the deep-search fallback's actual
        # intended job — finding a PAN number nested inside an unexpected
        # shape (e.g. a wrapper object the vision model returned).
        value = self.handler._document_value(
            {"nested": {"raw_text": "ABCDE1234F"}}, "pan_number", "pan", "pan_card_number"
        )
        self.assertEqual(value, "ABCDE1234F")

    def test_accepts_initial_expansion_for_name(self) -> None:
        user_profile = {
            "full_name": "J. Doe",
        }
        document_data = {
            "full_name": "John Doe",
        }

        errors = self.handler._validate_profile_data(user_profile, document_data)

        self.assertEqual(errors, [])

    def test_detects_guardian_name_mismatch(self) -> None:
        user_profile = {
            "guardian_name": "Ravi Sharma",
        }
        document_data = {
            "guardian_name": "Amit Sharma",
        }

        errors = self.handler._validate_profile_data(user_profile, document_data)

        self.assertIn("guardian_name", errors)

    def test_aadhaar_prompt_requests_profile_fields(self) -> None:
        prompt = build_document_prompt({"step": STEP_COLLECT_AADHAAR}, "aadhaar.png")

        self.assertIn("full_name", prompt)
        self.assertIn("date_of_birth", prompt)
        self.assertIn("address", prompt)
        self.assertIn("guardian_name", prompt)

    def test_pan_prompt_requests_profile_fields(self) -> None:
        prompt = build_document_prompt({"step": STEP_COLLECT_PAN}, "pan.png")

        self.assertIn("full_name", prompt)
        self.assertIn("date_of_birth", prompt)
        self.assertIn("address", prompt)
        self.assertIn("guardian_name", prompt)

    def test_stops_onboarding_for_stop_commands(self) -> None:
        with patch("app.workflows.processors.onboarding.complete_workflow") as mock_complete:
            workflow = {"step": STEP_COLLECT_AADHAAR}
            result = asyncio.run(self.handler.handle(workflow, "919000000000", "please stop"))

        self.assertTrue(result["handled"])
        self.assertIn("stopped", result["response"].lower())
        mock_complete.assert_called_once_with("919000000000")


class OnboardingStartsAtAadhaarTests(unittest.IsolatedAsyncioTestCase):
    """Registration no longer asks for the customer's name as a separate
    typed step — it's derived from the Aadhaar/PAN OCR content, the same
    way date_of_birth/address/guardian_name already were."""

    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "447700900120"
        self.handler = OnboardingWorkflowHandler()

    async def test_aadhaar_upload_derives_full_name_with_no_prior_name_step(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)
        parsed_document = {
            "mime_type": "image/jpeg",
            "content": {"aadhaar_number": "123456789012", "full_name": "Jordan Smith"},
        }

        result = await self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "", parsed_document, "t1"
        )

        self.assertTrue(result["handled"])
        stored = get_workflow(self.phone)["data"]
        self.assertEqual(stored.get("full_name"), "Jordan Smith")
        self.assertEqual(stored.get("aadhaar_number"), "123456789012")

    async def test_initial_yes_starts_registration_with_aadhaar_prompt(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        result = await self.handler.handle(
            {"step": STEP_COLLECT_AADHAAR}, self.phone, "yes", None, "t2"
        )

        self.assertTrue(result["handled"])
        self.assertEqual(
            result["response"],
            "Great! Please upload a clear image of your Aadhaar card to begin registration."
        )

    async def test_initial_no_declines_registration(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        with patch("app.workflows.processors.onboarding.complete_workflow") as mock_complete:
            result = await self.handler.handle({"step": STEP_COLLECT_AADHAAR}, self.phone, "no", None, "t3")

        self.assertTrue(result["handled"])
        self.assertEqual(result["response"], "We are not proceeding with your registration but you can still chat with us.")
        mock_complete.assert_called_once_with(self.phone)
        self.assertIsNone(get_workflow(self.phone))

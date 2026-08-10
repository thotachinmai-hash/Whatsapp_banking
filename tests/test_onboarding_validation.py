import asyncio
import unittest
from unittest.mock import patch

from app.services.message_handler import build_document_prompt
from app.workflows.constants import STEP_COLLECT_AADHAAR, STEP_COLLECT_PAN
from app.workflows.processors.onboarding import OnboardingWorkflowHandler


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

"""End-to-end wiring tests for every WhatsApp interactive button/list row
in the app — each test drives the REAL WorkflowManager + real processor
(only the database layer is mocked), simulating exactly what happens when
a customer taps a button/list row rather than typing text. Complements:

- tests/test_conversation_manager.py (main-menu digit taps with no active
  workflow reaching WorkflowManager.start_requested())
- tests/test_response_ux.py's InteractiveListConversionTests (row id
  shapes) and NavigationTests (Back/Cancel)
- tests/test_workflow_nlu.py (interpret_confirmation's "yes"/"no" parsing
  in isolation)

This file is the missing piece: does tapping "Yes, submit" / "No, cancel"
/ a loan-type row / a source-account row / a transfer confirm button
actually advance the real processor and hit the real database call it's
supposed to.
"""

import unittest
from unittest.mock import patch

from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_CONFIRM_KYC,
    STEP_CONFIRM_LOAN,
    STEP_CONFIRM_REGISTRATION,
    STEP_CONFIRM_TRANSFER,
    STEP_SELECT_ACCOUNT_TYPE,
    STEP_SELECT_AMOUNT,
    STEP_SELECT_SOURCE_ACCOUNT,
    WORKFLOW_KYC,
    WORKFLOW_LOAN,
    WORKFLOW_ONBOARDING,
    WORKFLOW_TRANSFER,
)
from app.workflows.manager import WorkflowManager
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis — same pattern as
    tests/test_response_ux.py's FakeRedis, duplicated locally to avoid a
    cross-test-module import (tests/ has no __init__.py)."""

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


class ButtonWiringTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = WorkflowManager()
        self.phone = "447700900099"
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)


class LoanConfirmButtonTests(ButtonWiringTestCase):
    def _complete_loan_workflow(self):
        workflow = create_workflow_model(WORKFLOW_LOAN, STEP_CONFIRM_LOAN, data={
            "loan_type": "personal",
            "account_number": "GB12FNCL00010001234567",
            "applicant_name": "Alex Doe",
            "monthly_income": "3000",
            "employment_type": "salaried",
            "requested_amount": "10000",
            "tenure_months": "24",
            "purpose": "home renovation",
        })
        create_workflow(self.phone, workflow)
        return workflow

    async def test_tapping_yes_submits_the_loan(self):
        self._complete_loan_workflow()
        with patch("app.workflows.processors.loan.create_loan_request") as mock_create:
            result = await self.manager.handle(self.phone, "yes", trace_id="lt1")
        self.assertTrue(result["handled"])
        mock_create.assert_called_once()
        self.assertIsNone(get_workflow(self.phone))
        self.assertIn("loan", str(result["response"]).lower())

    async def test_tapping_no_cancels_the_loan(self):
        self._complete_loan_workflow()
        with patch("app.workflows.processors.loan.create_loan_request") as mock_create:
            result = await self.manager.handle(self.phone, "no", trace_id="lt2")
        self.assertTrue(result["handled"])
        mock_create.assert_not_called()
        self.assertIsNone(get_workflow(self.phone))

    async def test_tapped_loan_type_row_id_advances_and_persists_type(self):
        from app.workflows.constants import STEP_SELECT_LOAN_TYPE, STEP_UPLOAD_LOAN_FORM

        workflow = create_workflow_model(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE)
        create_workflow(self.phone, workflow)
        fake_accounts = [{"account_number": "GB12FNCL00010001234567", "account_type": "current", "balance": "500.00", "currency": "INR"}]
        with patch("app.workflows.processors.loan.get_accounts_by_phone", return_value=fake_accounts):
            result = await self.manager.handle(self.phone, "3", trace_id="lt3")  # row id "3" = Vehicle Loan
        self.assertTrue(result["handled"])
        stored = get_workflow(self.phone)
        self.assertEqual(stored["step"], STEP_UPLOAD_LOAN_FORM)
        self.assertEqual(stored["data"]["loan_type"], "vehicle")


class KYCConfirmButtonTests(ButtonWiringTestCase):
    def _complete_kyc_workflow(self):
        workflow = create_workflow_model(WORKFLOW_KYC, STEP_CONFIRM_KYC, data={
            "full_name": "Alex Doe",
            "date_of_birth": "1990-01-01",
            "address": "1 Test Street",
            "aadhaar_number": "123412341234",
            "pan_number": "ABCDE1234F",
        })
        create_workflow(self.phone, workflow)
        return workflow

    async def test_tapping_yes_submits_the_kyc_update(self):
        self._complete_kyc_workflow()
        with patch("app.workflows.processors.kyc.create_kyc_request") as mock_create:
            result = await self.manager.handle(self.phone, "yes", trace_id="kt1")
        self.assertTrue(result["handled"])
        mock_create.assert_called_once()
        self.assertIsNone(get_workflow(self.phone))

    async def test_tapping_no_cancels_the_kyc_update(self):
        self._complete_kyc_workflow()
        with patch("app.workflows.processors.kyc.create_kyc_request") as mock_create:
            result = await self.manager.handle(self.phone, "no", trace_id="kt2")
        self.assertTrue(result["handled"])
        mock_create.assert_not_called()
        self.assertIsNone(get_workflow(self.phone))


class OnboardingButtonTests(ButtonWiringTestCase):
    def _confirm_registration_workflow(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_CONFIRM_REGISTRATION, data={
            "full_name": "Alex Doe",
            "aadhaar_number": "123412341234",
            "pan_number": "ABCDE1234F",
            "date_of_birth": "1990-01-01",
            "guardian_name": "Sam Doe",
            "address": "1 Test Street",
        })
        create_workflow(self.phone, workflow)
        return workflow

    async def test_tapping_yes_confirm_creates_customer_and_moves_to_account_type(self):
        with patch("app.workflows.processors.onboarding.create_customer", return_value={"id": 1}) as mock_create:
            self._confirm_registration_workflow()
            result = await self.manager.handle(self.phone, "yes", trace_id="ot1")
        self.assertTrue(result["handled"])
        mock_create.assert_called_once()
        stored = get_workflow(self.phone)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["step"], STEP_SELECT_ACCOUNT_TYPE)

    async def test_tapping_no_restart_cancels_registration(self):
        with patch("app.workflows.processors.onboarding.create_customer") as mock_create:
            self._confirm_registration_workflow()
            result = await self.manager.handle(self.phone, "no", trace_id="ot2")
        self.assertTrue(result["handled"])
        mock_create.assert_not_called()
        self.assertIsNone(get_workflow(self.phone))

    async def test_tapping_yes_at_welcome_asks_for_aadhaar(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        result = await self.manager.handle(self.phone, "yes", trace_id="ot6")

        self.assertTrue(result["handled"])
        self.assertEqual(
            result["response"],
            "Great! Please upload a clear image of your Aadhaar card to begin registration."
        )
        stored = get_workflow(self.phone)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["step"], STEP_COLLECT_AADHAAR)

    async def test_tapping_no_at_welcome_declines_registration(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)

        result = await self.manager.handle(self.phone, "no", trace_id="ot7")

        self.assertTrue(result["handled"])
        self.assertEqual(
            result["response"],
            "We are not proceeding with your registration but you can still chat with us."
        )
        self.assertIsNone(get_workflow(self.phone))

    async def test_tapped_account_type_row_creates_account(self):
        workflow = create_workflow_model(WORKFLOW_ONBOARDING, STEP_SELECT_ACCOUNT_TYPE, data={"full_name": "Alex Doe"})
        create_workflow(self.phone, workflow)
        fake_account = {"account_number": "GB12FNCL00010009999999", "account_type": "current"}
        with patch("app.workflows.processors.onboarding.create_zero_balance_account", return_value=fake_account) as mock_acct, \
             patch("app.workflows.processors.onboarding.cache_active_account"), \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]):
            result = await self.manager.handle(self.phone, "2", trace_id="ot3")  # row id "2" = Current Account
        self.assertTrue(result["handled"])
        mock_acct.assert_called_once()
        self.assertEqual(mock_acct.call_args.kwargs["account_type"], "current")
        self.assertIsNone(get_workflow(self.phone))


class TransferButtonTests(ButtonWiringTestCase):
    def _accounts(self):
        return [
            {"account_number": "GB12FNCL00010001111111", "account_type": "savings", "balance": "500.00"},
            {"account_number": "GB12FNCL00010002222222", "account_type": "current", "balance": "50.00"},
        ]

    async def test_tapped_source_account_row_advances_to_confirm(self):
        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_SOURCE_ACCOUNT, data={
            "beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999", "amount": "£25.00",
        })
        create_workflow(self.phone, workflow)
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=self._accounts()):
            result = await self.manager.handle(self.phone, "1", trace_id="tt1")  # row id "1" = first account
        self.assertTrue(result["handled"])
        stored = get_workflow(self.phone)
        self.assertEqual(stored["step"], STEP_CONFIRM_TRANSFER)
        self.assertEqual(stored["data"]["source_account"], "GB12FNCL00010001111111")

    async def test_tapping_yes_send_it_creates_the_transfer(self):
        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_CONFIRM_TRANSFER, data={
            "beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999",
            "amount": "£25.00", "source_account": "GB12FNCL00010001111111",
        })
        create_workflow(self.phone, workflow)
        with patch("app.workflows.processors.transfer.create_transfer") as mock_transfer:
            result = await self.manager.handle(self.phone, "1", trace_id="tt2")  # button id "1" = Yes, send it
        self.assertTrue(result["handled"])
        mock_transfer.assert_called_once()
        self.assertIsNone(get_workflow(self.phone))

    async def test_transfer_success_returns_main_menu_list(self):
        from app.conversation.renderer import ResponseKind, as_structured_response

        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_CONFIRM_TRANSFER, data={
            "beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999",
            "amount": "£25.00", "source_account": "GB12FNCL00010001111111",
        })
        create_workflow(self.phone, workflow)
        with patch("app.workflows.processors.transfer.create_transfer") as mock_transfer, \
             patch("app.workflows.processors.transfer.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result = await self.manager.handle(self.phone, "1", trace_id="tt2a")
        self.assertTrue(result["handled"])
        mock_transfer.assert_called_once()
        structured = as_structured_response(result["response"])
        self.assertEqual(structured.kind, ResponseKind.LIST)
        rows = [row for section in structured.list_sections for row in section.rows]
        self.assertEqual([row.id for row in rows], ["1", "2", "3", "4", "5", "6", "7", "8"])
        self.assertIn("What would you like to do?", structured.text)

    async def test_tapping_edit_amount_returns_to_amount_step(self):
        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_CONFIRM_TRANSFER, data={
            "beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999",
            "amount": "£25.00", "source_account": "GB12FNCL00010001111111",
        })
        create_workflow(self.phone, workflow)
        with patch("app.workflows.processors.transfer.create_transfer") as mock_transfer:
            result = await self.manager.handle(self.phone, "2", trace_id="tt3")  # button id "2" = Edit amount
        self.assertTrue(result["handled"])
        mock_transfer.assert_not_called()
        self.assertEqual(get_workflow(self.phone)["step"], STEP_SELECT_AMOUNT)

    async def test_tapped_quick_amount_row_advances_to_source_account(self):
        workflow = create_workflow_model(WORKFLOW_TRANSFER, STEP_SELECT_AMOUNT, data={
            "beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999",
        })
        create_workflow(self.phone, workflow)
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=self._accounts()):
            result = await self.manager.handle(self.phone, "2", trace_id="tt4")  # row id "2" = £50
        self.assertTrue(result["handled"])
        stored = get_workflow(self.phone)
        self.assertEqual(stored["step"], STEP_SELECT_SOURCE_ACCOUNT)
        self.assertEqual(stored["data"]["amount"], "£50.00")


class ContinueStopButtonTests(ButtonWiringTestCase):
    """The active-workflow interruption confirm buttons — Continue/Stop
    appear when a cancel/closing word is said mid-workflow; Continue/Switch
    when a different service is requested mid-workflow. Both are already
    covered end-to-end in tests/test_response_ux.py's
    LlmFallbackWorkflowTests; these are a focused smoke test scoped to
    just the button ids themselves."""

    async def test_stop_button_id_cancels_and_returns_to_menu(self):
        from app.workflows.constants import WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)
        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            await self.manager.handle(self.phone, "Cancel", trace_id="cs1")  # -> asks Continue/Stop
            result = await self.manager.handle(self.phone, "stop", trace_id="cs2")  # tap "Stop"
        self.assertTrue(result["handled"])
        self.assertIsNone(get_workflow(self.phone))

    async def test_continue_button_id_resumes_the_workflow(self):
        from app.workflows.constants import WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE

        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(self.phone, workflow)
        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            await self.manager.handle(self.phone, "Cancel", trace_id="cs3")  # -> asks Continue/Stop
            result = await self.manager.handle(self.phone, "continue", trace_id="cs4")  # tap "Continue"
        self.assertTrue(result["handled"])
        stored = get_workflow(self.phone)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["step"], STEP_UPLOAD_CHEQUE)


if __name__ == "__main__":
    unittest.main()

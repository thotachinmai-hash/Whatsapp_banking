"""Tests for the "Create Another Account" flow — an already-registered
customer opening a second/third account by reusing the onboarding
workflow's Aadhaar/PAN collection and confirmation steps, without ever
calling create_customer() again. See app/workflows/processors/onboarding.py
(start_add_account_workflow, and the WORKFLOW_ADD_ACCOUNT branches in
_handle_collect_aadhaar/_handle_collect_pan/_handle_confirm_registration/
_handle_select_account_type) and app/workflows/manager.py (menu row "8"),
app/conversation/workflow_adapter.py (natural-language trigger)."""

import unittest
from unittest.mock import patch

from app.conversation.renderer import ResponseKind, as_structured_response
from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_COLLECT_PAN,
    STEP_CONFIRM_REGISTRATION,
    STEP_SELECT_ACCOUNT_TYPE,
    WORKFLOW_ADD_ACCOUNT,
    WORKFLOW_ONBOARDING,
)
from app.workflows.manager import WorkflowManager
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow
from app.workflows.processors.onboarding import OnboardingWorkflowHandler, start_add_account_workflow


class FakeRedis:
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


CUSTOMER = {
    "full_name": "Alex Doe",
    "aadhaar_number": "123456789012",
    "pan_number": "ABCDE1234F",
}


def _accounts(*types):
    return [{"account_number": f"GB{i}", "account_type": t, "balance": "0.00"} for i, t in enumerate(types)]


class StartAddAccountWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"

    async def test_starts_workflow_when_customer_has_no_accounts(self):
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]):
            result = start_add_account_workflow(self.phone, trace_id="t1")

        self.assertTrue(result["handled"])
        self.assertIn("Aadhaar", result["response"])
        workflow = get_workflow(self.phone)
        self.assertIsNotNone(workflow)
        self.assertEqual(workflow["type"], WORKFLOW_ADD_ACCOUNT)
        self.assertEqual(workflow["step"], STEP_COLLECT_AADHAAR)

    async def test_starts_workflow_when_one_type_already_held(self):
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=_accounts("savings")):
            result = start_add_account_workflow(self.phone, trace_id="t2")

        self.assertTrue(result["handled"])
        self.assertIsNotNone(get_workflow(self.phone))

    async def test_no_workflow_started_when_every_type_already_held(self):
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone",
                    return_value=_accounts("savings", "current", "salary")), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value=CUSTOMER):
            result = start_add_account_workflow(self.phone, trace_id="t3")

        self.assertTrue(result["handled"])
        response = as_structured_response(result["response"])
        self.assertIn("already have", response.text.lower())
        self.assertIsNone(get_workflow(self.phone))


class AddAccountIdentityChecksTests(unittest.IsolatedAsyncioTestCase):
    """Aadhaar/PAN uploaded for an additional account must match the
    identity already on file — a fresh registration has nothing to check
    against, so this only applies to WORKFLOW_ADD_ACCOUNT."""

    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.handler = OnboardingWorkflowHandler()

    async def test_mismatched_aadhaar_is_rejected(self):
        workflow = create_workflow_model(WORKFLOW_ADD_ACCOUNT, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)
        parsed_document = {
            "mime_type": "image/jpeg",
            "content": {"aadhaar_number": "999999999999", "full_name": "Alex Doe"},
        }
        with patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value=CUSTOMER):
            result = self.handler.handle(
                {"step": STEP_COLLECT_AADHAAR, "type": WORKFLOW_ADD_ACCOUNT}, self.phone, "", parsed_document, trace_id="t1"
            )

        self.assertTrue(result["handled"])
        self.assertIn("doesn't match", result["response"])
        # Must not have advanced to the PAN step on a rejected document.
        workflow = get_workflow(self.phone)
        self.assertEqual(workflow["step"], STEP_COLLECT_AADHAAR)

    async def test_matching_aadhaar_advances_to_pan(self):
        workflow = create_workflow_model(WORKFLOW_ADD_ACCOUNT, STEP_COLLECT_AADHAAR)
        create_workflow(self.phone, workflow)
        parsed_document = {
            "mime_type": "image/jpeg",
            "content": {"aadhaar_number": "123456789012", "full_name": "Alex Doe"},
        }
        with patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value=CUSTOMER):
            result = self.handler.handle(
                {"step": STEP_COLLECT_AADHAAR, "type": WORKFLOW_ADD_ACCOUNT}, self.phone, "", parsed_document, trace_id="t2"
            )

        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["step"], STEP_COLLECT_PAN)

    async def test_mismatched_pan_is_rejected(self):
        workflow = create_workflow_model(WORKFLOW_ADD_ACCOUNT, STEP_COLLECT_PAN, data={"aadhaar_number": "123456789012"})
        create_workflow(self.phone, workflow)
        parsed_document = {
            "mime_type": "image/jpeg",
            "content": {"pan_number": "ZZZZZ9999Z", "full_name": "Alex Doe"},
        }
        with patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value=CUSTOMER):
            result = self.handler.handle(
                {"step": STEP_COLLECT_PAN, "type": WORKFLOW_ADD_ACCOUNT}, self.phone, "", parsed_document, trace_id="t3"
            )

        self.assertTrue(result["handled"])
        self.assertIn("doesn't match", result["response"])


class AddAccountConfirmSkipsCreateCustomerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.handler = OnboardingWorkflowHandler()

    async def test_confirm_yes_does_not_create_a_new_customer(self):
        workflow_dict = create_workflow_model(
            WORKFLOW_ADD_ACCOUNT, STEP_CONFIRM_REGISTRATION,
            data={"aadhaar_number": "123456789012", "pan_number": "ABCDE1234F", "full_name": "Alex Doe"},
        )
        create_workflow(self.phone, workflow_dict)
        workflow = get_workflow(self.phone)

        with patch("app.workflows.processors.onboarding.create_customer") as mock_create_customer, \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value=CUSTOMER), \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=_accounts("savings")):
            result = self.handler.handle(workflow, self.phone, "yes", None, trace_id="t1")

        mock_create_customer.assert_not_called()
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["step"], STEP_SELECT_ACCOUNT_TYPE)

    async def test_confirm_yes_offers_only_eligible_types(self):
        workflow_dict = create_workflow_model(
            WORKFLOW_ADD_ACCOUNT, STEP_CONFIRM_REGISTRATION,
            data={"aadhaar_number": "123456789012", "pan_number": "ABCDE1234F", "full_name": "Alex Doe"},
        )
        create_workflow(self.phone, workflow_dict)
        workflow = get_workflow(self.phone)

        with patch("app.workflows.processors.onboarding.create_customer") as mock_create_customer, \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value=CUSTOMER), \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=_accounts("savings")):
            result = self.handler.handle(workflow, self.phone, "yes", None, trace_id="t2")

        mock_create_customer.assert_not_called()
        response = as_structured_response(result["response"])
        self.assertEqual(response.kind, ResponseKind.LIST)
        row_titles = {row.title for section in response.list_sections for row in section.rows}
        self.assertEqual(row_titles, {"Current Account", "Salary Account"})


class AddAccountSelectTypeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.handler = OnboardingWorkflowHandler()

    async def test_choosing_an_already_held_type_is_rejected(self):
        workflow_dict = create_workflow_model(WORKFLOW_ADD_ACCOUNT, STEP_SELECT_ACCOUNT_TYPE, data={"full_name": "Alex Doe"})
        create_workflow(self.phone, workflow_dict)
        workflow = get_workflow(self.phone)

        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=_accounts("savings")), \
             patch("app.workflows.processors.onboarding.create_zero_balance_account") as mock_create:
            result = self.handler.handle(workflow, self.phone, "savings", None, trace_id="t1")

        mock_create.assert_not_called()
        self.assertTrue(result["handled"])
        response = as_structured_response(result["response"])
        self.assertIn("already have", response.text.lower())
        row_titles = {row.title for section in response.list_sections for row in section.rows}
        self.assertEqual(row_titles, {"Current Account", "Salary Account"})
        self.assertIsNotNone(get_workflow(self.phone))  # workflow not abandoned

    async def test_choosing_an_eligible_type_creates_the_account(self):
        workflow_dict = create_workflow_model(WORKFLOW_ADD_ACCOUNT, STEP_SELECT_ACCOUNT_TYPE, data={"full_name": "Alex Doe"})
        create_workflow(self.phone, workflow_dict)
        workflow = get_workflow(self.phone)
        fake_account = {"account_number": "GB99", "account_type": "current"}

        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=_accounts("savings")), \
             patch("app.workflows.processors.onboarding.create_zero_balance_account", return_value=fake_account) as mock_create, \
             patch("app.workflows.processors.onboarding.cache_active_account"):
            result = self.handler.handle(workflow, self.phone, "current", None, trace_id="t2")

        mock_create.assert_called_once()
        self.assertEqual(mock_create.call_args.kwargs["account_type"], "current")
        self.assertTrue(result["handled"])
        self.assertIsNone(get_workflow(self.phone))


class MenuAndAdapterWiringTests(unittest.TestCase):
    def test_menu_digit_8_starts_add_account_workflow(self):
        from app.workflows.manager import WorkflowManager

        fake_redis = FakeRedis()
        with patch("app.workflows.memory.redis_client", fake_redis), \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]):
            manager = WorkflowManager()
            result = manager.start_requested("441111111111", "8", trace_id="t1")
            self.assertTrue(result["handled"])
            workflow = get_workflow("441111111111")
            self.assertEqual(workflow["type"], WORKFLOW_ADD_ACCOUNT)

    def test_natural_language_trigger_reaches_add_account_via_adapter(self):
        from app.conversation.workflow_adapter import start_workflow_directly

        fake_redis = FakeRedis()
        with patch("app.workflows.memory.redis_client", fake_redis), \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]):
            result = start_workflow_directly(WORKFLOW_ONBOARDING, "441111111111", query="I'd like to open another account", trace_id="t2")
            self.assertIsNotNone(result)
            self.assertTrue(result["handled"])
            workflow = get_workflow("441111111111")
            self.assertEqual(workflow["type"], WORKFLOW_ADD_ACCOUNT)


class AccountTypeSelectionNotMisroutedTests(unittest.IsolatedAsyncioTestCase):
    """Regression: a bare answer to "which account would you like to
    open?" (e.g. "Current Account") during WORKFLOW_ADD_ACCOUNT's
    SELECT_ACCOUNT_TYPE step must reach the deterministic step processor
    directly, never the mid-workflow LLM switch-detection call — see
    app/workflows/manager.py::_is_account_type_selection_input.

    Confirmed live: without this bypass, "Current Account" was
    misclassified by the LLM router as a side question, handed off to the
    general LLM+tools agent (which has no account-opening tool), and the
    agent hallucinated one — leaking a fake tool-call string to the
    customer instead of actually opening the account."""

    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.manager = WorkflowManager()

    async def test_account_type_phrase_never_reaches_the_llm_router(self):
        create_workflow(self.phone, create_workflow_model(
            WORKFLOW_ADD_ACCOUNT, STEP_SELECT_ACCOUNT_TYPE, data={"full_name": "Alex Doe"},
        ))
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm, \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.create_zero_balance_account", return_value={
                 "account_number": "FNCL000000000099", "account_type": "current",
             }), \
             patch("app.workflows.processors.onboarding.cache_active_account"):
            result = self.manager.handle(self.phone, "Current Account", trace_id="t1", llm_decision=None)

        mock_llm.assert_not_called()
        self.assertTrue(result["handled"])
        # The workflow actually completed (account opened) rather than
        # being handed off elsewhere.
        self.assertIsNone(get_workflow(self.phone))

    async def test_bare_digit_account_type_also_never_reaches_the_llm_router(self):
        create_workflow(self.phone, create_workflow_model(
            WORKFLOW_ADD_ACCOUNT, STEP_SELECT_ACCOUNT_TYPE, data={"full_name": "Alex Doe"},
        ))
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm, \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.create_zero_balance_account", return_value={
                 "account_number": "FNCL000000000099", "account_type": "savings",
             }), \
             patch("app.workflows.processors.onboarding.cache_active_account"):
            result = self.manager.handle(self.phone, "1", trace_id="t2", llm_decision=None)

        mock_llm.assert_not_called()
        self.assertTrue(result["handled"])


if __name__ == "__main__":
    unittest.main()

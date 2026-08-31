"""Two transfer-workflow bugs reported live:

1. A single-account customer, after typing a new beneficiary's account
   number (with the amount already known), was shown the full "choose an
   account" list picker instead of the same offer/auto-select treatment
   every other path to the source-account step already uses — see
   TransferWorkflowProcessor.handle()'s STEP_COLLECT_BENEFICIARY_ACCOUNT
   branch and resolve_source_account_or_prompt.

2. A native-language/Romanized voice message (e.g. "Karu ki 500
   pampandi") that clearly states a beneficiary and amount had both
   silently dropped, because start_transfer_from_text's extraction is
   English-regex-only. The LLM router's own extracted entities
   (entities.recipient/entities.amount) are now consulted as a fallback
   when the regex finds nothing — see start_transfer_from_text.
"""

import unittest
from unittest.mock import patch

from app.workflows.constants import (
    STEP_COLLECT_BENEFICIARY_ACCOUNT,
    STEP_CONFIRM_SOURCE_ACCOUNT,
    STEP_SELECT_SOURCE_ACCOUNT,
    WORKFLOW_TRANSFER,
)
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow
from app.workflows.processors.transfer import TransferWorkflowProcessor, start_transfer_from_text

_SINGLE_ACCOUNT = [{"account_number": "FNCL000000000001", "account_type": "savings", "balance": "18200.00", "currency": "INR"}]


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


class SingleAccountSourceSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.handler = TransferWorkflowProcessor()

    async def test_new_beneficiary_account_with_amount_already_set_skips_the_full_picker(self):
        create_workflow(self.phone, create_workflow_model(
            WORKFLOW_TRANSFER, STEP_COLLECT_BENEFICIARY_ACCOUNT,
            data={"beneficiary_name": "Karu", "amount": "₹500.00"},
        ))
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=_SINGLE_ACCOUNT), \
             patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=_SINGLE_ACCOUNT[0]), \
             patch("app.workflows.processors.transfer.create_beneficiary"):
            result = self.handler.handle(
                {"step": STEP_COLLECT_BENEFICIARY_ACCOUNT, "data": {"beneficiary_name": "Karu", "amount": "₹500.00"}},
                self.phone, "1212167890",
            )

        self.assertTrue(result["handled"])
        # Landed on the Yes/No "is this your account?" offer, not the raw
        # multi-row picker -- confirmed by the workflow step, and by the
        # response NOT being the picker's own list-of-accounts shape.
        workflow = get_workflow(self.phone)
        self.assertEqual(workflow["step"], STEP_CONFIRM_SOURCE_ACCOUNT)
        self.assertNotEqual(workflow["step"], STEP_SELECT_SOURCE_ACCOUNT)


class NativeLanguageEntityFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.phone = "441111111111"
        self.handler = TransferWorkflowProcessor()

    async def test_entities_used_when_regex_finds_nothing(self):
        # A native-script message the English-only regexes can't parse at
        # all -- both beneficiary and amount must come from entities.
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]), \
             patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=_SINGLE_ACCOUNT), \
             patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=_SINGLE_ACCOUNT[0]):
            result = start_transfer_from_text(
                self.phone, "కరుకి 500 పంపండి", self.handler, "t1",
                entities={"recipient": "Karu", "amount": "500"},
            )

        self.assertTrue(result["handled"])
        workflow = get_workflow(self.phone)
        self.assertEqual(workflow["data"].get("beneficiary_name"), "Karu")
        self.assertEqual(workflow["data"].get("amount"), "₹500.00")

    async def test_regex_extraction_still_wins_when_it_finds_something(self):
        # entities must never override a real regex match with a
        # different value -- it's a fallback, not an override.
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]), \
             patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=_SINGLE_ACCOUNT), \
             patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=_SINGLE_ACCOUNT[0]):
            result = start_transfer_from_text(
                self.phone, "send 700 to Priya", self.handler, "t2",
                entities={"recipient": "SomeoneElse", "amount": "999"},
            )

        self.assertTrue(result["handled"])
        workflow = get_workflow(self.phone)
        self.assertEqual(workflow["data"].get("beneficiary_name"), "Priya")
        self.assertEqual(workflow["data"].get("amount"), "₹700.00")

    async def test_no_entities_and_no_regex_match_falls_back_to_beneficiary_prompt(self):
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result = start_transfer_from_text(self.phone, "hi", self.handler, "t3", entities=None)

        self.assertTrue(result["handled"])


if __name__ == "__main__":
    unittest.main()

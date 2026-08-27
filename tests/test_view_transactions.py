"""Tests for the "View transactions" main-menu flow (row "3").

See app/workflows/processors/transactions.py and app/workflows/manager.py's
menu_actions["3"] — this used to fall through to the LLM+tools agent via
reprocess_query, which sometimes answered "no account linked" without ever
calling a tool. It's now deterministic, matching the transfer workflow's
own source-account picker."""

import unittest
from decimal import Decimal
from unittest.mock import patch

from app.conversation.renderer import ResponseKind, as_structured_response
from app.workflows import manager as workflow_manager_module
from app.workflows.constants import STEP_SELECT_TRANSACTIONS_ACCOUNT, WORKFLOW_VIEW_TRANSACTIONS
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
        self._store.pop(key, None)
        return True


_SAVINGS = {
    "account_number": "FNCL000000000001", "account_type": "savings",
    "balance": Decimal("20000.00"), "currency": "INR", "status": "active",
}
_CURRENT = {
    "account_number": "FNCL000000000002", "account_type": "current",
    "balance": Decimal("5000.00"), "currency": "INR", "status": "active",
}
_TRANSACTIONS = [
    {
        "transaction_type": "credit", "category": "salary", "amount": Decimal("2500.00"),
        "description": "Salary credit", "created_at": "2026-08-01 10:00:00", "balance_after": Decimal("20000.00"),
    },
]


class ViewTransactionsMenuTests(unittest.TestCase):
    def setUp(self):
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = WorkflowManager()

    def test_single_account_shows_transactions_directly_no_workflow(self):
        with patch("app.workflows.processors.transactions.get_accounts_by_phone", return_value=[_SAVINGS]), \
             patch("app.workflows.processors.transactions.get_account_by_number", return_value={**_SAVINGS, "id": 1}), \
             patch("app.workflows.processors.transactions.get_transactions", return_value=_TRANSACTIONS):
            result = self.manager.start_requested("441111111111", "3", trace_id="tv1")

        self.assertTrue(result["handled"])
        structured = as_structured_response(result["response"])
        self.assertIn("Salary credit", structured.text)
        self.assertIsNone(get_workflow("441111111111"))

    def test_multiple_accounts_prompts_account_choice(self):
        with patch("app.workflows.processors.transactions.get_accounts_by_phone", return_value=[_SAVINGS, _CURRENT]):
            result = self.manager.start_requested("441111111111", "3", trace_id="tv2")

        self.assertTrue(result["handled"])
        structured = as_structured_response(result["response"])
        self.assertEqual(structured.kind, ResponseKind.LIST)
        workflow = get_workflow("441111111111")
        self.assertEqual(workflow["type"], WORKFLOW_VIEW_TRANSACTIONS)
        self.assertEqual(workflow["step"], STEP_SELECT_TRANSACTIONS_ACCOUNT)

    def test_no_accounts_returns_plain_message(self):
        with patch("app.workflows.processors.transactions.get_accounts_by_phone", return_value=[]):
            result = self.manager.start_requested("441111111111", "3", trace_id="tv3")

        self.assertTrue(result["handled"])
        self.assertIn("no active account", result["response"].lower())

    def test_selecting_account_row_shows_its_transactions_and_completes_workflow(self):
        workflow = create_workflow_model(WORKFLOW_VIEW_TRANSACTIONS, STEP_SELECT_TRANSACTIONS_ACCOUNT)
        create_workflow("441111111111", workflow)

        with patch("app.workflows.processors.transactions.get_accounts_by_phone", return_value=[_SAVINGS, _CURRENT]), \
             patch("app.workflows.processors.transactions.get_account_by_number", return_value={**_CURRENT, "id": 2}), \
             patch("app.workflows.processors.transactions.get_transactions", return_value=_TRANSACTIONS):
            result = self.manager.handle(
                phone_number="441111111111", query="vtxn_2", trace_id="tv4"
            )

        self.assertTrue(result["handled"])
        structured = as_structured_response(result["response"])
        self.assertIn("Salary credit", structured.text)
        self.assertIsNone(get_workflow("441111111111"))


if __name__ == "__main__":
    unittest.main()

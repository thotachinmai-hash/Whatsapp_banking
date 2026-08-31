"""tool_check_loan_eligibility -- a personalized loan-borrowing estimate
computed from the customer's own transaction history (average monthly
income) and the bank's published loan terms. See app/agent/tools.py.

This is explicitly an ESTIMATE, never an approval -- app/agent/agent.py's
system prompt carve-out (next to the existing get_loan_product_info
guardrail) tells the LLM to relay it as such."""

import unittest
from unittest.mock import patch

from app.agent.tools import tool_check_loan_eligibility

_ACCOUNT = {"id": 1, "account_number": "FNCL000000000001", "account_type": "savings"}
_PRODUCT = {
    "loan_type": "personal",
    "display_name": "Personal Loan",
    "interest_rate_min": 10.50,
    "interest_rate_max": 15.00,
    "min_amount": 10000.00,
    "max_amount": 1000000.00,
    "min_tenure_months": 2,
    "max_tenure_months": 60,
    "processing_fee_percent": 1.50,
    "currency": "INR",
    "notes": "",
}


def _tx(amount: float) -> dict:
    return {"amount": amount}


class LoanEligibilityToolTests(unittest.TestCase):
    def test_invalid_loan_type_is_rejected_before_any_db_call(self) -> None:
        result = tool_check_loan_eligibility(loan_type="mortgage-for-a-yacht", phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])
        self.assertIn("isn't a loan type", result["message"])

    def test_missing_product_reports_not_found(self) -> None:
        with patch("app.agent.tools.get_loan_product", return_value=None):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])

    def test_multiple_accounts_asks_which_one(self) -> None:
        accounts = [
            {"account_number": "A1", "account_type": "savings"},
            {"account_number": "A2", "account_type": "current"},
        ]
        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=accounts):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertTrue(result.get("multiple_accounts"))
        self.assertEqual(len(result["accounts"]), 2)

    def test_no_transaction_history_reports_not_found_with_guidance(self) -> None:
        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", return_value=[]):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])
        self.assertIn("monthly income", result["message"])

    def test_salary_credits_preferred_over_all_credits(self) -> None:
        salary = [_tx(50000), _tx(50000)]
        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", return_value=salary) as mock_tx:
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")

        self.assertTrue(result["found"])
        self.assertEqual(result["income_basis"], "salary_credits")
        self.assertEqual(result["avg_monthly_income"], 50000.0)
        # First call must have asked for category="salary" specifically.
        first_call_kwargs = mock_tx.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs.get("category"), "salary")

    def test_falls_back_to_all_credits_when_no_salary_history(self) -> None:
        def fake_get_transactions(account_id, limit=100, transaction_type=None, category=None):
            if category == "salary":
                return []
            return [_tx(20000), _tx(30000)]

        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", side_effect=fake_get_transactions):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")

        self.assertTrue(result["found"])
        self.assertEqual(result["income_basis"], "all_credits_estimate")
        self.assertEqual(result["avg_monthly_income"], 25000.0)

    def test_estimate_math_matches_amortization_formula(self) -> None:
        # avg_monthly_income = 50000 -> EMI capacity = 25000.
        # midpoint rate = (10.5+15.0)/2 = 12.75% APR -> monthly r = 0.010625.
        # n = 60. P = 25000 * (1 - (1+r)^-60) / r.
        r = ((10.50 + 15.00) / 2) / 12 / 100
        n = 60
        expected = min(25000 * (1 - (1 + r) ** -n) / r, _PRODUCT["max_amount"])

        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", return_value=[_tx(50000), _tx(50000)]):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")

        self.assertAlmostEqual(result["estimated_max_eligible_amount"], round(expected, 2), places=1)

    def test_estimate_is_capped_at_product_max_amount_never_inflated_to_min(self) -> None:
        # A huge income should still cap at the product's max_amount.
        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", return_value=[_tx(10_000_000)]):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertEqual(result["estimated_max_eligible_amount"], _PRODUCT["max_amount"])

        # A tiny income should NOT be inflated up to the product minimum.
        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", return_value=[_tx(10)]):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertLess(result["estimated_max_eligible_amount"], _PRODUCT["min_amount"])
        self.assertFalse(result["meets_product_minimum"])

    def test_result_always_carries_a_disclaimer(self) -> None:
        with patch("app.agent.tools.get_loan_product", return_value=_PRODUCT), \
             patch("app.agent.tools.get_accounts_by_phone", return_value=[_ACCOUNT]), \
             patch("app.agent.tools.get_account_by_number", return_value=_ACCOUNT), \
             patch("app.agent.tools.get_transactions", return_value=[_tx(50000)]):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertIn("not an approval", result["disclaimer"].lower())

    def test_db_failure_is_marked_as_error_not_silently_reshaped(self) -> None:
        with patch("app.agent.tools.get_loan_product", side_effect=Exception("connection timeout")):
            result = tool_check_loan_eligibility(loan_type="personal", phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])
        self.assertTrue(result.get("error"))


if __name__ == "__main__":
    unittest.main()

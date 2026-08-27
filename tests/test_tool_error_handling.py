"""A tool's normal "nothing found" result and a genuine technical failure
(a DB connection blip, a query timeout) used to be structurally identical
to the LLM -- both `{"found": False, "message": "..."}`, differing only in
wording the model had to parse correctly every time. Confirmed live in
production: an already-registered, active customer (verified directly
against the database -- customer and account records both existed, account
status 'active') was told "you don't have a registered bank account,
please contact your bank branch" after asking for their transaction
history -- the underlying DB call almost certainly hit a transient error
that got misread as "not found" rather than "a technical problem
occurred."

app/agent/tools.py::_tool_error() now marks a genuine technical failure
with "error": True and an explicit, self-contained instruction, and
app/agent/agent.py's system prompt has a matching TECHNICAL ERROR RULE
telling the LLM to relay it honestly rather than reinterpreting it as
"not registered." See the docstring on _tool_error itself for the full
incident writeup."""

import unittest
from unittest.mock import patch

from app.agent.tools import _tool_error, tool_get_account_balance, tool_list_beneficiaries


class ToolErrorHelperTests(unittest.TestCase):
    def test_error_result_is_clearly_marked(self) -> None:
        result = _tool_error("get_account_balance", "retrieving your account balance", Exception("boom"), "t1")
        self.assertFalse(result["found"])
        self.assertTrue(result["error"])
        self.assertIn("temporary technical problem", result["message"].lower())
        self.assertIn("not related to registration", result["message"].lower())

    def test_message_never_implies_unregistered_or_branch_visit(self) -> None:
        result = _tool_error("get_last_transactions", "retrieving your transactions", Exception("boom"), "t1")
        text = result["message"].lower()
        self.assertNotIn("not registered", text)
        self.assertNotIn("branch", text)


class ToolFunctionsMarkRealFailuresTests(unittest.TestCase):
    """A couple of the actual tool functions, confirming a real DB
    exception reaches the caller as an "error": True result -- not
    silently reshaped into the same "nothing found" shape a genuinely
    empty/unregistered lookup produces."""

    def test_get_account_balance_db_failure_is_marked_as_error(self) -> None:
        with patch("app.agent.tools.get_accounts_by_phone", side_effect=Exception("connection timeout")):
            result = tool_get_account_balance(phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])
        self.assertTrue(result.get("error"))

    def test_list_beneficiaries_db_failure_is_marked_as_error(self) -> None:
        with patch("app.agent.tools.get_beneficiaries_by_phone", side_effect=Exception("connection timeout")):
            result = tool_list_beneficiaries(phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])
        self.assertTrue(result.get("error"))

    def test_genuinely_no_beneficiaries_is_not_marked_as_error(self) -> None:
        # The other shape must stay distinct: a clean, real "nothing
        # here" result has no "error" key at all.
        with patch("app.agent.tools.get_beneficiaries_by_phone", return_value=[]):
            result = tool_list_beneficiaries(phone_number="441111111111", trace_id="t1")
        self.assertFalse(result["found"])
        self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()

"""_beneficiary_list_response() (app/agent/agent.py) — turns a successful
list_beneficiaries tool call into a real tappable WhatsApp list instead of
plain LLM prose, using "Transfer to {name}" as each row's id specifically
because that phrase already round-trips correctly through the existing,
unrelated start_transfer_from_text free-text trigger — see the function's
own docstring for why a bare digit id would be unsafe here (no active
workflow exists yet when this listing is shown, so a bare digit could
collide with the main menu's own digit map)."""

import unittest
from types import SimpleNamespace

from app.agent.agent import _beneficiary_list_response
from app.conversation.renderer import ResponseKind


def _tool_message(name: str, artifact) -> SimpleNamespace:
    return SimpleNamespace(name=name, artifact=artifact)


class BeneficiaryListResponseTests(unittest.TestCase):
    def test_builds_list_with_safe_row_ids(self) -> None:
        messages = [
            _tool_message("list_beneficiaries", {
                "found": True,
                "count": 2,
                "beneficiaries": [
                    {"name": "Priya", "account_number_masked": "•••• 4567", "bank_name": "Finacle"},
                    {"name": "Amit", "account_number_masked": "•••• 9999", "bank_name": "Finacle"},
                ],
            }),
        ]
        response = _beneficiary_list_response(messages, "Here you go:")
        self.assertIsNotNone(response)
        self.assertEqual(response.kind, ResponseKind.LIST)
        rows = [row for section in response.list_sections for row in section.rows]
        self.assertEqual([row.id for row in rows], ["Transfer to Priya", "Transfer to Amit"])
        self.assertEqual([row.title for row in rows], ["Priya", "Amit"])
        # Never a bare digit -- the one thing this whole feature exists to avoid.
        for row in rows:
            self.assertFalse(row.id.isdigit())

    def test_no_tool_call_returns_none(self) -> None:
        self.assertIsNone(_beneficiary_list_response([], "text"))

    def test_not_found_returns_none(self) -> None:
        messages = [_tool_message("list_beneficiaries", {"found": False, "message": "No saved beneficiaries yet."})]
        self.assertIsNone(_beneficiary_list_response(messages, "text"))

    def test_empty_beneficiaries_returns_none(self) -> None:
        messages = [_tool_message("list_beneficiaries", {"found": True, "beneficiaries": []})]
        self.assertIsNone(_beneficiary_list_response(messages, "text"))

    def test_over_ten_beneficiaries_falls_back_to_none(self) -> None:
        many = [{"name": f"Person{i}", "account_number_masked": "•••• 0000"} for i in range(11)]
        messages = [_tool_message("list_beneficiaries", {"found": True, "beneficiaries": many})]
        self.assertIsNone(_beneficiary_list_response(messages, "text"))

    def test_unrelated_tool_call_returns_none(self) -> None:
        messages = [_tool_message("get_account_balance", {"found": True, "balance": "100.00"})]
        self.assertIsNone(_beneficiary_list_response(messages, "text"))

    def test_malformed_artifact_returns_none(self) -> None:
        messages = [_tool_message("list_beneficiaries", None)]
        self.assertIsNone(_beneficiary_list_response(messages, "text"))


if __name__ == "__main__":
    unittest.main()

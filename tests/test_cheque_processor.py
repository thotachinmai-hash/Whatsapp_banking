import unittest
from unittest.mock import patch

from app.workflows.processors.cheque import ChequeWorkflowProcessor


class ChequeProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ChequeWorkflowProcessor()

    @patch("app.workflows.processors.cheque.set_workflow_step")
    @patch("app.workflows.processors.cheque.update_workflow_data")
    @patch("app.workflows.processors.cheque.get_customer_by_phone")
    def test_rejects_cheque_when_payee_does_not_match_customer(
        self,
        mock_get_customer_by_phone,
        mock_update_workflow_data,
        mock_set_workflow_step,
    ) -> None:
        mock_get_customer_by_phone.return_value = {"full_name": "John Smith"}

        result = self.processor._validate_or_finalize(
            "447000000000",
            {"payee": "Jane Doe", "amount_in_figures": "250.00"},
        )

        self.assertTrue(result["handled"])
        self.assertIn("payee", result["response"].lower())
        self.assertIn("john smith", result["response"].lower())
        mock_update_workflow_data.assert_called_once()
        mock_set_workflow_step.assert_called_once()

    @patch("app.workflows.processors.cheque.complete_workflow")
    @patch("app.workflows.processors.cheque.create_cheque_request")
    @patch("app.workflows.processors.cheque.get_customer_by_phone")
    def test_accepts_matching_payee_and_surfaces_extra_cheque_details(
        self,
        mock_get_customer_by_phone,
        mock_create_cheque_request,
        mock_complete_workflow,
    ) -> None:
        mock_get_customer_by_phone.return_value = {"full_name": "John Smith"}
        mock_create_cheque_request.return_value = {"request_id": "CHQ-1234"}

        result = self.processor._validate_or_finalize(
            "447000000000",
            {
                "payee": "John Smith",
                "amount_in_figures": "250.00",
                "date_written": "14/08/2026",
                "drawer_name": "Alice Brown",
            },
        )

        self.assertTrue(result["handled"])
        self.assertIn("request id", result["response"].lower())
        self.assertIn("date", result["response"].lower())
        self.assertIn("drawer", result["response"].lower())
        mock_create_cheque_request.assert_called_once()
        mock_complete_workflow.assert_called_once_with("447000000000")

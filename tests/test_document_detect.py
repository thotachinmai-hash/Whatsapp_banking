import unittest

from app.workflows.document_detect import detect_workflow_type


class DocumentDetectTests(unittest.TestCase):
    def test_trusts_declared_document_type(self) -> None:
        self.assertEqual(detect_workflow_type({"document_type": "cheque"}), "cheque")
        self.assertEqual(detect_workflow_type({"document_type": "kyc"}), "kyc")
        self.assertEqual(detect_workflow_type({"document_type": "loan_form"}), "loan_form")

    def test_ignores_unrecognized_declared_type_and_falls_back_to_fields(self) -> None:
        content = {
            "document_type": "other",
            "payee": "John Doe",
            "amount_in_figures": "5000",
        }
        self.assertEqual(detect_workflow_type(content), "cheque")

    def test_falls_back_to_field_presence_for_kyc(self) -> None:
        content = {"aadhaar_number": "123456789012", "pan_number": "ABCDE1234F"}
        self.assertEqual(detect_workflow_type(content), "kyc")

    def test_falls_back_to_field_presence_for_loan_form(self) -> None:
        content = {
            "applicant_name": "Jane Doe",
            "monthly_income": "50000",
            "requested_amount": "200000",
        }
        self.assertEqual(detect_workflow_type(content), "loan_form")

    def test_single_stray_field_is_not_enough(self) -> None:
        # One field alone (e.g. a name) isn't confident enough to guess a
        # whole workflow from.
        self.assertIsNone(detect_workflow_type({"full_name": "Jane Doe"}))

    def test_empty_or_non_dict_content(self) -> None:
        self.assertIsNone(detect_workflow_type({}))
        self.assertIsNone(detect_workflow_type(None))


if __name__ == "__main__":
    unittest.main()

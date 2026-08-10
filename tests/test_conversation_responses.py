import unittest

from app.conversation.responses import cheque, common, errors, kyc, loan, onboarding, status, transfer


class CommonTemplateTests(unittest.TestCase):
    def test_01_main_menu_renders(self):
        text = common.render_main_menu("John", greeting=True)
        self.assertIn("John", text)
        self.assertIn("Transfer money", text)
        self.assertIn("1.", text)

    def test_02_greeting_renders(self):
        self.assertIn("Hi", common.render_greeting("John"))
        self.assertIn("John", common.render_greeting("John"))
        self.assertIn("Hi there", common.render_greeting())

    def test_03_out_of_scope_renders(self):
        text = common.render_out_of_scope()
        self.assertIn("banking", text.lower())
        for forbidden in ("intent", "classifier", "router", "llm", "ai routing"):
            self.assertNotIn(forbidden, text.lower())

    def test_04_clarification_renders(self):
        text = common.render_clarification("transfer_request")
        self.assertIn("send money", text.lower())
        generic = common.render_clarification("nonexistent_intent")
        self.assertTrue(generic)
        self.assertNotEqual(generic, text)


class TransferTemplateTests(unittest.TestCase):
    def test_05_transfer_summary(self):
        text = transfer.render_transfer_summary(
            beneficiary_name="Priya",
            beneficiary_account_masked="•••• 1234",
            amount_label="£500.00",
            source_account_label="GB12FNCL00010001234567",
        )
        self.assertIn("Priya", text)
        self.assertIn("£500.00", text)
        self.assertIn("Please check these details", text)

    def test_06_transfer_confirmation(self):
        text = transfer.render_transfer_confirmation()
        self.assertIn("Confirm transfer", text)
        self.assertIn("Reply 1 or 2", text)

    def test_07_transfer_success(self):
        text = transfer.render_transfer_success(
            reference="TRF-ABCD1234",
            beneficiary_name="Priya",
            beneficiary_account_masked="1234",
            amount_label="£500.00",
            source_account_label="GB12FNCL00010001234567",
        )
        self.assertIn("TRF-ABCD1234", text)
        self.assertIn("initiated", text.lower())
        self.assertNotIn("completed", text.lower())

    def test_08_insufficient_balance(self):
        text_generic = transfer.render_insufficient_balance()
        self.assertIn("sufficient balance", text_generic.lower())
        text_specific = transfer.render_insufficient_balance(
            account_label="GB12FNCL00010001234567", available_label="£10.00", amount_label="£500.00"
        )
        self.assertIn("£10.00", text_specific)
        self.assertIn("£500.00", text_specific)
        self.assertNotIn("psycopg2", text_specific)
        self.assertNotIn("Error", text_specific)


class LoanTemplateTests(unittest.TestCase):
    def test_09_loan_summary(self):
        text = loan.render_loan_summary(
            loan_type="personal",
            account_number="GB12FNCL00010001234567",
            applicant_name="John Smith",
            monthly_income="3000",
            requested_amount="10000",
            tenure_months="24",
            employment_type="Salaried",
            purpose="Home renovation",
        )
        self.assertIn("Personal Loan", text)
        self.assertIn("John Smith", text)
        self.assertIn("verify your loan application", text.lower())

    def test_10_loan_confirmation(self):
        text = loan.render_loan_confirmation()
        self.assertIn("YES", text)
        self.assertIn("NO", text)

    def test_11_loan_eligibility_guidance_never_claims_approval(self):
        text = loan.render_loan_eligibility_guidance(loan_type="personal", monthly_income_label="£5,000")
        lowered = text.lower()
        self.assertNotIn("congratulations", lowered)
        self.assertNotIn("you are eligible", lowered)
        self.assertNotIn("approved", lowered)
        self.assertIn("eligibility", lowered)
        self.assertIn("credit", lowered)
        self.assertIn("requirements", lowered)
        # Offers a next step rather than deciding one for the customer.
        self.assertIn("?", text)


class ChequeTemplateTests(unittest.TestCase):
    def test_12_cheque_confirmation_flow(self):
        summary = cheque.render_cheque_summary(
            request_id="CHQ-ABCD1234",
            payee="John Smith",
            amount_label="£500.00",
            date_written="14/08/2026",
            drawer_name="Alice Brown",
            bank_name="Finacle Banking",
        )
        self.assertIn("CHQ-ABCD1234", summary)
        self.assertIn("PENDING", summary)
        self.assertIn("£500.00", summary)


class KYCTemplateTests(unittest.TestCase):
    def test_13_kyc_confirmation(self):
        summary = kyc.render_kyc_summary("John Smith", "1990-01-01", "1 Test Street")
        self.assertIn("John Smith", summary)
        self.assertNotIn("123456789012", summary)
        confirmation = kyc.render_kyc_confirmation()
        self.assertIn("YES", confirmation)


class OnboardingTemplateTests(unittest.TestCase):
    def test_14_onboarding_confirmation_no_digits(self):
        summary = onboarding.render_registration_summary(
            phone_number="447000000000",
            full_name="John Smith",
            date_of_birth="1990-01-01",
            guardian_name="Robert Smith",
            address="1 Test Street",
        )
        self.assertIn("John Smith", summary)
        self.assertIn("Provided", summary)
        self.assertNotIn("123456789012", summary)
        confirmation = onboarding.render_registration_confirmation()
        self.assertIn("YES", confirmation)
        self.assertIn("NO", confirmation)


class StatusTemplateTests(unittest.TestCase):
    def test_15_account_summary(self):
        text = status.render_account_summary([
            {"account_number": "GB12FNCL00010001234567", "account_type": "savings"},
        ])
        self.assertIn("GB12FNCL00010001234567", text)
        self.assertIn("Savings Account", text)
        empty = status.render_account_summary([])
        self.assertIn("No active accounts", empty)

    def test_16_transaction_formatting(self):
        text = status.render_transaction_list([
            {"type": "debit", "amount": 45.99, "description": "Tesco Superstore", "date": "2026-07-29", "currency": "GBP"},
            {"type": "credit", "amount": 2500, "description": "Salary Payment", "date": "2026-07-11", "currency": "GBP"},
        ])
        self.assertIn("Tesco Superstore", text)
        self.assertIn("-£45.99", text)
        self.assertIn("+£2,500.00", text)
        self.assertNotIn("45.99000", text)


class FormattingHelperTests(unittest.TestCase):
    def test_17_currency_formatting(self):
        self.assertEqual(common.format_currency(500, "GBP"), "£500.00")
        self.assertEqual(common.format_currency(1250, "GBP"), "£1,250.00")
        self.assertEqual(common.format_currency(50000, "GBP"), "£50,000.00")
        self.assertEqual(common.format_currency(500.0, "GBP"), "£500.00")
        self.assertNotIn("500.0", common.format_currency(500, "GBP").replace("£500.00", ""))
        self.assertEqual(common.format_currency(5000, "INR"), "₹5,000.00")

    def test_18_account_masking(self):
        masked = common.mask_account_number("GB12FNCL00010001234567")
        self.assertTrue(masked.endswith("4567"))
        self.assertNotIn("GB12FNCL0001", masked)
        self.assertEqual(common.mask_account_number(""), "")

    def test_18b_amount_formatting_no_python_repr(self):
        self.assertEqual(common.format_amount(500.0), "500")
        self.assertEqual(common.format_amount(24), "24")
        self.assertNotIn(".0", common.format_amount(500.0))


class SensitiveDataProtectionTests(unittest.TestCase):
    """Test 19: sensitive values are never included in generated messages."""

    def test_19_onboarding_summary_never_includes_aadhaar_or_pan_digits(self):
        text = onboarding.render_registration_summary(
            phone_number="447000000000", full_name="John Smith",
        )
        # Mentioning that Aadhaar/PAN were received is fine (and expected);
        # what must never appear is the actual digits/value.
        self.assertNotIn("123456789012", text)
        self.assertNotIn("ABCDE1234F", text)
        self.assertIn("Aadhaar", text)
        self.assertIn("Provided", text)

    def test_19_kyc_summary_never_includes_aadhaar_or_pan(self):
        text = kyc.render_kyc_summary("John Smith", "1990-01-01", "1 Test Street")
        self.assertNotIn("aadhaar:", text.lower().replace("aadhaar: provided", ""))

    def test_19_no_template_function_accepts_aadhaar_or_pan_parameters(self):
        import inspect

        forbidden_params = {"aadhaar_number", "pan_number", "otp", "pin", "cvv", "password"}
        for module in (onboarding, kyc, common, transfer, loan, cheque, status, errors):
            for name, func in inspect.getmembers(module, inspect.isfunction):
                params = set(inspect.signature(func).parameters)
                overlap = params & forbidden_params
                self.assertFalse(
                    overlap, f"{module.__name__}.{name} accepts sensitive parameter(s): {overlap}"
                )

    def test_19_beneficiary_selection_masks_account_numbers(self):
        text = transfer.render_beneficiary_selection([
            {"beneficiary_name": "Priya", "account_number": "GB12FNCL00019999999"}
        ])
        self.assertNotIn("GB12FNCL0001999", text)
        self.assertIn("9999", text)


class ErrorTemplateTests(unittest.TestCase):
    """Test 20: error responses never expose internal exceptions."""

    def test_20_error_templates_never_expose_internals(self):
        renderers = [
            errors.render_invalid_input,
            errors.render_missing_information,
            errors.render_document_error,
            errors.render_transcription_error,
            errors.render_service_error,
            errors.render_rate_limit_error,
            errors.render_database_error,
            errors.render_unknown_error,
        ]
        forbidden = (
            "traceback", "exception", "psycopg2", "redis.exceptions", "groq.error",
            "stack trace", "nonetype", "keyerror", "valueerror", "api_key", "sk-",
        )
        for render in renderers:
            text = render().lower()
            for term in forbidden:
                self.assertNotIn(term, text, f"{render.__name__} leaked '{term}'")

    def test_20_database_error_is_generic(self):
        text = errors.render_database_error()
        self.assertIn("try again", text.lower())
        self.assertNotIn("unique", text.lower())
        self.assertNotIn("violation", text.lower())


if __name__ == "__main__":
    unittest.main()

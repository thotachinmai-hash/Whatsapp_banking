import unittest
from unittest.mock import patch

from app.conversation.builder import build_context
from app.conversation.context import ConversationContext
from app.conversation.context_store import ConversationContextStore


class _FakeRedis:
    """Minimal in-memory stand-in for the subset of redis-py used by
    ConversationContextStore, so these tests never need a real Redis
    server — matching the existing test suite's approach of mocking
    collaborators rather than spinning up real infrastructure."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)

    def exists(self, key):
        return 1 if key in self.store else 0


class ConversationContextModelTests(unittest.TestCase):
    def test_new_context_creation_has_expected_defaults(self) -> None:
        context = ConversationContext(phone_number="447000000000")

        self.assertEqual(context.phone_number, "447000000000")
        self.assertIsNone(context.customer_id)
        self.assertFalse(context.is_registered)
        self.assertIsNone(context.current_workflow)
        self.assertIsNone(context.current_step)
        self.assertEqual(context.workflow_data, {})
        self.assertEqual(context.retry_count, 0)
        self.assertEqual(context.allowed_actions, [])
        self.assertIsNotNone(context.created_at)
        self.assertIsNotNone(context.updated_at)

    def test_sensitive_fields_not_included_in_serialized_context(self) -> None:
        context = ConversationContext(
            phone_number="447000000000",
            workflow_data={"full_name": "John Smith", "loan_type": "personal"},
        )
        serialized = context.model_dump_json()

        for sensitive in ("aadhaar", "pan_number", "otp", "password", "api_key", "cvv"):
            self.assertNotIn(sensitive, serialized.lower())

        # No dedicated top-level field for any banking-PII value either.
        for field_name in ConversationContext.model_fields:
            self.assertNotIn("aadhaar", field_name.lower())
            self.assertNotIn("pan_number", field_name.lower())
            self.assertNotIn("password", field_name.lower())


class ConversationContextStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_redis = _FakeRedis()
        patcher = patch("app.conversation.context_store.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ConversationContextStore()

    def test_save_and_retrieval_round_trips(self) -> None:
        context = ConversationContext(phone_number="447111111111", last_intent="check_balance")

        self.assertTrue(self.store.save(context))
        loaded = self.store.get("447111111111")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.phone_number, "447111111111")
        self.assertEqual(loaded.last_intent, "check_balance")

    def test_update_merges_fields_into_existing_context(self) -> None:
        context = ConversationContext(phone_number="447222222222")
        self.store.save(context)

        updated = self.store.update("447222222222", last_intent="transfer_money", retry_count=2)

        self.assertIsNotNone(updated)
        self.assertEqual(updated.last_intent, "transfer_money")
        self.assertEqual(updated.retry_count, 2)

        reloaded = self.store.get("447222222222")
        self.assertEqual(reloaded.last_intent, "transfer_money")
        self.assertEqual(reloaded.retry_count, 2)

    def test_update_with_no_existing_context_is_a_safe_noop(self) -> None:
        result = self.store.update("447000000099", last_intent="check_balance")
        self.assertIsNone(result)

    def test_clear_removes_context(self) -> None:
        context = ConversationContext(phone_number="447333333333")
        self.store.save(context)
        self.assertTrue(self.store.exists("447333333333"))

        self.store.clear("447333333333")

        self.assertIsNone(self.store.get("447333333333"))
        self.assertFalse(self.store.exists("447333333333"))

    def test_missing_or_expired_context_returns_none(self) -> None:
        self.assertIsNone(self.store.get("447999999999"))
        self.assertFalse(self.store.exists("447999999999"))


class BuildContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_redis = _FakeRedis()
        patcher = patch("app.conversation.context_store.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ConversationContextStore()

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_context_with_no_active_workflow(self, mock_get_customer, mock_get_workflow) -> None:
        mock_get_customer.return_value = {"id": 1, "full_name": "John Smith"}
        mock_get_workflow.return_value = None

        context = build_context("447000000001")

        self.assertTrue(context.is_registered)
        self.assertIsNone(context.current_workflow)
        self.assertIsNone(context.current_step)
        self.assertEqual(context.workflow_data, {})

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_context_with_active_onboarding_workflow_strips_sensitive_fields(
        self, mock_get_customer, mock_get_workflow
    ) -> None:
        mock_get_customer.return_value = None
        mock_get_workflow.return_value = {
            "type": "onboarding",
            "step": "CONFIRM_REGISTRATION",
            "workflow_id": "wf-1",
            "data": {
                "full_name": "John Smith",
                "aadhaar_number": "123456789012",
                "pan_number": "ABCDE1234F",
            },
        }

        context = build_context("447000000002")

        self.assertEqual(context.current_workflow, "onboarding")
        self.assertEqual(context.current_step, "CONFIRM_REGISTRATION")
        self.assertEqual(context.workflow_data.get("full_name"), "John Smith")
        self.assertNotIn("aadhaar_number", context.workflow_data)
        self.assertNotIn("pan_number", context.workflow_data)

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_context_with_active_transfer_workflow_strips_account_numbers(
        self, mock_get_customer, mock_get_workflow
    ) -> None:
        mock_get_customer.return_value = {"id": 2, "full_name": "Sarah Johnson"}
        mock_get_workflow.return_value = {
            "type": "transfer",
            "step": "CONFIRM_TRANSFER",
            "workflow_id": "wf-2",
            "data": {
                "beneficiary_name": "Priya Sharma",
                "beneficiary_account": "GB12FNCL00019999999",
                "source_account": "GB12FNCL00010007654321",
                "amount": "£50.00",
            },
        }

        context = build_context("447000000003")

        self.assertEqual(context.current_workflow, "transfer")
        self.assertEqual(context.workflow_data.get("beneficiary_name"), "Priya Sharma")
        self.assertEqual(context.workflow_data.get("amount"), "£50.00")
        self.assertNotIn("beneficiary_account", context.workflow_data)
        self.assertNotIn("source_account", context.workflow_data)

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_context_with_active_cheque_workflow(self, mock_get_customer, mock_get_workflow) -> None:
        mock_get_customer.return_value = {"id": 1, "full_name": "John Smith"}
        mock_get_workflow.return_value = {
            "type": "cheque",
            "step": "UPLOAD_CHEQUE",
            "workflow_id": "wf-3",
            "data": {},
        }

        context = build_context("447000000004")

        self.assertEqual(context.current_workflow, "cheque")
        self.assertEqual(context.current_step, "UPLOAD_CHEQUE")

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_context_with_active_loan_workflow_strips_account_number(
        self, mock_get_customer, mock_get_workflow
    ) -> None:
        mock_get_customer.return_value = {"id": 1, "full_name": "John Smith"}
        mock_get_workflow.return_value = {
            "type": "loan",
            "step": "UPLOAD_LOAN_FORM",
            "workflow_id": "wf-4",
            "data": {
                "account_number": "GB12FNCL00010001234567",
                "monthly_income": "3000",
                "loan_type": "personal",
            },
        }

        context = build_context("447000000005")

        self.assertEqual(context.current_workflow, "loan")
        self.assertEqual(context.workflow_data.get("monthly_income"), "3000")
        self.assertNotIn("account_number", context.workflow_data)

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_context_with_active_kyc_workflow_strips_sensitive_fields(
        self, mock_get_customer, mock_get_workflow
    ) -> None:
        mock_get_customer.return_value = {"id": 1, "full_name": "John Smith"}
        mock_get_workflow.return_value = {
            "type": "kyc",
            "step": "UPLOAD_KYC_FORM",
            "workflow_id": "wf-5",
            "data": {
                "full_name": "John Smith",
                "aadhaar_number": "123456789012",
                "pan_number": "ABCDE1234F",
                "address": "1 Test Street",
            },
        }

        context = build_context("447000000006")

        self.assertEqual(context.current_workflow, "kyc")
        self.assertEqual(context.workflow_data.get("address"), "1 Test Street")
        self.assertNotIn("aadhaar_number", context.workflow_data)
        self.assertNotIn("pan_number", context.workflow_data)

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_customer_lookup_failure_does_not_mean_unregistered(
        self, mock_get_customer, mock_get_workflow
    ) -> None:
        mock_get_workflow.return_value = None

        # First turn: customer is confirmed registered and persisted.
        mock_get_customer.return_value = {"id": 7, "full_name": "John Smith"}
        first = build_context("447000000007")
        self.assertTrue(first.is_registered)
        self.store.save(first)

        # Second turn: the DB lookup blows up. Registration must NOT flip
        # to False just because the lookup failed this time.
        mock_get_customer.side_effect = Exception("connection reset")
        second = build_context("447000000007")

        self.assertTrue(second.is_registered)

    @patch("app.conversation.builder.get_workflow")
    @patch("app.conversation.builder.get_customer_by_phone")
    def test_customer_lookup_failure_with_no_prior_context_records_error(
        self, mock_get_customer, mock_get_workflow
    ) -> None:
        mock_get_workflow.return_value = None
        mock_get_customer.side_effect = Exception("connection reset")

        context = build_context("447000000008")

        # No prior knowledge to fall back on — defaults to an unregistered
        # state, but the failure (not a confirmed "unregistered" fact) is
        # recorded so callers can tell the difference.
        self.assertFalse(context.is_registered)
        self.assertEqual(context.last_error, "customer_lookup_failed")


if __name__ == "__main__":
    unittest.main()

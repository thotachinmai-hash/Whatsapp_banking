"""Tests for Task 9.1 — Banking Guidance Policy layer.

Each test classifies real text through the existing (unmodified) intent
classifier and feeds the result into build_guidance(), so these are true
integration tests of "already-classified IntentResult -> GuidanceResult",
not hand-constructed IntentResult fixtures — matching how a future caller
would actually use this layer. Follows the project's unittest convention
(no pytest installed in this environment).
"""

import unittest

from app.conversation.intent.classifier import classify_intent
from app.conversation.guidance.models import GuidanceType, ResponseMode
from app.conversation.guidance.policy import build_guidance


async def _guidance(text):
    intent_result = await classify_intent(text)
    return intent_result, build_guidance(text, intent_result)


class GuidancePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_01_loan_eligibility_guidance(self):
        intent_result, guidance = await _guidance("I earn 50000 a month and want a personal loan")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.LOAN_ELIGIBILITY_GUIDANCE)
        self.assertEqual(guidance.response_mode, ResponseMode.OFFER_ACTIONS)
        self.assertEqual(guidance.entities.get("monthly_income"), 50000)
        self.assertEqual(guidance.entities.get("loan_type"), "personal")
        action_values = {a.value for a in guidance.suggested_actions}
        self.assertIn("check_eligibility", action_values)
        self.assertIn("start_application", action_values)
        self.assertIn("required_documents", action_values)

    async def test_02_loan_application_request_remains_an_action(self):
        intent_result, guidance = await _guidance("I want to apply for a personal loan")

        self.assertEqual(intent_result.intent, "loan_application_request")
        self.assertIsNone(guidance, "a direct loan application request must not become guidance")

    async def test_03_kyc_guidance(self):
        intent_result, guidance = await _guidance("What documents do I need for KYC?")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.KYC_GUIDANCE)
        self.assertEqual(guidance.response_mode, ResponseMode.EXPLAIN)

    async def test_04_transfer_guidance(self):
        intent_result, guidance = await _guidance("I don't know how to transfer money")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.TRANSFER_GUIDANCE)
        self.assertEqual(guidance.response_mode, ResponseMode.OFFER_ACTIONS)

    async def test_05_cheque_guidance(self):
        intent_result, guidance = await _guidance("Why is my cheque rejected?")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.CHEQUE_GUIDANCE)
        self.assertEqual(guidance.response_mode, ResponseMode.EXPLAIN)

    async def test_06_cheque_status_guidance(self):
        intent_result, guidance = await _guidance("My cheque is still pending")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.CHEQUE_STATUS_GUIDANCE)
        self.assertEqual(guidance.response_mode, ResponseMode.REDIRECT_TO_WORKFLOW)

    async def test_07_account_guidance(self):
        intent_result, guidance = await _guidance("What is the difference between savings and current account?")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.ACCOUNT_GUIDANCE)
        self.assertEqual(guidance.response_mode, ResponseMode.EXPLAIN)

    async def test_08_transaction_guidance(self):
        intent_result, guidance = await _guidance("How much did I spend on groceries?")

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance.guidance_type, GuidanceType.TRANSACTION_GUIDANCE)
        self.assertEqual(guidance.entities.get("category"), "groceries")

    async def test_09_direct_balance_request_remains_unchanged(self):
        intent_result, guidance = await _guidance("Show my balance")

        self.assertEqual(intent_result.intent, "balance_request")
        self.assertIsNone(guidance, "balance is a status lookup, not guidance")

    async def test_10_direct_transfer_request_remains_unchanged(self):
        intent_result, guidance = await _guidance("Transfer 500 GBP to Priya")

        self.assertEqual(intent_result.intent, "transfer_request")
        self.assertIsNone(guidance, "a concrete transfer request must continue to the transfer workflow")

    async def test_11_direct_loan_application_remains_unchanged(self):
        intent_result, guidance = await _guidance("I want a personal loan")

        self.assertEqual(intent_result.intent, "loan_application_request")
        self.assertIsNone(guidance)

    async def test_12_unknown_input(self):
        intent_result, guidance = await _guidance("qwerty asdf zzz")

        # Pure gibberish carries no banking keyword, so the classifier
        # itself resolves it to out_of_scope — guidance must stay silent,
        # matching the existing "out_of_scope never reaches banking logic" rule.
        self.assertIn(intent_result.intent, ("out_of_scope", "unknown"))
        if intent_result.intent == "out_of_scope":
            self.assertIsNone(guidance)
        else:
            self.assertEqual(guidance.guidance_type, GuidanceType.UNKNOWN)
            self.assertEqual(guidance.response_mode, ResponseMode.ASK_CLARIFICATION)

    async def test_13_no_banking_policy_hallucination(self):
        forbidden_keys = {
            "interest_rate", "approved", "eligible", "eligibility_result",
            "fee", "fees", "credit_limit", "loan_limit", "approval_status",
        }
        cases = [
            "I earn 50000 a month and want a personal loan",
            "Can I afford a loan?",
            "What documents do I need for KYC?",
            "I don't know how to transfer money",
            "Why is my cheque rejected?",
            "What is the difference between savings and current account?",
        ]
        for text in cases:
            _, guidance = await _guidance(text)
            if guidance is None:
                continue
            # The model schema itself has no field for a banking fact/decision —
            # only presentation-intent fields exist at all.
            self.assertEqual(
                set(type(guidance).model_fields.keys()),
                {"guidance_type", "response_mode", "suggested_actions", "entities", "confidence"},
            )
            leaked = forbidden_keys & set(guidance.entities.keys())
            self.assertFalse(leaked, f"hallucinated banking-policy keys {leaked} for {text!r}")

    async def test_14_entities_only_extracted_when_explicitly_present(self):
        # No income mentioned at all -> no monthly_income key, not a guessed default.
        _, guidance_no_income = await _guidance("Can I get a loan?")
        self.assertIsNotNone(guidance_no_income)
        self.assertNotIn("monthly_income", guidance_no_income.entities)

        # Income explicitly present -> extracted verbatim, nothing else invented.
        _, guidance_with_income = await _guidance("My income is 3000 and I want a loan")
        self.assertEqual(guidance_with_income.entities.get("monthly_income"), 3000)


class GuidancePolicyBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Structural checks that this layer cannot reach into banking execution."""

    def test_module_has_no_database_or_tool_imports(self):
        import ast
        import inspect

        from app.conversation.guidance import policy as policy_module

        source = inspect.getsource(policy_module)
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        forbidden_prefixes = ("app.database", "app.workflows", "app.agent.tools", "psycopg2", "redis")
        for module_name in imported_modules:
            for forbidden in forbidden_prefixes:
                self.assertFalse(
                    module_name == forbidden or module_name.startswith(forbidden + "."),
                    f"guidance policy must not import {module_name}",
                )

    def test_guidance_result_has_no_execution_capability(self):
        from app.conversation.guidance.models import GuidanceResult

        result = GuidanceResult()
        for forbidden_attr in ("execute", "commit", "run", "call", "approve", "create_workflow"):
            self.assertFalse(hasattr(result, forbidden_attr))

    async def test_out_of_scope_never_becomes_guidance(self):
        # Step 7 of the LLM-first routing migration removed
        # classify_out_of_scope()'s keyword-absence guess, so a message
        # like "why is the sky blue" is now "unknown" rather than
        # "out_of_scope" (see tests/test_conversation_router.py's
        # test_01_sky_is_blue_reaches_clarification_then_the_agent for that
        # change). Use a deny-list phrase here instead, so this test keeps
        # protecting the invariant it's named for.
        intent_result, guidance = await _guidance("tell me a joke")
        self.assertEqual(intent_result.intent, "out_of_scope")
        self.assertIsNone(guidance)

    async def test_navigation_never_becomes_guidance(self):
        intent_result, guidance = await _guidance("cancel")
        self.assertEqual(intent_result.intent, "cancel")
        self.assertIsNone(guidance)


if __name__ == "__main__":
    unittest.main()

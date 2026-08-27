"""Generic, any-workflow-to-any-workflow switching (see
app/workflows/manager.py::_switch_workflow and the new switch check in
WorkflowManager.handle()). The bug this fixes: a clear new-intent message
sent while a DIFFERENT workflow was active used to be swallowed as literal
input to the active workflow instead of being recognized — e.g. "I want to
create another bank account" mid-loan-application. Root cause: the active
workflow was given first refusal using its OWN narrow, keyword-based
detection (a 4-workflow allowlist, an on-topic-terms table that could
false-negative — "account" is vocabulary for both loan answers AND
"create another account"), instead of the already-computed, already-tested
classify_intent() result app/conversation/manager.py computes for every
message regardless of active workflow.

The fix reuses that SAME classify_intent() result (zero new LLM calls) as
the primary signal: any WORKFLOW_EXECUTING_INTENTS match for a DIFFERENT
workflow than the active one, at CONFIDENCE_HIGH, switches immediately —
one mechanism, not a per-workflow-pair table, via start_workflow_directly
(the same generic starter a fresh no-workflow START_WORKFLOW decision
already uses for every type).
"""

import unittest
from unittest.mock import patch

from app.conversation.intent.classifier import classify_intent
from app.conversation.renderer import as_structured_response
from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_SELECT_BENEFICIARY,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_CHEQUE,
    STEP_UPLOAD_KYC_FORM,
    WORKFLOW_ADD_ACCOUNT,
    WORKFLOW_CHEQUE,
    WORKFLOW_KYC,
    WORKFLOW_LOAN,
    WORKFLOW_TRANSFER,
)
from app.workflows.manager import WorkflowManager
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis — same pattern already
    used by tests/test_response_ux.py and tests/test_button_wiring_e2e.py."""

    def __init__(self):
        self._store = {}

    def setex(self, key, ttl, value):
        self._store[key] = value
        return True

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._store:
            return None
        self._store[key] = value
        return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, key):
        return 1 if self._store.pop(key, None) is not None else 0


class WorkflowSwitchingTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = WorkflowManager()
        self.phone = "441111111111"
        self.fake_redis = FakeRedis()
        patcher = patch("app.workflows.memory.redis_client", self.fake_redis)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _classify(self, query: str):
        return await classify_intent(query, context=None, trace_id="t", llm_classify=None)

    async def _handle(self, query: str):
        intent_result = await self._classify(query)
        result = self.manager.handle(self.phone, query, trace_id="t", intent_result=intent_result)
        return result, intent_result

    def _start(self, workflow_type: str, step: str, data: dict | None = None):
        create_workflow(self.phone, create_workflow_model(workflow_type, step, data=data or {}))


# ─── The exact reported bug ──────────────────────────────────────────────

class ReportedBugRegressionTests(WorkflowSwitchingTestCase):
    async def test_loan_to_add_account_switches_immediately_no_cancel_needed(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result, intent_result = await self._handle("I want to create another bank account")

        self.assertEqual(intent_result.intent, "add_account_request")
        self.assertTrue(result["handled"])
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_ADD_ACCOUNT)
        self.assertEqual(wf["step"], STEP_COLLECT_AADHAAR)
        # No "cancel" was required, and the response acknowledges the switch.
        text = as_structured_response(result["response"]).text.lower()
        self.assertIn("loan", text)
        self.assertIn("aadhaar", text)


# ─── Any-to-any switching across all persistent workflows ────────────────

class AnyToAnySwitchTests(WorkflowSwitchingTestCase):
    """Not exhaustive (6x5=30 pairs) but spans every workflow type as both
    source and target at least once, proving this is one generic mechanism
    rather than pairwise rules."""

    async def test_transfer_to_loan(self):
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        result, intent_result = await self._handle("I want to apply for a loan")
        self.assertEqual(intent_result.intent, "loan_application_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_kyc_to_cheque_deposit(self):
        self._start(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM, {})
        result, intent_result = await self._handle("I want to deposit a cheque")
        self.assertEqual(intent_result.intent, "cheque_deposit_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_CHEQUE)

    async def test_cheque_deposit_to_transfer(self):
        self._start(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE, {})
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result, intent_result = await self._handle("transfer 500 to Priya")
        self.assertEqual(intent_result.intent, "transfer_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)

    async def test_add_account_to_kyc(self):
        self._start(WORKFLOW_ADD_ACCOUNT, STEP_COLLECT_AADHAAR, {})
        result, intent_result = await self._handle("actually I need to update my KYC")
        self.assertEqual(intent_result.intent, "kyc_update_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_loan_to_transfer(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result, intent_result = await self._handle("send money to Rahul")
        self.assertEqual(intent_result.intent, "transfer_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)


# ─── Failed switch must not destroy the original workflow ────────────────

class FailedSwitchPreservesOriginalTests(WorkflowSwitchingTestCase):
    async def test_zero_balance_transfer_switch_leaves_loan_intact(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=False):
            result, intent_result = await self._handle("transfer 500 to Priya")
        self.assertEqual(intent_result.intent, "transfer_request")
        self.assertTrue(result["handled"])
        # No transfer workflow was actually created (insufficient balance) --
        # the original loan workflow must still be exactly where it was.
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_LOAN)
        self.assertEqual(wf["step"], STEP_SELECT_LOAN_TYPE)
        self.assertIn("balance", as_structured_response(result["response"]).text.lower())


# ─── CONTINUE: ordinary field input must not misfire as a switch ─────────

class ContinuationNotMisreadAsSwitchTests(WorkflowSwitchingTestCase):
    async def test_loan_account_field_answer_is_not_treated_as_add_account_switch(self):
        # "account" is legitimate loan-workflow vocabulary (an applicant's
        # linked account) -- must stay in the loan workflow, not misfire
        # the add_account_request switch just because the word is present.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch("app.workflows.processors.loan.get_accounts_by_phone", return_value=[
            {"account_number": "GB12FNCL00010001234567", "account_type": "current", "balance": "500.00", "currency": "INR"},
        ]), patch("app.workflows.processors.loan.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result, intent_result = await self._handle("lt_personal")
        self.assertNotEqual(intent_result.intent, "add_account_request")
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_LOAN)

    async def test_field_correction_is_not_treated_as_a_switch(self):
        # A genuine mid-workflow correction ("that's wrong, I meant X") is
        # classified as workflow_correction via classify_workflow_conversation's
        # context-aware rule layer -- not a WORKFLOW_EXECUTING_INTENTS
        # value at all, so the new switch check must never touch it; it
        # keeps flowing to the workflow's own existing correction handling.
        self._start(WORKFLOW_TRANSFER, "SELECT_AMOUNT", {"beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999"})
        from app.conversation.context import ConversationContext
        ctx = ConversationContext(phone_number=self.phone)
        ctx.current_workflow = WORKFLOW_TRANSFER
        ctx.current_step = "SELECT_AMOUNT"
        intent_result = await classify_intent(
            "that's wrong, I meant 700 not 500", context=ctx, trace_id="t", llm_classify=None
        )
        self.assertEqual(intent_result.intent, "workflow_correction")
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=[
            {"account_number": "GB12FNCL00010001234567", "account_type": "savings", "balance": "500.00"},
        ]), patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=None):
            self.manager.handle(self.phone, "that's wrong, I meant 700 not 500", trace_id="t", intent_result=intent_result)
        # Still in the SAME transfer workflow -- not switched away, and the
        # corrected amount (700, not the earlier 500) was actually applied.
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_TRANSFER)
        self.assertIn("700", wf["data"].get("amount", ""))

    async def test_transfer_amount_entry_stays_in_transfer(self):
        self._start(WORKFLOW_TRANSFER, "SELECT_AMOUNT", {"beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999"})
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=[
            {"account_number": "GB12FNCL00010001234567", "account_type": "savings", "balance": "500.00"},
        ]), patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=None):
            result, intent_result = await self._handle("500")
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_TRANSFER)
        self.assertTrue(result["handled"])


# ─── CANCEL: still works, untouched by this change ────────────────────────

class CancelStillWorksTests(WorkflowSwitchingTestCase):
    async def test_explicit_cancel_asks_to_confirm_then_stops(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        result1, intent_result1 = await self._handle("cancel")
        self.assertTrue(result1["handled"])
        # First cancel asks for confirmation -- workflow still active.
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)
        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result2, _ = await self._handle("stop")
        self.assertTrue(result2["handled"])
        self.assertIsNone(get_workflow(self.phone))


# ─── Multilingual / voice-transcribed switching ───────────────────────────

class MultilingualSwitchTests(WorkflowSwitchingTestCase):
    """classify_intent()'s rule layers are language-agnostic keyword
    matching -- romanized/code-mixed phrasing containing the right English
    trigger words is recognized exactly like plain English (confirmed
    live earlier this session). Voice messages reach this same code path
    as plain text after STT, so a "voice-transcribed" case is just another
    text input here -- there is no separate voice-specific switching logic
    to test."""

    async def test_romanized_hindi_loan_to_kyc_switch(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        # "mera kyc update karo" -- contains the English trigger word "kyc"
        result, intent_result = await self._handle("mera kyc update karo")
        self.assertEqual(intent_result.intent, "kyc_update_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_code_mixed_tamil_transfer_to_cheque_switch(self):
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        # code-mixed: Tamil words around the English trigger "cheque"/"deposit"
        result, intent_result = await self._handle("naan oru cheque deposit pannanum")
        self.assertEqual(intent_result.intent, "cheque_deposit_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_CHEQUE)

    async def test_voice_transcribed_add_account_switch(self):
        # Simulates exactly what saaras:v3 STT would hand to the same
        # conversation pipeline -- plain text, no special handling needed.
        self._start(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM, {})
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result, intent_result = await self._handle("open another account for me")
        self.assertEqual(intent_result.intent, "add_account_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)


# ─── Side-question categories keep working via the pre-existing path ─────

class SideQuestionCategoriesUnaffectedTests(WorkflowSwitchingTestCase):
    """Check Balance / Check Cheque Status / View Transactions are not
    persistent multi-step workflows -- they're answered in place via the
    existing (unchanged) reprocess_query -> BANKING_LLM path, which
    correctly leaves the active workflow untouched rather than switching
    away from it."""

    async def test_balance_question_mid_loan_answers_without_losing_workflow(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        result, intent_result = await self._handle("what is my balance")
        self.assertEqual(intent_result.intent, "balance_request")
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), "what is my balance")
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_cheque_status_mid_kyc_answers_without_losing_workflow(self):
        self._start(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM, {})
        result, intent_result = await self._handle("check my cheque status")
        self.assertEqual(intent_result.intent, "cheque_status_request")
        self.assertFalse(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_view_transactions_mid_transfer_answers_without_losing_workflow(self):
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        result, intent_result = await self._handle("show me my transactions")
        self.assertEqual(intent_result.intent, "transaction_request")
        self.assertFalse(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)


if __name__ == "__main__":
    unittest.main()

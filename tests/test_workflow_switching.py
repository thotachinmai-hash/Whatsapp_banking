"""Generic, any-workflow-to-any-workflow switching (see
app/workflows/manager.py::_switch_workflow and the switch/cancel/side-
question detection in WorkflowManager.handle()).

LLM-first migration note: the old rule-based classify_intent() could
recognize a subset of workflow-request phrasings "for free" (no LLM call),
so earlier versions of these tests relied on that to synthesize an
IntentResult automatically. That rule layer has been removed entirely —
intent understanding for anything beyond literal protocol/navigation words
is now the LLM router's job (app/conversation/intent/llm_routing.py). These
tests now unit-test WorkflowManager's DISPATCH logic by passing an explicit
LLMRoutingDecision (what the router decided) rather than re-deriving it
from text — that decouples "does the manager act correctly on a given
decision" (tested here, fast, no API key needed) from "does the LLM
actually produce the right decision for this phrasing" (validated
separately, against the real Sarvam API, by
scripts/real_sarvam_validation.py).
"""

import unittest
from unittest.mock import patch

from app.conversation.intent.classifier import classify_intent
from app.conversation.intent.llm_routing import LLMRoutingDecision
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
from app.workflows.manager import WorkflowManager, _is_cancel_command
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow


def _decision(intent: str, action: str, target_workflow=None, certainty: str = "high") -> LLMRoutingDecision:
    return LLMRoutingDecision(intent=intent, action=action, certainty=certainty, target_workflow=target_workflow)


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis."""

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
        """The deterministic pre-filter only — used to confirm a premise
        (e.g. "this phrasing is NOT resolved for free") in a few tests."""
        return await classify_intent(query, context=None, trace_id="t")

    def _handle(self, query: str, decision: LLMRoutingDecision | None = None):
        """Dispatch through WorkflowManager, injecting `decision` as the
        already-computed LLM routing decision for this turn (the same
        contract app/conversation/manager.py uses) — never re-derived from
        text inside this helper."""
        return self.manager.handle(self.phone, query, trace_id="t", llm_decision=decision)

    def _start(self, workflow_type: str, step: str, data: dict | None = None):
        create_workflow(self.phone, create_workflow_model(workflow_type, step, data=data or {}))


# ─── The exact reported bug ──────────────────────────────────────────────

class ReportedBugRegressionTests(WorkflowSwitchingTestCase):
    async def test_loan_to_add_account_switches_immediately_no_cancel_needed(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        decision = _decision("add_account_request", "SWITCH", target_workflow="add_account")
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result = self._handle("I want to create another bank account", decision)

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
        decision = _decision("loan_application_request", "SWITCH", target_workflow="loan")
        result = self._handle("I want to apply for a loan", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_kyc_to_cheque_deposit(self):
        self._start(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM, {})
        decision = _decision("cheque_deposit_request", "SWITCH", target_workflow="cheque")
        result = self._handle("I want to deposit a cheque", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_CHEQUE)

    async def test_cheque_deposit_to_transfer(self):
        self._start(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE, {})
        decision = _decision("transfer_request", "SWITCH", target_workflow="transfer")
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result = self._handle("transfer 500 to Priya", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)

    async def test_add_account_to_kyc(self):
        self._start(WORKFLOW_ADD_ACCOUNT, STEP_COLLECT_AADHAAR, {})
        decision = _decision("kyc_update_request", "SWITCH", target_workflow="kyc")
        result = self._handle("actually I need to update my KYC", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_loan_to_transfer(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        decision = _decision("transfer_request", "SWITCH", target_workflow="transfer")
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result = self._handle("send money to Rahul", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)


# ─── Failed switch must not destroy the original workflow ────────────────

class FailedSwitchPreservesOriginalTests(WorkflowSwitchingTestCase):
    async def test_zero_balance_transfer_switch_leaves_loan_intact(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        decision = _decision("transfer_request", "SWITCH", target_workflow="transfer")
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=False):
            result = self._handle("transfer 500 to Priya", decision)
        self.assertTrue(result["handled"])
        # No transfer workflow was actually created (insufficient balance) --
        # the original loan workflow must still be exactly where it was.
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_LOAN)
        self.assertEqual(wf["step"], STEP_SELECT_LOAN_TYPE)
        self.assertIn("balance", as_structured_response(result["response"]).text.lower())


# ─── CONTINUE/CORRECT: ordinary field input must not misfire as a switch ─

class ContinuationNotMisreadAsSwitchTests(WorkflowSwitchingTestCase):
    async def test_loan_account_field_answer_is_not_treated_as_add_account_switch(self):
        # "lt_personal" is a literal button-tap protocol id (has an
        # underscore, no space) -- always belongs to the active step's
        # own processor, never diverted, and never needs an LLM call.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm, \
             patch("app.workflows.processors.loan.get_accounts_by_phone", return_value=[
                {"account_number": "GB12FNCL00010001234567", "account_type": "current", "balance": "500.00", "currency": "INR"},
             ]), patch("app.workflows.processors.loan.get_customer_by_phone", return_value={"full_name": "Alex"}):
            self._handle("lt_personal")
        mock_llm.assert_not_called()
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_LOAN)

    async def test_field_correction_is_not_treated_as_a_switch(self):
        # A genuine mid-workflow correction ("that's wrong, I meant X") is
        # a CORRECT decision, not a workflow-executing SWITCH -- the
        # switch check must never touch it, and it keeps flowing to the
        # workflow's own existing correction handling.
        self._start(WORKFLOW_TRANSFER, "SELECT_AMOUNT", {"beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999"})
        decision = _decision("transfer_request", "CORRECT")
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=[
            {"account_number": "GB12FNCL00010001234567", "account_type": "savings", "balance": "500.00"},
        ]), patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=None):
            self._handle("that's wrong, I meant 700 not 500", decision)
        # Still in the SAME transfer workflow -- not switched away, and the
        # corrected amount (700, not the earlier 500) was actually applied.
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_TRANSFER)
        self.assertIn("700", wf["data"].get("amount", ""))

    async def test_transfer_amount_entry_stays_in_transfer(self):
        # A bare number is a protocol/field-shape input -- resolved
        # without any LLM call.
        self._start(WORKFLOW_TRANSFER, "SELECT_AMOUNT", {"beneficiary_name": "Priya", "beneficiary_account": "GB12FNCL00010009999999"})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm, \
             patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=[
                {"account_number": "GB12FNCL00010001234567", "account_type": "savings", "balance": "500.00"},
             ]), patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=None):
            result = self._handle("500")
        mock_llm.assert_not_called()
        wf = get_workflow(self.phone)
        self.assertEqual(wf["type"], WORKFLOW_TRANSFER)
        self.assertTrue(result["handled"])


# ─── CANCEL: still works, both the literal and LLM-recognized paths ──────

class CancelStillWorksTests(WorkflowSwitchingTestCase):
    async def test_explicit_cancel_asks_to_confirm_then_stops(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            result1 = self._handle("cancel")
        mock_llm.assert_not_called()  # a literal cancel word never needs the LLM
        self.assertTrue(result1["handled"])
        # First cancel asks for confirmation -- workflow still active.
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)
        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result2 = self._handle("stop")
        self.assertTrue(result2["handled"])
        self.assertIsNone(get_workflow(self.phone))

    async def test_never_mind_dont_open_the_account_cancels(self):
        # A longer, natural-language cancellation ("never mind, don't open
        # the account") is not a literal exact-phrase match -- the LLM
        # router's CANCEL action recognizes it instead (no keyword
        # substring guess is made here anymore).
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        self.assertFalse(_is_cancel_command("never mind, don't open the account"))
        decision = _decision("unknown", "CANCEL")
        result1 = self._handle("never mind, don't open the account", decision)
        self.assertTrue(result1["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)  # confirmation step first
        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result2 = self._handle("stop")
        self.assertTrue(result2["handled"])
        self.assertIsNone(get_workflow(self.phone))


# ─── Multilingual / voice-transcribed / code-mixed switching ─────────────

class MultilingualSwitchTests(WorkflowSwitchingTestCase):
    """The LLM router understands romanized/code-mixed/native-script
    phrasing directly (validated separately against the real Sarvam API —
    see scripts/real_sarvam_validation.py); these tests only confirm
    WorkflowManager acts correctly once the router has decided, using
    representative multilingual trigger text with an explicit decision. A
    voice-transcribed message reaches this same code path as plain text
    after STT, so there is no separate voice-specific switching logic."""

    async def test_romanized_hindi_loan_to_kyc_switch(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        decision = _decision("kyc_update_request", "SWITCH", target_workflow="kyc")
        result = self._handle("mera kyc update karo", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_code_mixed_tamil_transfer_to_cheque_switch(self):
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        decision = _decision("cheque_deposit_request", "SWITCH", target_workflow="cheque")
        result = self._handle("naan oru cheque deposit pannanum", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_CHEQUE)

    async def test_voice_transcribed_add_account_switch(self):
        self._start(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM, {})
        decision = _decision("add_account_request", "SWITCH", target_workflow="add_account")
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result = self._handle("open another account for me", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)

    CODE_MIXED_SWITCH = "loan application ni pause chesi, naaku ఒక కొత్త account కావాలి"

    async def test_code_mixed_pivot_switches(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        decision = _decision("add_account_request", "SWITCH", target_workflow="add_account")
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=decision) as mock_llm, \
             patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result = self._handle(self.CODE_MIXED_SWITCH)
        mock_llm.assert_called_once()
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)

    async def test_pure_native_script_switches_via_llm(self):
        # Devanagari "loan" ("लोन") -- a case no literal keyword/regex
        # rule could ever read -- resolved entirely by the LLM decision.
        self._start(WORKFLOW_KYC, STEP_COLLECT_AADHAAR, {})
        message = "never mind, मुझे लोन चाहिए"  # never mind, I want a loan
        decision = _decision("loan_application_request", "SWITCH", target_workflow="loan")
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=decision):
            result = self._handle(message)
        self.assertTrue(result["handled"])
        text = as_structured_response(result["response"]).text.lower()
        self.assertIn("pausing", text)
        self.assertIn("loan", text)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)


# ─── Context-aware cancellation vs switching ──────────────────────────────

class CancelVsSwitchContextAwarenessTests(WorkflowSwitchingTestCase):
    """"actually never mind, apply for a loan for me" mid-add_account must
    be treated as a SWITCH, not a cancellation, because the LLM router
    reads the whole sentence (not just a "never mind" prefix) and returns
    action=SWITCH with high certainty -- WorkflowManager always prefers a
    confident switch over a cancel signal in the same decision."""

    async def test_explicit_cancellation_still_cancels(self):
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result = self._handle("cancel this")
        self.assertTrue(result["handled"])
        self.assertIn("continue", as_structured_response(result["response"]).text.lower())
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)  # confirmation first

    async def test_never_mind_naming_nothing_new_still_cancels(self):
        # "never mind" alone (no trailing new request) is caught by the
        # literal deterministic check -- no LLM call needed.
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            result = self._handle("never mind")
        mock_llm.assert_not_called()
        self.assertTrue(result["handled"])
        self.assertIn("continue", as_structured_response(result["response"]).text.lower())
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)

    async def test_never_mind_apply_for_a_loan_switches_not_cancels(self):
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        decision = _decision("loan_application_request", "SWITCH", target_workflow="loan")
        result = self._handle("actually never mind, apply for a loan for me", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)
        text = as_structured_response(result["response"]).text.lower()
        self.assertNotIn("continue, or stop", text)  # not the cancel-confirmation prompt
        self.assertIn("loan", text)

    async def test_never_mind_create_another_account_switches_from_loan(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        decision = _decision("add_account_request", "SWITCH", target_workflow="add_account")
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result = self._handle("never mind, I want to create another account", decision)
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)


# ─── Side questions mid-workflow ───────────────────────────────────────────

class MultilingualSideQuestionTests(WorkflowSwitchingTestCase):
    """A mid-workflow side question (balance/transactions/status/RAG),
    in any language or script, is answered by the real LLM+tools agent
    without losing the active flow — the router's TOOL/RAG action drives
    this, not an English question-marker heuristic."""

    def _side_question(self, intent="balance_request", action="TOOL"):
        return _decision(intent, action, certainty="high")

    async def test_side_question_preserves_workflow(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        message = "Naa bank khatalo entha dabbu undi"  # romanized Telugu
        result = self._handle(message, self._side_question())
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)  # workflow preserved

    async def test_native_tamil_side_question_preserves_workflow(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        message = "ஆ இல்லை என்னோட வங்கியில் பணம் போக்குவரத்து எப்படி"
        result = self._handle(message, self._side_question(intent="transfer_question", action="RAG"))
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_code_mixed_side_question_preserves_workflow(self):
        self._start(WORKFLOW_KYC, STEP_COLLECT_AADHAAR, {})
        message = "yaar naa account lo entha undi ippudu"
        result = self._handle(message, self._side_question())
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_voice_transcribed_side_question_preserves_workflow(self):
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        message = "um yeah so naa bank lo entha dabbu undi ippudu cheppu"
        result = self._handle(message, self._side_question())
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)

    async def test_low_certainty_tool_does_not_divert_a_real_field_answer(self):
        # An ordinary amount-shaped answer sometimes gets classified TOOL
        # at low certainty by the model itself -- diverting that would
        # silently swallow a real field answer instead of letting the
        # step processor parse it.
        self._start(WORKFLOW_TRANSFER, "COLLECT_AMOUNT", {"beneficiary_name": "Priya"})
        decision = _decision("balance_request", "TOOL", certainty="low")
        with patch("app.workflows.processors.transfer.get_accounts_by_phone", return_value=[
            {"account_number": "GB12FNCL00010001234567", "account_type": "savings", "balance": "500.00"},
        ]), patch("app.workflows.processors.transfer.get_frequently_used_account", return_value=None):
            result = self._handle("500", decision)
        # "500" is a bare digit -- resolved as a protocol/field input
        # before the LLM decision is even consulted.
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)

    async def test_explicit_field_value_answer_never_triggers_the_llm_check(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            result = self._handle("income: 60000")
        mock_llm.assert_not_called()
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_bare_field_answer_still_resolves_correctly_as_continue(self):
        # A bare word like "personal" isn't a recognized protocol shape,
        # so it DOES get checked -- but the LLM recognizes it as CONTINUE
        # (not a side question), so the loan processor still gets its
        # normal chance to handle it.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        decision = _decision("loan_application_request", "CONTINUE")
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=decision) as mock_llm, \
             patch("app.workflows.processors.loan.get_customer_by_phone", return_value={"full_name": "Alex"}), \
             patch("app.workflows.processors.loan.get_accounts_by_phone", return_value=[]):
            result = self._handle("personal")
        mock_llm.assert_called_once()
        self.assertIsNone(result.get("reprocess_query"))
        self.assertTrue(result["handled"])


# ─── Side-question categories keep working via the pre-existing path ─────

class SideQuestionCategoriesUnaffectedTests(WorkflowSwitchingTestCase):
    """Check Balance / Check Cheque Status / View Transactions are not
    persistent multi-step workflows -- they're answered in place via the
    existing reprocess_query -> BANKING_LLM path, leaving the active
    workflow untouched rather than switching away from it."""

    async def test_balance_question_mid_loan_answers_without_losing_workflow(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        decision = _decision("balance_request", "TOOL")
        result = self._handle("what is my balance", decision)
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), "what is my balance")
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_cheque_status_mid_kyc_answers_without_losing_workflow(self):
        self._start(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM, {})
        decision = _decision("cheque_status_request", "TOOL")
        result = self._handle("check my cheque status", decision)
        self.assertFalse(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_view_transactions_mid_transfer_answers_without_losing_workflow(self):
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        decision = _decision("transaction_request", "TOOL")
        result = self._handle("show me my transactions", decision)
        self.assertFalse(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)


if __name__ == "__main__":
    unittest.main()

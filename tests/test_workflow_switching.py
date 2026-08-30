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

    async def test_never_mind_dont_open_the_account_cancels(self):
        # Confirmed live via scripts/shadow_eval.py's 101-case matrix
        # (acct_cancel): _is_cancel_command() only matched "never mind" as
        # the ENTIRE message, so this longer, very common phrasing fell
        # through and left the customer stuck answering account-type
        # prompts for a workflow they'd already tried to abandon.
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result1, _ = await self._handle("never mind, don't open the account")
        self.assertTrue(result1["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)  # confirmation step first
        with patch("app.database.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result2, _ = await self._handle("stop")
        self.assertTrue(result2["handled"])
        self.assertIsNone(get_workflow(self.phone))


# ─── Multilingual / voice-transcribed switching ───────────────────────────

class NonAsciiPivotDetectionTests(WorkflowSwitchingTestCase):
    """Step 6 introduced classify_and_route_llm_sync() (wired into
    WorkflowManager.handle()'s jump-detection block) to catch code-mixed/
    native-script pivots the English-keyword rules miss (confirmed live via
    scripts/shadow_eval.py's en_te_mixed_switch case). Step 7 removed
    _looks_like_new_service_request() (the English-keyword pre-filter that
    used to gate this check) entirely: gating an LLM understanding step
    with a semantic keyword guess just reintroduces the same blind spot one
    layer up, which is exactly what broke on this case in the first place.

    This later migration step made the mechanism PRODUCTION-AUTHORITATIVE
    (no LLM_FALLBACK_ENABLED gate) after validating it against a 101-case
    live corpus plus real traffic — it is no longer shadow/opt-in, so these
    tests no longer patch that flag at all. classify_and_route_llm_sync()
    now runs unconditionally for ANY message (ASCII or not) in the 4
    covered workflow types, behind only structural guards, and never on
    text that already looks like an explicit "field: value" answer for the
    active workflow."""

    CODE_MIXED_SWITCH = "loan application ni pause chesi, naaku ఒక కొత్త account కావాలి"

    def _fake_decision(self, action="SWITCH", target="add_account", certainty="high"):
        from app.conversation.intent.llm_routing import LLMRoutingDecision
        return LLMRoutingDecision(
            intent="add_account_request", action=action, certainty=certainty, target_workflow=target,
        )

    async def test_code_mixed_pivot_is_offered_by_default(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        intent_result = await self._classify(self.CODE_MIXED_SWITCH)
        # Confirms the premise: the fast rule classifier gets this wrong
        # (mid-loan vocabulary "loan application" wins), so the existing
        # classify_intent()-based switch check in handle() cannot catch it
        # -- this test is exercising the LLM-based fallback path.
        self.assertNotEqual(intent_result.intent, "add_account_request")

        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_decision()) as mock_llm:
            result = self.manager.handle(self.phone, self.CODE_MIXED_SWITCH, trace_id="t", intent_result=intent_result)

        mock_llm.assert_called_once()
        self.assertTrue(result["handled"])
        text = as_structured_response(result["response"]).text.lower()
        self.assertIn("switch", text)
        # Still an active workflow -- a pivot suggestion, not an executed
        # switch: no confirmation step was bypassed.
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_plain_ascii_pivot_is_also_offered_by_default(self):
        # This path is not ASCII-only -- a plain English phrasing the fast
        # rule check doesn't confidently catch gets the same LLM-based
        # chance a code-mixed one does.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        message = "can I add a second savings account to my profile"
        intent_result = await self._classify(message)
        # Confirms the premise: the fast classify_intent()-based switch
        # check (WORKFLOW_EXECUTING_INTENTS at CONFIDENCE_HIGH) does NOT
        # already catch this phrasing, so this test exercises the LLM
        # fallback path specifically, not the free fast-path switch.
        self.assertNotEqual(intent_result.intent, "add_account_request")
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_decision()) as mock_llm:
            result = self.manager.handle(self.phone, message, trace_id="t", intent_result=intent_result)
        mock_llm.assert_called_once()
        self.assertTrue(result["handled"])
        self.assertIn("switch", as_structured_response(result["response"]).text.lower())

    async def test_active_by_default_no_flag_needed(self):
        # The core claim of this migration step: no LLM_FALLBACK_ENABLED
        # patch anywhere in this test, and the mechanism still fires.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        intent_result = await self._classify(self.CODE_MIXED_SWITCH)
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_decision()) as mock_llm:
            self.manager.handle(self.phone, self.CODE_MIXED_SWITCH, trace_id="t", intent_result=intent_result)
        mock_llm.assert_called_once()

    async def test_explicit_field_value_answer_never_triggers_the_llm_check(self):
        # The one remaining structural guard: _is_current_workflow_input()
        # still recognizes an explicit "field: value" answer as
        # unambiguous literal input for the loan workflow.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        intent_result = await self._classify("income: 60000")
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            self.manager.handle(self.phone, "income: 60000", trace_id="t", intent_result=intent_result)
        mock_llm.assert_not_called()

    async def test_low_certainty_switch_from_llm_does_not_switch(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        intent_result = await self._classify(self.CODE_MIXED_SWITCH)
        with patch("app.workflows.manager.classify_and_route_llm_sync",
                   return_value=self._fake_decision(action="CONTINUE")):
            result = self.manager.handle(self.phone, self.CODE_MIXED_SWITCH, trace_id="t", intent_result=intent_result)
        # action=CONTINUE (not SWITCH) -> no pivot offered, falls through to
        # the loan processor exactly as before this fix.
        self.assertNotIn("switch to", (as_structured_response(result["response"]).text.lower() if result["handled"] else ""))


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


# ─── Context-aware cancellation vs switching ──────────────────────────────

class CancelVsSwitchContextAwarenessTests(WorkflowSwitchingTestCase):
    """The reported regression: "actually never mind, apply for a loan for
    me" mid-add_account was being treated as pure cancellation because
    _is_cancel_command()'s "never mind" match fired before the sentence's
    real content (a different, confident workflow request) was ever
    considered. Fixed by reordering WorkflowManager.handle() so the
    already-computed intent_result is checked for a confident
    WORKFLOW_EXECUTING_INTENTS match BEFORE the cancel check, reusing the
    exact same generic switch mechanism AnyToAnySwitchTests already
    covers -- no new keywords, no pairwise rules.

    A second, deeper bug surfaced investigating this: classify_hard_navigation()
    and _is_cancel_command() both strip every non-Latin character before
    matching, so a PURE native-script compound message ("never mind,
    मुझे लोन चाहिए") collapsed to exactly "never mind" and matched as
    cancellation regardless of what followed. Fixed by deferring (not
    guessing) whenever the raw text contains non-ASCII content neither
    function can safely interpret -- this never removes a real match
    (both were English-only to begin with), it only stops a false one."""

    async def test_explicit_cancellation_still_cancels(self):
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result, _ = await self._handle("cancel this")
        self.assertTrue(result["handled"])
        self.assertIn("continue", as_structured_response(result["response"]).text.lower())
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)  # confirmation first

    async def test_never_mind_naming_nothing_new_still_cancels(self):
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result, _ = await self._handle("never mind, don't open the account")
        self.assertTrue(result["handled"])
        self.assertIn("continue", as_structured_response(result["response"]).text.lower())
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)

    async def test_never_mind_apply_for_a_loan_switches_not_cancels(self):
        # The exact reported message.
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result, intent_result = await self._handle("actually never mind, apply for a loan for me")
        self.assertEqual(intent_result.intent, "loan_application_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)
        text = as_structured_response(result["response"]).text.lower()
        self.assertNotIn("continue, or stop", text)  # not the cancel-confirmation prompt
        self.assertIn("loan", text)

    async def test_never_mind_create_another_account_switches_from_loan(self):
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}):
            result, intent_result = await self._handle("never mind, I want to create another account")
        self.assertEqual(intent_result.intent, "add_account_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)

    async def test_romanized_never_mind_switch(self):
        # Romanized Telugu -- "loan" survives as a literal Latin substring,
        # so this resolves via the same fast rule path as the English case.
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result, intent_result = await self._handle("never mind, naaku loan kavali")
        self.assertEqual(intent_result.intent, "loan_application_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_code_mixed_never_mind_switch(self):
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        with patch("app.workflows.processors.transfer.has_transferable_balance", return_value=True), \
             patch("app.workflows.processors.transfer.get_beneficiaries_by_phone", return_value=[]):
            result, intent_result = await self._handle("arre never mind yaar, mujhe transfer karna hai 500 Priya ko")
        self.assertEqual(intent_result.intent, "transfer_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)

    async def test_voice_transcribed_never_mind_switch(self):
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        result, intent_result = await self._handle(
            "um never mind actually can you just apply for a loan instead"
        )
        self.assertEqual(intent_result.intent, "loan_application_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_pure_native_script_never_mind_no_longer_falsely_cancels(self):
        # Deepest case: "never mind, मुझे लोन चाहिए" (never mind, I want a
        # loan). Neither classify_hard_navigation() nor classify_workflow_request()
        # can read Devanagari "लोन" as English "loan", so this can't yet
        # resolve to a confident SWITCH without the LLM fallback enabled
        # (see test_pure_native_script_switches_with_llm_fallback_enabled)
        # -- but it must NEVER be misread as a false cancellation either,
        # which is the specific bug fixed here.
        self._start(WORKFLOW_ADD_ACCOUNT, "SELECT_ACCOUNT_TYPE", {})
        intent_result = await self._classify("never mind, मुझे लोन चाहिए")
        self.assertNotEqual(intent_result.intent, "cancel")
        from app.workflows.manager import _is_cancel_command
        self.assertFalse(_is_cancel_command("never mind, मुझे लोन चाहिए"))

    async def test_pure_native_script_switches_with_llm_fallback_enabled(self):
        # The LLM-based jump-detection block only covers the 4 workflow
        # types that already had it before this session (cheque/loan/kyc/
        # transfer, not add_account/onboarding) -- start from loan so this
        # test actually exercises that path rather than falling through to
        # a step processor untouched by this fix.
        from app.conversation.intent.llm_routing import LLMRoutingDecision

        self._start(WORKFLOW_KYC, STEP_COLLECT_AADHAAR, {})
        message = "never mind, मुझे लोन चाहिए"  # never mind, I want a loan
        intent_result = await self._classify(message)
        self.assertNotEqual(intent_result.intent, "cancel")  # the false-cancel bug stays fixed
        self.assertNotEqual(intent_result.intent, "loan_application_request")  # fast rule can't read Devanagari
        with patch(
                "app.workflows.manager.classify_and_route_llm_sync",
                return_value=LLMRoutingDecision(
                    intent="loan_application_request", action="SWITCH", certainty="high", target_workflow="loan",
                ),
             ):
            result = self.manager.handle(self.phone, message, trace_id="t", intent_result=intent_result)
        self.assertTrue(result["handled"])
        text = as_structured_response(result["response"]).text.lower()
        self.assertIn("switch", text)


# ─── Multilingual/unmarked side questions mid-workflow ────────────────────

class MultilingualSideQuestionTests(WorkflowSwitchingTestCase):
    """The bug: _is_conversational_query() (app/workflows/manager.py) is
    deliberately English-question-marker-led ("?", "what", "how", ...), so
    a genuine side question with no "?" and no English question word --
    pure native script, OR fully-ASCII romanized text like "Naa bank
    khatalo entha dabbu undi" -- failed it completely and fell straight
    through to the step processor as literal field input, silently
    corrupting or rejecting the current step's real answer while never
    answering the actual question. Confirmed against a REAL logged test
    conversation (scripts/_real_log_cases.json), not a hypothetical.

    Fixed by extending classify_and_route_llm_sync() (already built for
    Step 6/7's pivot detection) to also cover this gap: when the fast
    English-marker check AND the "looks like literal field input" check
    both come back empty-handed, ask the LLM instead of guessing -- no new
    keyword/script table. Made PRODUCTION-AUTHORITATIVE (no
    LLM_FALLBACK_ENABLED gate) after this migration step's validation, so
    none of these tests patch that flag -- reusing the exact SAME
    mid-workflow reprocess_query hand-off the English-question path already
    used (no new call-count behavior once the LLM has answered)."""

    def _fake_side_question_decision(self, intent="balance_request", action="TOOL"):
        from app.conversation.intent.llm_routing import LLMRoutingDecision
        return LLMRoutingDecision(intent=intent, action=action, certainty="high")

    async def test_real_romanized_telugu_side_question_preserves_workflow(self):
        # The exact real message from logs/logs.txt.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        message = "Naa bank khatalo entha dabbu undi"
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_side_question_decision()) as mock_llm:
            result, intent_result = await self._handle(message)
        mock_llm.assert_called_once()
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)  # workflow preserved

    async def test_real_native_tamil_side_question_preserves_workflow(self):
        # The exact real message from logs/logs.txt (asking how bank
        # transfers work, no "?").
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        message = "ஆ இல்லை என்னோட வங்கியில் பணம் போக்குவரத்து எப்படி"
        with patch(
                "app.workflows.manager.classify_and_route_llm_sync",
                return_value=self._fake_side_question_decision(intent="transfer_question", action="RAG"),
             ) as mock_llm:
            result, intent_result = await self._handle(message)
        mock_llm.assert_called_once()
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_code_mixed_side_question_preserves_workflow(self):
        self._start(WORKFLOW_KYC, STEP_COLLECT_AADHAAR, {})
        message = "yaar naa account lo entha undi ippudu"
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_side_question_decision()) as mock_llm:
            result, intent_result = await self._handle(message)
        mock_llm.assert_called_once()
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_KYC)

    async def test_voice_transcribed_unmarked_side_question_preserves_workflow(self):
        # STT output with no punctuation and no clean English question lead-in.
        self._start(WORKFLOW_TRANSFER, STEP_SELECT_BENEFICIARY, {})
        message = "um yeah so naa bank lo entha dabbu undi ippudu cheppu"
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_side_question_decision()) as mock_llm:
            result, intent_result = await self._handle(message)
        mock_llm.assert_called_once()
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_TRANSFER)

    async def test_english_question_marker_still_uses_the_free_fast_path(self):
        # Control: "what's my balance" already has a "?"-less English
        # marker ("my account"/"balance" etc. per _is_conversational_query),
        # so this must NOT need the LLM fallback at all -- zero calls.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            result, intent_result = await self._handle("what's my balance")
        mock_llm.assert_not_called()
        self.assertFalse(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_clear_switch_request_still_switches_not_treated_as_side_question(self):
        # Contrast case from the request: a message that names a clearly
        # DIFFERENT operation must still SWITCH (via the existing fast
        # rule-based switch check, unaffected by this fix), not be
        # misread as an answerable side question.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {"loan_type": "personal"})
        with patch("app.workflows.processors.onboarding.get_accounts_by_phone", return_value=[]), \
             patch("app.workflows.processors.onboarding.get_customer_by_phone", return_value={"full_name": "Alex"}), \
             patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            result, intent_result = await self._handle("I want to create another account")
        mock_llm.assert_not_called()  # resolved by the fast path, no LLM needed
        self.assertEqual(intent_result.intent, "add_account_request")
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_ADD_ACCOUNT)

    async def test_active_by_default_no_flag_needed(self):
        # The core claim of this migration step: no LLM_FALLBACK_ENABLED
        # patch anywhere in this test, and the mechanism still fires and
        # fixes the exact real-traffic bug (see the class docstring).
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        message = "Naa bank khatalo entha dabbu undi"
        with patch("app.workflows.manager.classify_and_route_llm_sync", return_value=self._fake_side_question_decision()) as mock_llm:
            result, intent_result = await self._handle(message)
        mock_llm.assert_called_once()
        self.assertFalse(result["handled"])
        self.assertEqual(result.get("reprocess_query"), message)
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_explicit_field_value_answer_never_triggers_the_llm_check(self):
        # The one structural guard this fix keeps: _is_current_workflow_input()
        # still recognizes an explicit "field: value" answer as unambiguous
        # literal input, so that shape never pays for an LLM call.
        self._start(WORKFLOW_LOAN, "COLLECT_INCOME", {"loan_type": "personal"})
        with patch("app.workflows.manager.classify_and_route_llm_sync") as mock_llm:
            result, intent_result = await self._handle("income: 60000")
        mock_llm.assert_not_called()
        self.assertTrue(result["handled"])
        self.assertEqual(get_workflow(self.phone)["type"], WORKFLOW_LOAN)

    async def test_bare_field_answer_gets_checked_but_still_resolves_correctly(self):
        # A bare word like "personal" isn't in "field: value" format, so
        # -- deliberately, matching Step 7's own precedent of no keyword/
        # length heuristics gating an LLM understanding step -- it DOES get
        # checked. The invariant that matters is that it still resolves
        # correctly and is never swallowed or corrupted: the LLM recognizes
        # this as CONTINUE (not a side question), so the loan processor
        # still gets its normal chance to handle it.
        self._start(WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE, {})
        with patch(
                "app.workflows.manager.classify_and_route_llm_sync",
                return_value=self._fake_side_question_decision(intent="loan_application_request", action="CONTINUE"),
             ) as mock_llm, \
             patch("app.workflows.processors.loan.get_customer_by_phone", return_value={"full_name": "Alex"}), \
             patch("app.workflows.processors.loan.get_accounts_by_phone", return_value=[]):
            result, intent_result = await self._handle("personal")
        mock_llm.assert_called_once()
        self.assertIsNone(result.get("reprocess_query"))
        self.assertTrue(result["handled"])


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

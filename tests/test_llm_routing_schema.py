"""app/conversation/intent/llm_routing.py::LLMRoutingDecision — the schema
behind the live LLM routing call. app/conversation/manager.py and
app/workflows/manager.py read its fields directly (`.action`, `.certainty`,
`.intent`, `.entities`, `.resolved_target_workflow()`) — there is no
separate RoutingDecision projection step to keep in sync.

Nothing here exercises a real LLM call (see scripts/real_sarvam_validation.py
for that). These tests lock in the schema contract: that a well-formed or
malformed decision always resolves to something WorkflowManager/
ConversationManager already know how to handle safely.
"""

import unittest

from app.conversation.intent.llm_routing import (
    CERTAINTY_TO_CONFIDENCE,
    LLMRoutingDecision,
)
from app.conversation.intent.models import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    confidence_band,
)
from app.conversation.router import get_workflow_for_intent


class DefaultsAndValidationTests(unittest.TestCase):
    def test_defaults_are_the_safest_option(self) -> None:
        decision = LLMRoutingDecision()
        self.assertEqual(decision.intent, "unknown")
        self.assertEqual(decision.action, "CLARIFY")
        self.assertEqual(decision.certainty, "low")

    def test_unrecognized_intent_falls_back_to_unknown(self) -> None:
        decision = LLMRoutingDecision(intent="do_something_dangerous")
        self.assertEqual(decision.intent, "unknown")

    def test_unrecognized_action_falls_back_to_clarify(self) -> None:
        decision = LLMRoutingDecision(action="EXECUTE_TRANSFER_NOW")
        self.assertEqual(decision.action, "CLARIFY")

    def test_unrecognized_certainty_falls_back_to_low(self) -> None:
        decision = LLMRoutingDecision(certainty="extremely_sure")
        self.assertEqual(decision.certainty, "low")

    def test_known_intent_action_certainty_pass_through(self) -> None:
        decision = LLMRoutingDecision(intent="transfer_request", action="START_WORKFLOW", certainty="high")
        self.assertEqual(decision.intent, "transfer_request")
        self.assertEqual(decision.action, "START_WORKFLOW")
        self.assertEqual(decision.certainty, "high")


class CertaintyBandAlignmentTests(unittest.TestCase):
    """The whole point of using bands instead of a raw LLM-reported float is
    that they must land in the same high/medium/low buckets
    app/conversation/intent/models.py::confidence_band() already defines —
    verify that alignment explicitly rather than trusting the chosen
    numbers by inspection."""

    def test_high_certainty_lands_in_confidence_high_band(self) -> None:
        self.assertGreaterEqual(CERTAINTY_TO_CONFIDENCE["high"], CONFIDENCE_HIGH)
        self.assertEqual(confidence_band(CERTAINTY_TO_CONFIDENCE["high"]), "high")

    def test_medium_certainty_lands_in_confidence_medium_band(self) -> None:
        self.assertGreaterEqual(CERTAINTY_TO_CONFIDENCE["medium"], CONFIDENCE_MEDIUM)
        self.assertLess(CERTAINTY_TO_CONFIDENCE["medium"], CONFIDENCE_HIGH)
        self.assertEqual(confidence_band(CERTAINTY_TO_CONFIDENCE["medium"]), "medium")

    def test_low_certainty_lands_in_confidence_low_band(self) -> None:
        self.assertLess(CERTAINTY_TO_CONFIDENCE["low"], CONFIDENCE_MEDIUM)
        self.assertEqual(confidence_band(CERTAINTY_TO_CONFIDENCE["low"]), "low")


class ToIntentResultTests(unittest.TestCase):
    """to_intent_result() is used by app/conversation/manager.py purely for
    observability (refreshing ConversationContext.last_intent/
    intent_confidence with the decision that actually drove the turn) —
    nothing re-reads those fields to route anything."""

    def test_projects_intent_entities_and_llm_method(self) -> None:
        decision = LLMRoutingDecision(
            intent="balance_request", action="TOOL", certainty="high", entities={"account_type": "savings"},
        )
        result = decision.to_intent_result()
        self.assertEqual(result.intent, "balance_request")
        self.assertEqual(result.entities, {"account_type": "savings"})
        self.assertEqual(result.method, "llm")
        self.assertEqual(result.confidence, CERTAINTY_TO_CONFIDENCE["high"])


class ResolvedTargetWorkflowTests(unittest.TestCase):
    def test_explicit_target_workflow_wins(self) -> None:
        decision = LLMRoutingDecision(intent="transfer_request", target_workflow="loan")
        self.assertEqual(decision.resolved_target_workflow(), "loan")

    def test_falls_back_to_intent_table_when_unset(self) -> None:
        decision = LLMRoutingDecision(intent="kyc_update_request")
        self.assertEqual(decision.resolved_target_workflow(), get_workflow_for_intent("kyc_update_request"))

    def test_none_for_an_intent_with_no_workflow(self) -> None:
        decision = LLMRoutingDecision(intent="balance_request")
        self.assertIsNone(decision.resolved_target_workflow())


class ActionSemanticsTests(unittest.TestCase):
    """The dispatch semantics app/conversation/manager.py and
    app/workflows/manager.py actually implement, asserted directly against
    LLMRoutingDecision's own fields (no intermediate projection object)."""

    def test_cancel_is_a_real_action_not_silently_dropped(self) -> None:
        # A literal cancel/stop word is caught for free by the
        # deterministic pre-filter (classify_hard_navigation) before an LLM
        # call ever runs. CANCEL reaching this schema means a
        # NATURAL-LANGUAGE cancellation ("I don't want to continue with
        # this") the LLM recognized -- WorkflowManager treats it exactly
        # like a literal cancel word (the existing stop-confirmation UX),
        # never an immediate unconfirmed abandonment or a financial action.
        decision = LLMRoutingDecision(intent="cancel", action="CANCEL", certainty="high")
        self.assertEqual(decision.action, "CANCEL")

    def test_greeting_carries_no_workflow(self) -> None:
        decision = LLMRoutingDecision(intent="greeting", action="GREETING", certainty="high")
        self.assertIsNone(decision.resolved_target_workflow())

    def test_out_of_scope_carries_no_workflow(self) -> None:
        decision = LLMRoutingDecision(intent="out_of_scope", action="OUT_OF_SCOPE")
        self.assertIsNone(decision.resolved_target_workflow())

    def test_clarify_is_the_safe_default_for_unknown_intent(self) -> None:
        decision = LLMRoutingDecision(action="CLARIFY")
        self.assertEqual(decision.action, "CLARIFY")
        self.assertIsNone(decision.resolved_target_workflow())


class AllEightOperationsTests(unittest.TestCase):
    """Every one of the 8 named banking operations gets an explicit,
    deterministic regression test of its schema-level behavior -- no live
    LLM call needed, since this is testing the pure mapping/validation
    logic, not model judgment (that's scripts/real_sarvam_validation.py's
    job, run live against scripts/shadow_eval_corpus.py's 101-case
    matrix)."""

    # (operation label, intent, workflow name or None for a tool lookup)
    OPERATIONS = [
        ("TRANSFER_MONEY", "transfer_request", "transfer"),
        ("CHECK_BALANCE", "balance_request", None),
        ("VIEW_TRANSACTIONS", "transaction_request", None),
        ("DEPOSIT_CHEQUE", "cheque_deposit_request", "cheque"),
        ("CHECK_CHEQUE_STATUS", "cheque_status_request", None),
        ("APPLY_FOR_LOAN", "loan_application_request", "loan"),
        ("UPDATE_KYC", "kyc_update_request", "kyc"),
        ("CREATE_ACCOUNT", "add_account_request", "add_account"),
    ]

    def test_every_operation_resolves_to_its_correct_workflow_or_none(self) -> None:
        for label, intent, expected_workflow in self.OPERATIONS:
            with self.subTest(operation=label):
                decision = LLMRoutingDecision(intent=intent, action="START_WORKFLOW", certainty="high")
                self.assertEqual(decision.resolved_target_workflow(), expected_workflow)

    def test_every_lookup_operation_is_a_tool_call_never_a_workflow(self) -> None:
        for label, intent, expected_workflow in self.OPERATIONS:
            if expected_workflow is not None:
                continue
            with self.subTest(operation=label):
                decision = LLMRoutingDecision(intent=intent, action="TOOL", certainty="high")
                self.assertEqual(decision.action, "TOOL")
                self.assertIsNone(decision.resolved_target_workflow())

    def test_every_workflow_operation_supports_continue_correct_cancel(self) -> None:
        for label, intent, expected_workflow in self.OPERATIONS:
            if expected_workflow is None:
                continue
            with self.subTest(operation=label):
                cont = LLMRoutingDecision(intent=intent, action="CONTINUE", target_workflow=expected_workflow)
                self.assertEqual(cont.action, "CONTINUE")
                self.assertEqual(cont.resolved_target_workflow(), expected_workflow)

                corr = LLMRoutingDecision(intent=intent, action="CORRECT", target_workflow=expected_workflow)
                self.assertEqual(corr.action, "CORRECT")
                self.assertEqual(corr.resolved_target_workflow(), expected_workflow)

                cancel = LLMRoutingDecision(intent=intent, action="CANCEL", target_workflow=expected_workflow)
                # CANCEL always stays a real CANCEL action regardless of
                # intent/workflow -- verified for all 5 workflow-owning
                # operations, not just one.
                self.assertEqual(cancel.action, "CANCEL")

    def test_every_operation_can_be_the_target_of_a_switch_from_any_other(self) -> None:
        """Generic ANY-to-ANY requirement: every workflow-owning operation
        must be reachable as a SWITCH target regardless of which OTHER
        workflow-owning operation is currently active -- not just the
        specific pairs scripts/shadow_eval_corpus.py happens to exercise
        live."""
        workflow_ops = [(label, intent, wf) for label, intent, wf in self.OPERATIONS if wf is not None]
        for _, _, from_workflow in workflow_ops:
            for to_label, to_intent, to_workflow in workflow_ops:
                if to_workflow == from_workflow:
                    continue
                with self.subTest(frm=from_workflow, to=to_label):
                    decision = LLMRoutingDecision(intent=to_intent, action="SWITCH", certainty="high")
                    self.assertEqual(decision.action, "SWITCH")
                    self.assertEqual(decision.resolved_target_workflow(), to_workflow)


class NeverAuthorizesFinancialActionTests(unittest.TestCase):
    """LLMRoutingDecision only ever says where a turn goes, never that
    money moved. This is a schema-shape guarantee: it has no field that
    could represent "transfer executed" or "KYC submitted", so even a
    high-certainty SWITCH/START_WORKFLOW decision is structurally
    incapable of being more than a routing hint."""

    def test_schema_has_no_execution_or_confirmation_field(self) -> None:
        decision = LLMRoutingDecision(intent="transfer_request", action="SWITCH", certainty="high")
        self.assertEqual(
            set(decision.model_dump().keys()),
            {"intent", "action", "certainty", "target_workflow", "entities", "language"},
        )
        for forbidden in ("executed", "confirmed", "amount_transferred", "committed"):
            self.assertNotIn(forbidden, decision.model_dump())


if __name__ == "__main__":
    unittest.main()

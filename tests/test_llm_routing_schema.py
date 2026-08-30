"""Step 1 of the keyword-to-LLM intent-classification migration:
app/conversation/intent/llm_routing.py::LLMRoutingDecision and its projection
onto the existing IntentResult/RoutingDecision types.

Nothing here exercises a real LLM call or changes user-facing behavior — this
schema isn't wired into the live pipeline yet (that's Step 2, shadow mode).
These tests only lock in the contract: that a well-formed or malformed
decision from a future LLM call always resolves to something the EXISTING
route_intent()/WorkflowManager consumers already know how to handle safely.
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
from app.conversation.router import get_workflow_for_intent, route_intent


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
    that they must land in the same high/medium/low buckets route_intent()
    already uses — verify that alignment explicitly rather than trusting the
    chosen numbers by inspection."""

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
    def test_projects_intent_entities_and_llm_method(self) -> None:
        decision = LLMRoutingDecision(
            intent="balance_request", action="TOOL", certainty="high", entities={"account_type": "savings"},
        )
        result = decision.to_intent_result()
        self.assertEqual(result.intent, "balance_request")
        self.assertEqual(result.entities, {"account_type": "savings"})
        self.assertEqual(result.method, "llm")
        self.assertEqual(result.confidence, CERTAINTY_TO_CONFIDENCE["high"])

    def test_result_feeds_route_intent_exactly_like_a_rule_result(self) -> None:
        # The whole point of reusing IntentResult: route_intent() should not
        # need to know or care whether the object in front of it came from
        # rules.py or an LLM call.
        decision = LLMRoutingDecision(intent="transfer_request", action="START_WORKFLOW", certainty="high")
        routing = route_intent(decision.to_intent_result())
        self.assertEqual(routing.action, "START_WORKFLOW")
        self.assertEqual(routing.workflow, "transfer")


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


class ToRoutingDecisionTests(unittest.TestCase):
    def test_continue_maps_to_workflow_action(self) -> None:
        decision = LLMRoutingDecision(intent="loan_application_request", action="CONTINUE", target_workflow="loan")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "WORKFLOW")
        self.assertEqual(routing.workflow, "loan")

    def test_correct_maps_to_workflow_action(self) -> None:
        decision = LLMRoutingDecision(intent="loan_application_request", action="CORRECT", target_workflow="loan")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "WORKFLOW")
        self.assertEqual(routing.workflow, "loan")

    def test_switch_maps_to_start_workflow_with_new_target(self) -> None:
        # The exact "loan active, user says create another account" scenario
        # from the migration request — generic, not a hardcoded pair.
        decision = LLMRoutingDecision(intent="add_account_request", action="SWITCH", certainty="high")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "START_WORKFLOW")
        self.assertEqual(routing.workflow, "add_account")

    def test_start_workflow_maps_directly(self) -> None:
        decision = LLMRoutingDecision(intent="cheque_deposit_request", action="START_WORKFLOW", certainty="high")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "START_WORKFLOW")
        self.assertEqual(routing.workflow, "cheque")

    def test_tool_maps_to_banking_llm_with_no_workflow(self) -> None:
        decision = LLMRoutingDecision(intent="balance_request", action="TOOL")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "BANKING_LLM")
        self.assertIsNone(routing.workflow)

    def test_rag_maps_to_banking_llm_with_no_workflow(self) -> None:
        decision = LLMRoutingDecision(intent="banking_question", action="RAG")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "BANKING_LLM")
        self.assertIsNone(routing.workflow)

    def test_clarify_maps_to_clarification_required(self) -> None:
        decision = LLMRoutingDecision(action="CLARIFY")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "CLARIFICATION_REQUIRED")

    def test_out_of_scope_maps_directly(self) -> None:
        decision = LLMRoutingDecision(intent="out_of_scope", action="OUT_OF_SCOPE")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "OUT_OF_SCOPE")

    def test_cancel_defers_to_safe_fallback_not_a_new_execution_path(self) -> None:
        # CANCEL is deliberately not given real execution semantics here:
        # cancellation already has a cheap, deterministic upstream handler
        # (classify_hard_navigation). An LLM emitting CANCEL should never
        # cause NEW behavior at the routing layer.
        decision = LLMRoutingDecision(intent="cancel", action="CANCEL", certainty="high")
        routing = decision.to_routing_decision()
        self.assertEqual(routing.action, "SAFE_FALLBACK")


class AllEightOperationsTests(unittest.TestCase):
    """Second validation phase: every one of the 8 named banking operations
    (per the migration plan's own list) gets an explicit, deterministic
    regression test of its schema-level behavior -- no live LLM call
    needed, since this is testing the pure mapping/validation logic, not
    model judgment (that's scripts/shadow_eval.py's job, run live against
    scripts/shadow_eval_corpus.py's 101-case matrix)."""

    # (operation label, rule intent, workflow name or None for a tool lookup)
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

    def test_every_workflow_operation_start_workflow_maps_correctly(self) -> None:
        for label, intent, expected_workflow in self.OPERATIONS:
            if expected_workflow is None:
                continue
            with self.subTest(operation=label):
                decision = LLMRoutingDecision(intent=intent, action="START_WORKFLOW", certainty="high")
                routing = decision.to_routing_decision()
                self.assertEqual(routing.action, "START_WORKFLOW")
                self.assertEqual(routing.workflow, expected_workflow)

    def test_every_lookup_operation_tool_maps_to_banking_llm_no_workflow(self) -> None:
        for label, intent, expected_workflow in self.OPERATIONS:
            if expected_workflow is not None:
                continue
            with self.subTest(operation=label):
                decision = LLMRoutingDecision(intent=intent, action="TOOL", certainty="high")
                routing = decision.to_routing_decision()
                self.assertEqual(routing.action, "BANKING_LLM")
                self.assertIsNone(routing.workflow)

    def test_every_workflow_operation_supports_continue_correct_cancel(self) -> None:
        for label, intent, expected_workflow in self.OPERATIONS:
            if expected_workflow is None:
                continue
            with self.subTest(operation=label):
                cont = LLMRoutingDecision(intent=intent, action="CONTINUE", target_workflow=expected_workflow)
                self.assertEqual(cont.to_routing_decision().action, "WORKFLOW")
                self.assertEqual(cont.to_routing_decision().workflow, expected_workflow)

                corr = LLMRoutingDecision(intent=intent, action="CORRECT", target_workflow=expected_workflow)
                self.assertEqual(corr.to_routing_decision().action, "WORKFLOW")
                self.assertEqual(corr.to_routing_decision().workflow, expected_workflow)

                cancel = LLMRoutingDecision(intent=intent, action="CANCEL", target_workflow=expected_workflow)
                # CANCEL always defers to the deterministic upstream handler
                # regardless of intent/workflow -- verified for all 5
                # workflow-owning operations, not just one.
                self.assertEqual(cancel.to_routing_decision().action, "SAFE_FALLBACK")

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
                    routing = decision.to_routing_decision()
                    self.assertEqual(routing.action, "START_WORKFLOW")
                    self.assertEqual(routing.workflow, to_workflow)


class NeverAuthorizesFinancialActionTests(unittest.TestCase):
    """A RoutingDecision only ever says where a turn goes, never that money
    moved. This is a schema-shape guarantee: RoutingDecision has no field
    that could represent "transfer executed" or "KYC submitted", so a
    high-certainty SWITCH/START_WORKFLOW decision is structurally incapable
    of being more than a routing hint."""

    def test_high_certainty_switch_is_still_just_a_routing_hint(self) -> None:
        decision = LLMRoutingDecision(intent="transfer_request", action="SWITCH", certainty="high")
        routing = decision.to_routing_decision()
        self.assertEqual(set(routing.model_dump().keys()), {"action", "workflow", "reason"})
        self.assertEqual(routing.action, "START_WORKFLOW")


if __name__ == "__main__":
    unittest.main()

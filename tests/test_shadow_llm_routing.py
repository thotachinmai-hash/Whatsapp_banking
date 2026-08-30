"""Step 2 of the keyword-to-LLM intent-classification migration: the
shadow-mode LLM routing call (app/conversation/intent/llm_routing.py) and
its wiring into ConversationManager (app/conversation/manager.py).

Two properties matter most here and are tested explicitly:
1. Off by default, and completely inert when off -- no task is created, no
   LLM call happens, nothing is logged.
2. When on, it NEVER blocks or changes the turn's response -- it only logs
   a comparison in the background. The existing rule pipeline stays fully
   authoritative.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from app.conversation.context import ConversationContext
from app.conversation.intent.llm_routing import (
    LLMRoutingDecision,
    classify_and_route_llm,
    is_shadow_llm_routing_enabled,
)
from app.conversation.intent.models import IntentResult
from app.conversation.manager import ConversationManager


def _fresh_context(phone_number="441111111111", current_workflow=None, current_step=None):
    ctx = ConversationContext(phone_number=phone_number)
    ctx.current_workflow = current_workflow
    ctx.current_step = current_step
    return ctx


class _FakeWorkflowManager:
    def __init__(self):
        self.transfer_handler = object()

    def handle(self, phone_number, query, parsed_document=None, trace_id="", intent_result=None):
        return {"handled": False, "response": None}

    def start_requested(self, phone_number, query, trace_id=""):
        return {"handled": False, "response": None}


async def _fake_llm_fallback(query, phone_number, trace_id, parsed_document=None):
    return f"llm-answer:{query}"


class FlagDefaultTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SHADOW_LLM_ROUTING_ENABLED", None)
            self.assertFalse(is_shadow_llm_routing_enabled())

    def test_enabled_when_env_var_set(self) -> None:
        with patch.dict(os.environ, {"SHADOW_LLM_ROUTING_ENABLED": "true"}):
            self.assertTrue(is_shadow_llm_routing_enabled())


class CustomerContextTests(unittest.IsolatedAsyncioTestCase):
    """Registration status (ConversationContext.is_registered, a real
    DB-backed fact set by build_context()) must reach the prompt so the LLM
    can distinguish add_account_request from registration_request without
    guessing from ambiguous phrasing -- confirmed live this session:
    "मुझे नया बैंक अकाउंट खोलना है" resolved identically regardless of
    registration status until this context was threaded through."""

    async def test_registration_status_is_included_in_the_prompt(self) -> None:
        from app.conversation.intent.llm_routing import _routing_messages
        messages = _routing_messages("hi", "none", "already a registered customer")
        self.assertIn("already a registered customer", messages[1]["content"])

    async def test_registered_customer_context_reaches_the_prompt_call(self) -> None:
        captured = {}

        def fake_completions(**kwargs):
            captured["messages"] = kwargs["messages"]
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {
                "content": '{"intent": "unknown", "action": "CLARIFY", "certainty": "low"}'
            })()})]})()

        context = ConversationContext(phone_number="441111111111")
        context.is_registered = True
        with patch("app.conversation.intent.llm_routing._get_sarvam_client") as mock_client, \
             patch("app.conversation.intent.llm_routing._get_fast_model", return_value="sarvam-105b-conversations"):
            mock_client.return_value.chat.completions = fake_completions
            await classify_and_route_llm("I want to open a bank account", context=context, trace_id="t1")
        self.assertIn("already a registered customer", captured["messages"][1]["content"])


class ClassifyAndRouteLlmTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_a_well_formed_llm_response(self) -> None:
        fake_response = type(
            "R", (), {"choices": [type("C", (), {"message": type("M", (), {
                "content": (
                    '{"intent": "add_account_request", "action": "SWITCH", "certainty": "high", '
                    '"target_workflow": "add_account", "entities": {}, "language": "en"}'
                )
            })()})]},
        )()
        with patch("app.conversation.intent.llm_routing._get_sarvam_client") as mock_client, \
             patch("app.conversation.intent.llm_routing._get_fast_model", return_value="sarvam-105b-conversations"):
            mock_client.return_value.chat.completions = lambda **kwargs: fake_response
            result = await classify_and_route_llm("I want to create another bank account", context=None, trace_id="t1")

        self.assertIsInstance(result, LLMRoutingDecision)
        self.assertEqual(result.intent, "add_account_request")
        self.assertEqual(result.action, "SWITCH")
        self.assertEqual(result.target_workflow, "add_account")

    async def test_llm_call_failure_returns_none_not_raise(self) -> None:
        with patch("app.conversation.intent.llm_routing._get_sarvam_client", side_effect=Exception("network down")):
            result = await classify_and_route_llm("hello", context=None, trace_id="t1")
        self.assertIsNone(result)

    async def test_malformed_json_returns_none(self) -> None:
        fake_response = type(
            "R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "not json"})()})]},
        )()
        with patch("app.conversation.intent.llm_routing._get_sarvam_client") as mock_client, \
             patch("app.conversation.intent.llm_routing._get_fast_model", return_value="sarvam-105b-conversations"):
            mock_client.return_value.chat.completions = lambda **kwargs: fake_response
            result = await classify_and_route_llm("hello", context=None, trace_id="t1")
        self.assertIsNone(result)


class ShadowModeWiringTests(unittest.IsolatedAsyncioTestCase):
    """Exercises _fire_shadow_llm_routing / _shadow_llm_routing_comparison
    directly rather than through a full handle_message() turn, since those
    two methods are the entire surface of Step 2's behavior change."""

    def _manager(self):
        return ConversationManager(workflow_manager=_FakeWorkflowManager())

    async def test_disabled_creates_no_background_task(self) -> None:
        manager = self._manager()
        context = _fresh_context()
        with patch("app.conversation.manager.is_shadow_llm_routing_enabled", return_value=False), \
             patch("app.conversation.manager.classify_and_route_llm") as mock_llm:
            manager._fire_shadow_llm_routing(
                context, "hello", IntentResult(intent="greeting", confidence=0.9), "441111111111", "t1"
            )
            await asyncio.sleep(0)
        self.assertEqual(len(manager._shadow_tasks), 0)
        mock_llm.assert_not_called()

    async def test_enabled_creates_a_background_task_and_calls_the_llm(self) -> None:
        manager = self._manager()
        context = _fresh_context()
        fake_decision = LLMRoutingDecision(intent="greeting", action="CLARIFY", certainty="low")
        with patch("app.conversation.manager.is_shadow_llm_routing_enabled", return_value=True), \
             patch("app.conversation.manager.classify_and_route_llm", new_callable=AsyncMock, return_value=fake_decision) as mock_llm:
            manager._fire_shadow_llm_routing(
                context, "hello", IntentResult(intent="greeting", confidence=0.9), "441111111111", "t1"
            )
            # Let the background task actually run to completion.
            await asyncio.gather(*manager._shadow_tasks)
        mock_llm.assert_awaited_once()

    async def test_fire_and_forget_returns_before_the_llm_call_completes(self) -> None:
        """The whole point of asyncio.create_task here: firing it must not
        block on a slow LLM call. Simulate a slow call and assert control
        returns to the caller immediately."""
        manager = self._manager()
        context = _fresh_context()

        async def _slow_llm(*args, **kwargs):
            await asyncio.sleep(5)
            return LLMRoutingDecision()

        with patch("app.conversation.manager.is_shadow_llm_routing_enabled", return_value=True), \
             patch("app.conversation.manager.classify_and_route_llm", side_effect=_slow_llm):
            start = asyncio.get_event_loop().time()
            manager._fire_shadow_llm_routing(
                context, "hello", IntentResult(intent="greeting", confidence=0.9), "441111111111", "t1"
            )
            elapsed = asyncio.get_event_loop().time() - start
        self.assertLess(elapsed, 0.5)
        # Clean up the still-pending task so the test doesn't leak it.
        for task in list(manager._shadow_tasks):
            task.cancel()

    async def test_llm_failure_inside_the_background_task_never_raises(self) -> None:
        manager = self._manager()
        context = _fresh_context()
        with patch("app.conversation.manager.is_shadow_llm_routing_enabled", return_value=True), \
             patch("app.conversation.manager.classify_and_route_llm", new_callable=AsyncMock, side_effect=Exception("boom")):
            manager._fire_shadow_llm_routing(
                context, "hello", IntentResult(intent="greeting", confidence=0.9), "441111111111", "t1"
            )
            # Must not raise even though the underlying call blows up.
            await asyncio.gather(*manager._shadow_tasks)
        self.assertEqual(len(manager._shadow_tasks), 0)  # done_callback cleaned it up

    async def test_comparison_logs_agreement_when_rule_and_llm_match(self) -> None:
        manager = self._manager()
        context = _fresh_context(current_workflow=None)
        rule_result = IntentResult(intent="transfer_request", confidence=0.9)
        llm_decision = LLMRoutingDecision(intent="transfer_request", action="START_WORKFLOW", certainty="high")
        with patch("app.conversation.manager.classify_and_route_llm", new_callable=AsyncMock, return_value=llm_decision), \
             patch("app.conversation.manager.logger") as mock_logger:
            await manager._shadow_llm_routing_comparison(context, "send 500 to Priya", rule_result, "441111111111", "t1")
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        self.assertIn("agree=yes", logged)

    async def test_comparison_logs_disagreement_when_they_differ(self) -> None:
        manager = self._manager()
        context = _fresh_context(current_workflow="loan", current_step="COLLECT_INCOME")
        rule_result = IntentResult(intent="workflow_correction", confidence=0.8)
        llm_decision = LLMRoutingDecision(
            intent="add_account_request", action="SWITCH", certainty="high", target_workflow="add_account",
        )
        with patch("app.conversation.manager.classify_and_route_llm", new_callable=AsyncMock, return_value=llm_decision), \
             patch("app.conversation.manager.logger") as mock_logger:
            await manager._shadow_llm_routing_comparison(context, "actually create another account", rule_result, "441111111111", "t1")
        logged = " ".join(str(c) for c in mock_logger.info.call_args_list)
        self.assertIn("agree=no", logged)
        self.assertIn("llm_target_workflow=add_account", logged)

    async def test_comparison_does_not_call_workflow_manager(self) -> None:
        """route_intent() is pure -- the shadow comparison must never touch
        WorkflowManager.handle() (which mutates Redis workflow state), even
        though the live turn calls it separately for the same message."""
        manager = self._manager()
        context = _fresh_context(current_workflow="loan", current_step="COLLECT_INCOME")
        rule_result = IntentResult(intent="workflow_correction", confidence=0.8)
        llm_decision = LLMRoutingDecision(intent="workflow_correction", action="CORRECT")
        with patch("app.conversation.manager.classify_and_route_llm", new_callable=AsyncMock, return_value=llm_decision), \
             patch.object(manager.workflow_manager, "handle") as mock_handle:
            await manager._shadow_llm_routing_comparison(context, "actually 60000", rule_result, "441111111111", "t1")
        mock_handle.assert_not_called()


class FullTurnUnaffectedTests(unittest.IsolatedAsyncioTestCase):
    """Full handle_message() turns must produce byte-identical responses
    whether shadow mode is on or off -- Step 2's core promise."""

    def _patches(self, context):
        return (
            patch("app.conversation.manager.build_context", return_value=context),
            patch("app.conversation.manager.check_registration_gate", return_value=None),
            patch("app.conversation.manager.get_workflow", return_value=None),
            patch("app.conversation.manager.append_turn_to_session"),
        )

    async def test_response_identical_with_shadow_mode_on(self) -> None:
        context = _fresh_context()
        manager = ConversationManager(workflow_manager=_FakeWorkflowManager())
        p1, p2, p3, p4 = self._patches(context)
        with p1, p2, p3, p4, patch.object(manager.context_store, "save", return_value=True), \
             patch("app.conversation.manager.is_shadow_llm_routing_enabled", return_value=True), \
             patch("app.conversation.manager.classify_and_route_llm", new_callable=AsyncMock, return_value=None):
            response = await manager.handle_message(
                "441111111111", "What is an overdraft?", "t1", llm_fallback=_fake_llm_fallback
            )
        self.assertEqual(response, "llm-answer:What is an overdraft?")


if __name__ == "__main__":
    unittest.main()

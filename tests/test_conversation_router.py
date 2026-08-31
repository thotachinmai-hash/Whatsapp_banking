"""Full-pipeline integration tests for run_agent() -> ConversationManager,
proving the LLM routing decision (not a rule-based router) actually
controls the response for out-of-scope/eligibility-question/workflow-start/
active-workflow cases, and that the general LLM+tools agent is skipped
whenever it isn't needed.

The rule-based route_intent()/RouterRequiredCaseTests this file used to
contain tested app/conversation/router.py::route_intent() directly — that
function was deleted in the LLM-first routing migration (dead code once
classify_and_route_llm() became the sole routing decision; see
router.py's module docstring). Its financial-safety invariants (a
workflow only starts at high certainty, CANCEL/OUT_OF_SCOPE never start
one, an unmapped/failed decision degrades safely) are now covered by
tests/test_llm_routing_schema.py (which tests LLMRoutingDecision, the
type that replaced it) and tests/test_conversation_manager.py's
LlmRoutingSafetyTests.
"""

import unittest
from unittest.mock import AsyncMock, patch

from app.conversation.context import ConversationContext
from app.conversation.intent.llm_routing import LLMRoutingDecision


def _fresh_context(phone_number="441111111111"):
    return ConversationContext(phone_number=phone_number)


def _decision(intent="unknown", action="CLARIFY", certainty="high", target_workflow=None) -> LLMRoutingDecision:
    return LLMRoutingDecision(intent=intent, action=action, certainty=certainty, target_workflow=target_workflow)


class RunAgentRoutingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Drives run_agent() with collaborators mocked (no live Redis/Postgres/
    Sarvam needed) to prove the LLM routing decision actually controls the
    response for out_of_scope/eligibility-question/workflow-start/active-
    workflow cases, and that the general agent is skipped when it isn't
    needed."""

    def _patches(self):
        from app.agent import agent as agent_module

        return [
            patch("app.conversation.manager.check_registration_gate", return_value=None),
            patch("app.conversation.manager.build_context", side_effect=lambda *a, **k: _fresh_context()),
            patch.object(agent_module.conversation_context_store, "save", return_value=True),
            patch("app.conversation.manager.get_workflow", return_value=None),
            patch("app.conversation.manager.append_turn_to_session"),
        ]

    async def test_out_of_scope_message_never_calls_the_llm_agent(self):
        from app.agent import agent as agent_module

        patches = self._patches()
        decision = _decision(intent="out_of_scope", action="OUT_OF_SCOPE")
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("app.conversation.manager.classify_and_route_llm", return_value=decision), \
             patch.object(agent_module.workflow_manager, "handle", return_value={"handled": False, "response": None}), \
             patch.object(agent_module.workflow_manager, "start_requested") as mock_start_requested, \
             patch("app.agent.agent.build_agent") as mock_build_agent:

            response = await agent_module.run_agent(
                query="tell me a joke", phone_number="441111111111", trace_id="rt1"
            )

        mock_build_agent.assert_not_called()
        mock_start_requested.assert_not_called()
        self.assertIn("banking", response.lower())

    async def test_loan_eligibility_question_does_not_start_loan_workflow(self):
        from app.agent import agent as agent_module

        patches = self._patches()
        decision = _decision(intent="loan_eligibility_question", action="RAG")
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("app.conversation.manager.classify_and_route_llm", return_value=decision), \
             patch.object(agent_module.workflow_manager, "handle", return_value={"handled": False, "response": None}), \
             patch.object(agent_module.workflow_manager, "start_requested") as mock_start_requested, \
             patch("app.agent.agent.get_session_history", return_value=[]):

            fake_agent = AsyncMock()
            fake_agent.ainvoke = AsyncMock(return_value={
                "messages": [type("M", (), {"content": "Eligibility depends on income and other factors.", "name": None})()]
            })
            with patch("app.agent.agent.build_agent", return_value=fake_agent):
                response = await agent_module.run_agent(
                    query="I earn 5000 monthly and want a personal loan",
                    phone_number="441111111111",
                    trace_id="rt2",
                )

        mock_start_requested.assert_not_called()
        self.assertIn("eligibility", response.lower())

    async def test_personal_loan_request_starts_loan_workflow(self):
        from app.agent import agent as agent_module

        patches = self._patches()
        decision = _decision(intent="loan_application_request", action="START_WORKFLOW", target_workflow="loan")
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch("app.conversation.manager.classify_and_route_llm", return_value=decision), \
             patch("app.conversation.manager.start_workflow_directly",
                   return_value={"handled": True, "response": "Loan application started."}), \
             patch.object(agent_module.workflow_manager, "handle", return_value={"handled": False, "response": None}), \
             patch("app.agent.agent.build_agent") as mock_build_agent:

            response = await agent_module.run_agent(
                query="I want a personal loan", phone_number="441111111111", trace_id="rt3"
            )

        mock_build_agent.assert_not_called()
        self.assertEqual(response, "Loan application started.")

    async def test_active_workflow_input_never_reaches_routing_or_llm(self):
        from app.agent import agent as agent_module

        active_context = _fresh_context()
        active_context.current_workflow = "transfer"
        with patch("app.conversation.manager.check_registration_gate", return_value=None), \
             patch("app.conversation.manager.build_context", side_effect=lambda *a, **k: active_context), \
             patch.object(agent_module.conversation_context_store, "save", return_value=True), \
             patch("app.conversation.manager.get_workflow", return_value=None), \
             patch("app.conversation.manager.append_turn_to_session"), \
             patch("app.conversation.manager.classify_and_route_llm") as mock_route, \
             patch.object(
                 agent_module.workflow_manager,
                 "handle",
                 return_value={"handled": True, "response": "Which account should we use?"},
             ), \
             patch.object(agent_module.workflow_manager, "start_requested") as mock_start_requested, \
             patch("app.agent.agent.build_agent") as mock_build_agent:

            response = await agent_module.run_agent(
                query="500", phone_number="441111111111", trace_id="rt4"
            )

        mock_route.assert_not_called()
        mock_start_requested.assert_not_called()
        mock_build_agent.assert_not_called()
        self.assertEqual(response, "Which account should we use?")


if __name__ == "__main__":
    unittest.main()

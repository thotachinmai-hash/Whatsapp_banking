import unittest
from unittest.mock import MagicMock, patch

from app.services import llm_understanding as lu


def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class FeatureFlagTests(unittest.TestCase):
    def test_default_is_disabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(lu.is_llm_fallback_enabled())

    def test_enabled_values(self) -> None:
        for value in ("1", "true", "True", "yes", "on"):
            with patch.dict("os.environ", {"LLM_FALLBACK_ENABLED": value}):
                self.assertTrue(lu.is_llm_fallback_enabled())

    def test_disabled_values(self) -> None:
        for value in ("0", "false", "no", "off", ""):
            with patch.dict("os.environ", {"LLM_FALLBACK_ENABLED": value}):
                self.assertFalse(lu.is_llm_fallback_enabled())


class InterpretChoiceLlmTests(unittest.TestCase):
    def test_valid_choice_returned(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('{"choice": "2"}')
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.interpret_choice_llm("the second one", ["1", "2", "3"], "pick an account")
        self.assertEqual(result, "2")

    def test_choice_not_in_options_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response('{"choice": "5"}')
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.interpret_choice_llm("garbage", ["1", "2", "3"], "pick an account")
        self.assertIsNone(result)

    def test_malformed_json_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response("not json at all")
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.interpret_choice_llm("hmm", ["1", "2"], "context")
        self.assertIsNone(result)

    def test_client_exception_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.interpret_choice_llm("hmm", ["1", "2"], "context")
        self.assertIsNone(result)

    def test_empty_input_short_circuits_without_calling_client(self) -> None:
        client = MagicMock()
        with patch.object(lu, "_get_client", return_value=client):
            self.assertIsNone(lu.interpret_choice_llm("", ["1", "2"], "context"))
            self.assertIsNone(lu.interpret_choice_llm("hi", [], "context"))
        client.chat.completions.create.assert_not_called()


class AnswerSideQuestionTests(unittest.TestCase):
    def test_returns_answer_text(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response("Interest rates vary by loan type.")
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.answer_side_question("what's the interest rate?", "cheque", "UPLOAD_CHEQUE")
        self.assertEqual(result, "Interest rates vary by loan type.")

    def test_none_sentinel_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response("NONE")
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.answer_side_question("approve my loan now", "loan", "CONFIRM_LOAN")
        self.assertIsNone(result)

    def test_client_exception_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.answer_side_question("hello", "loan", "CONFIRM_LOAN")
        self.assertIsNone(result)


class DetectStepOrWorkflowJumpTests(unittest.TestCase):
    def test_confident_jump_returns_target(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            '{"target_workflow": "loan", "confidence": 0.9}'
        )
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.detect_step_or_workflow_jump("actually let me apply for a loan", "transfer", "SELECT_BENEFICIARY")
        self.assertIsNotNone(result)
        self.assertEqual(result.target_workflow, "loan")

    def test_low_confidence_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            '{"target_workflow": "loan", "confidence": 0.2}'
        )
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.detect_step_or_workflow_jump("hmm maybe", "transfer", "SELECT_BENEFICIARY")
        self.assertIsNone(result)

    def test_same_workflow_target_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            '{"target_workflow": "transfer", "confidence": 0.95}'
        )
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.detect_step_or_workflow_jump("500 to Priya", "transfer", "SELECT_BENEFICIARY")
        self.assertIsNone(result)

    def test_client_exception_returns_none(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch.object(lu, "_get_client", return_value=client):
            result = lu.detect_step_or_workflow_jump("switch to loan", "transfer", "SELECT_BENEFICIARY")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

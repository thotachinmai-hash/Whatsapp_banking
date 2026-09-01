import unittest
from unittest.mock import MagicMock, patch

from app.services.language import detect_explicit_language_change, detect_language, should_attempt_detection, translate_text


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class ShouldAttemptDetectionTests(unittest.TestCase):
    def test_skips_short_messages(self) -> None:
        self.assertFalse(should_attempt_detection("ok"))
        self.assertFalse(should_attempt_detection("1"))
        self.assertFalse(should_attempt_detection(""))

    def test_skips_pure_ascii_without_an_llm_call(self) -> None:
        # The whole point of the gate — plain English text must never
        # reach the model.
        self.assertFalse(should_attempt_detection("check my balance please"))
        self.assertFalse(should_attempt_detection("transfer 1500 to Riya"))

    def test_flags_non_ascii_text_for_detection(self) -> None:
        self.assertTrue(should_attempt_detection("मेरा बैलेंस क्या है"))  # Hindi/Devanagari
        self.assertTrue(should_attempt_detection("cuál es mi saldo"))  # accented Spanish


class DetectLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_detected_supported_code(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("hi")
            self.assertEqual(await detect_language("मेरा बैलेंस क्या है"), "hi")

    async def test_unsupported_code_defaults_to_english(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("zz")
            self.assertEqual(await detect_language("some unusual text élan"), "en")

    async def test_api_failure_defaults_to_english(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.side_effect = RuntimeError("boom")
            self.assertEqual(await detect_language("cuál es mi saldo"), "en")

    async def test_short_text_never_calls_the_client(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            self.assertEqual(await detect_language("ok"), "en")
            mock_get_client.assert_not_called()


class TranslateTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_english_target_returns_original_unchanged(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            self.assertEqual(await translate_text("Your balance is 100", "en"), "Your balance is 100")
            mock_get_client.assert_not_called()

    async def test_unsupported_target_returns_original_unchanged(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            self.assertEqual(await translate_text("Your balance is 100", "zz"), "Your balance is 100")
            mock_get_client.assert_not_called()

    async def test_translates_when_target_supported(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("உங்கள் இருப்பு 100")
            self.assertEqual(await translate_text("Your balance is 100", "ta"), "உங்கள் இருப்பு 100")

    async def test_api_failure_returns_original_text(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.side_effect = RuntimeError("boom")
            self.assertEqual(await translate_text("Your balance is 100", "ta"), "Your balance is 100")

    async def test_empty_translated_response_falls_back_to_original(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("")
            self.assertEqual(await translate_text("Your balance is 100", "ta"), "Your balance is 100")

    async def test_max_length_passed_to_model_as_a_budget_instruction(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("Short")
            await translate_text("Edit amount", "te", max_length=20)
            _, kwargs = mock_get_client.return_value.chat.completions.call_args
            system_prompt = kwargs["messages"][0]["content"]
            self.assertIn("20 characters", system_prompt)
            self.assertIn("button/menu label", system_prompt)

    async def test_translation_over_budget_is_truncated(self) -> None:
        # Regression: confirmed live -- a literal translation of a short
        # English label ("Edit amount") routinely ran longer than
        # WhatsApp's own button-title limit even with the budget
        # instruction, silently dropping the interactive buttons in favor
        # of a plain-numbered-text fallback. This is the last-resort
        # safety net when the model still overshoots.
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response(
                "This translation is deliberately far too long to fit"
            )
            result = await translate_text("Edit amount", "te", max_length=20)
            self.assertLessEqual(len(result), 20)

    async def test_translation_within_budget_is_returned_unchanged(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("Short label")
            result = await translate_text("Edit amount", "te", max_length=20)
            self.assertEqual(result, "Short label")

    async def test_no_max_length_uses_the_full_sentence_prompt(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.return_value = _mock_response("Full translation")
            await translate_text("Your balance is 100", "ta")
            _, kwargs = mock_get_client.return_value.chat.completions.call_args
            system_prompt = kwargs["messages"][0]["content"]
            self.assertNotIn("button/menu label", system_prompt)


class DetectExplicitLanguageChangeTests(unittest.TestCase):
    def test_recognizes_common_phrasings(self) -> None:
        self.assertEqual(detect_explicit_language_change("reply in Tamil please"), "ta")
        self.assertEqual(detect_explicit_language_change("please switch to Hindi"), "hi")
        self.assertEqual(detect_explicit_language_change("speak English"), "en")

    def test_plain_message_in_another_language_is_not_a_change_request(self) -> None:
        # This function only recognizes META-requests about language, not
        # text merely written in one — should_attempt_detection/
        # detect_language handle the latter for non-ASCII text.
        self.assertIsNone(detect_explicit_language_change("cuál es mi saldo"))
        self.assertIsNone(detect_explicit_language_change("check my balance"))

    def test_unsupported_language_name_returns_none(self) -> None:
        self.assertIsNone(detect_explicit_language_change("reply in Klingon"))

    def test_empty_message_returns_none(self) -> None:
        self.assertIsNone(detect_explicit_language_change(""))


if __name__ == "__main__":
    unittest.main()

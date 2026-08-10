import unittest
from unittest.mock import MagicMock, patch

from app.services.language import detect_language, should_attempt_detection, translate_text


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


class DetectLanguageTests(unittest.TestCase):
    def test_returns_detected_supported_code(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = _mock_response("hi")
            self.assertEqual(detect_language("मेरा बैलेंस क्या है"), "hi")

    def test_unsupported_code_defaults_to_english(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = _mock_response("zz")
            self.assertEqual(detect_language("some unusual text élan"), "en")

    def test_api_failure_defaults_to_english(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.side_effect = RuntimeError("boom")
            self.assertEqual(detect_language("cuál es mi saldo"), "en")

    def test_short_text_never_calls_the_client(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            self.assertEqual(detect_language("ok"), "en")
            mock_get_client.assert_not_called()


class TranslateTextTests(unittest.TestCase):
    def test_english_target_returns_original_unchanged(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            self.assertEqual(translate_text("Your balance is 100", "en"), "Your balance is 100")
            mock_get_client.assert_not_called()

    def test_unsupported_target_returns_original_unchanged(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            self.assertEqual(translate_text("Your balance is 100", "zz"), "Your balance is 100")
            mock_get_client.assert_not_called()

    def test_translates_when_target_supported(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = _mock_response("Su saldo es 100")
            self.assertEqual(translate_text("Your balance is 100", "es"), "Su saldo es 100")

    def test_api_failure_returns_original_text(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.side_effect = RuntimeError("boom")
            self.assertEqual(translate_text("Your balance is 100", "es"), "Your balance is 100")

    def test_empty_translated_response_falls_back_to_original(self) -> None:
        with patch("app.services.language._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = _mock_response("")
            self.assertEqual(translate_text("Your balance is 100", "es"), "Your balance is 100")


if __name__ == "__main__":
    unittest.main()

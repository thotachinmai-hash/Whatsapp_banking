import unittest
from unittest.mock import MagicMock, patch

from app.services.transcription import transcribe_audio


def _mock_transcription(transcript: str, language_code: str) -> MagicMock:
    result = MagicMock()
    result.transcript = transcript
    result.language_code = language_code
    return result


class TranscribeAudioLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_bcp47_language_code_to_iso_code(self) -> None:
        with patch("app.services.transcription.get_sarvam_client") as mock_get_client:
            mock_get_client.return_value.speech_to_text.transcribe.return_value = _mock_transcription(
                "mera balance kya hai", "hi-IN"
            )
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t1")

        self.assertEqual(text, "mera balance kya hai")
        self.assertEqual(language_code, "hi")

    async def test_english_maps_to_en(self) -> None:
        with patch("app.services.transcription.get_sarvam_client") as mock_get_client:
            mock_get_client.return_value.speech_to_text.transcribe.return_value = _mock_transcription(
                "check my balance", "en-IN"
            )
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t2")

        self.assertEqual(language_code, "en")

    async def test_short_transcript_drops_language_tag(self) -> None:
        with patch("app.services.transcription.get_sarvam_client") as mock_get_client:
            mock_get_client.return_value.speech_to_text.transcribe.return_value = _mock_transcription(
                "Ok", "pt-IN"
            )
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t3")

        self.assertIsNone(language_code)

    async def test_failure_returns_none_none(self) -> None:
        with patch("app.services.transcription.get_sarvam_client") as mock_get_client:
            mock_get_client.return_value.speech_to_text.transcribe.side_effect = RuntimeError("boom")
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t4")

        self.assertIsNone(text)
        self.assertIsNone(language_code)


if __name__ == "__main__":
    unittest.main()

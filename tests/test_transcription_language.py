import unittest
from unittest.mock import MagicMock, patch

from app.services.transcription import transcribe_audio


def _mock_transcription(text: str, language: str) -> MagicMock:
    result = MagicMock()
    result.text = text
    result.language = language
    return result


class TranscribeAudioLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_whisper_language_name_to_iso_code(self) -> None:
        with patch("app.services.transcription.groq_client") as mock_client:
            mock_client.audio.transcriptions.create.return_value = _mock_transcription(
                "mera balance kya hai", "hindi"
            )
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t1")

        self.assertEqual(text, "mera balance kya hai")
        self.assertEqual(language_code, "hi")

    async def test_english_maps_to_en(self) -> None:
        with patch("app.services.transcription.groq_client") as mock_client:
            mock_client.audio.transcriptions.create.return_value = _mock_transcription(
                "check my balance", "english"
            )
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t2")

        self.assertEqual(language_code, "en")

    async def test_unrecognized_language_name_returns_none_code(self) -> None:
        with patch("app.services.transcription.groq_client") as mock_client:
            mock_client.audio.transcriptions.create.return_value = _mock_transcription(
                "some text", "klingon"
            )
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t3")

        self.assertIsNone(language_code)

    async def test_failure_returns_none_none(self) -> None:
        with patch("app.services.transcription.groq_client") as mock_client:
            mock_client.audio.transcriptions.create.side_effect = RuntimeError("boom")
            text, language_code = await transcribe_audio(b"fake-audio-bytes", "t4")

        self.assertIsNone(text)
        self.assertIsNone(language_code)


if __name__ == "__main__":
    unittest.main()

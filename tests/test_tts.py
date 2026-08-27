import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.tts import synthesize_speech, synthesize_voice_note


class SynthesizeSpeechTests(unittest.IsolatedAsyncioTestCase):
    """synthesize_speech() checks SARVAM_API_KEY via os.getenv() before it
    ever touches the Sarvam client, so any test mocking get_sarvam_client
    (to test the actual request/response handling) needs a real-looking
    key present too, or it early-returns None regardless of what's
    mocked -- passing or failing for the wrong reason depending on
    whether the real environment happens to have a .env file. Setting a
    fake key here makes every test in this class deterministic
    regardless of the environment it runs in."""

    def setUp(self) -> None:
        self._env_patcher = patch.dict("os.environ", {"SARVAM_API_KEY": "test-key"})
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    async def test_returns_none_when_no_api_key(self) -> None:
        with patch("app.services.tts.os.getenv", return_value=""):
            self.assertIsNone(await synthesize_speech("hello"))

    async def test_returns_none_for_empty_text(self) -> None:
        self.assertIsNone(await synthesize_speech(""))
        self.assertIsNone(await synthesize_speech("   "))

    async def test_returns_wav_bytes_on_success(self) -> None:
        wav_bytes = b"RIFF....WAVEfmt "
        mock_response = SimpleNamespace(audios=[base64.b64encode(wav_bytes).decode("ascii")])
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = mock_response

        with patch("app.services.tts.get_sarvam_client", return_value=mock_client):
            result = await synthesize_speech("hello there")

        self.assertEqual(result, wav_bytes)
        mock_client.text_to_speech.convert.assert_called_once()
        self.assertEqual(mock_client.text_to_speech.convert.call_args.kwargs["language_code"], "en-IN")

    async def test_uses_matching_language_code_when_supported(self) -> None:
        wav_bytes = b"RIFF"
        mock_response = SimpleNamespace(audios=[base64.b64encode(wav_bytes).decode("ascii")])
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = mock_response

        with patch("app.services.tts.get_sarvam_client", return_value=mock_client):
            await synthesize_speech("namaste", language="hi")

        self.assertEqual(mock_client.text_to_speech.convert.call_args.kwargs["language_code"], "hi-IN")

    async def test_returns_none_when_response_has_no_audio(self) -> None:
        mock_response = SimpleNamespace(audios=[])
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = mock_response

        with patch("app.services.tts.get_sarvam_client", return_value=mock_client):
            result = await synthesize_speech("hello")

        self.assertIsNone(result)

    async def test_returns_none_on_exception(self) -> None:
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.side_effect = RuntimeError("boom")

        with patch("app.services.tts.get_sarvam_client", return_value=mock_client):
            result = await synthesize_speech("hello")

        self.assertIsNone(result)


class SynthesizeVoiceNoteTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_none_when_synthesis_fails(self) -> None:
        with patch("app.services.tts.synthesize_speech", new=AsyncMock(return_value=None)):
            result = await synthesize_voice_note("hello")
        self.assertIsNone(result)

    async def test_prefers_ogg_opus_when_conversion_succeeds(self) -> None:
        with patch("app.services.tts.synthesize_speech", new=AsyncMock(return_value=b"wav-bytes")), \
             patch("app.services.tts._wav_to_ogg_opus", new=AsyncMock(return_value=b"ogg-bytes")):
            result = await synthesize_voice_note("hello")
        self.assertEqual(result, (b"ogg-bytes", "audio/ogg; codecs=opus"))

    async def test_falls_back_to_wav_when_conversion_fails(self) -> None:
        with patch("app.services.tts.synthesize_speech", new=AsyncMock(return_value=b"wav-bytes")), \
             patch("app.services.tts._wav_to_ogg_opus", new=AsyncMock(return_value=None)):
            result = await synthesize_voice_note("hello")
        self.assertEqual(result, (b"wav-bytes", "audio/wav"))


if __name__ == "__main__":
    unittest.main()

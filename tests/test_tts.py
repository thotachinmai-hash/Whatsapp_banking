import base64
import wave
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.tts import _pcm_to_wav, synthesize_speech, synthesize_voice_note


class PcmToWavTests(unittest.TestCase):
    def test_produces_a_valid_wav_file(self) -> None:
        pcm = b"\x00\x01" * 1000  # arbitrary 16-bit samples
        wav_bytes = _pcm_to_wav(pcm)
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getnframes(), 1000)


class SynthesizeSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_none_when_no_api_key(self) -> None:
        with patch("app.services.tts.GEMINI_API_KEY", ""):
            self.assertIsNone(await synthesize_speech("hello"))

    async def test_returns_none_for_empty_text(self) -> None:
        with patch("app.services.tts.GEMINI_API_KEY", "fake-key"):
            self.assertIsNone(await synthesize_speech(""))
            self.assertIsNone(await synthesize_speech("   "))

    async def test_returns_wav_bytes_on_success(self) -> None:
        pcm = b"\x00\x01" * 500
        inline_data = {"data": base64.b64encode(pcm).decode("ascii")}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": inline_data}]}}]
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch("app.services.tts.GEMINI_API_KEY", "fake-key"), \
             patch("app.services.tts.httpx.AsyncClient", return_value=mock_client):
            result = await synthesize_speech("hello there")

        self.assertIsNotNone(result)
        with wave.open(io.BytesIO(result), "rb") as w:
            self.assertEqual(w.getnframes(), 500)

    async def test_returns_none_on_non_200_status(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch("app.services.tts.GEMINI_API_KEY", "fake-key"), \
             patch("app.services.tts.httpx.AsyncClient", return_value=mock_client):
            result = await synthesize_speech("hello")

        self.assertIsNone(result)

    async def test_returns_none_when_response_has_no_audio(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "oops"}]}}]}
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch("app.services.tts.GEMINI_API_KEY", "fake-key"), \
             patch("app.services.tts.httpx.AsyncClient", return_value=mock_client):
            result = await synthesize_speech("hello")

        self.assertIsNone(result)

    async def test_returns_none_on_exception(self) -> None:
        with patch("app.services.tts.GEMINI_API_KEY", "fake-key"), \
             patch("app.services.tts.httpx.AsyncClient", side_effect=RuntimeError("boom")):
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

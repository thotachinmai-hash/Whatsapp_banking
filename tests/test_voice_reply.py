import unittest
from unittest.mock import AsyncMock, patch

from app.services.message_handler import send_voice_reply


class SendVoiceReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_as_voice_when_synthesis_succeeds(self) -> None:
        with patch(
            "app.services.message_handler.synthesize_voice_note",
            new=AsyncMock(return_value=(b"ogg-bytes", "audio/ogg; codecs=opus")),
        ), \
             patch("app.services.message_handler.send_voice_message", new=AsyncMock(return_value=True)) as mock_send_voice, \
             patch("app.services.message_handler.render_and_send", new=AsyncMock()) as mock_render_text:
            result = await send_voice_reply("Your balance is 100", "447700900000", "t1")

        self.assertTrue(result)
        mock_send_voice.assert_awaited_once_with("447700900000", b"ogg-bytes", "audio/ogg; codecs=opus", "t1")
        mock_render_text.assert_not_awaited()

    async def test_falls_back_to_text_when_synthesis_fails(self) -> None:
        with patch("app.services.message_handler.synthesize_voice_note", new=AsyncMock(return_value=None)), \
             patch("app.services.message_handler.send_voice_message", new=AsyncMock()) as mock_send_voice, \
             patch("app.services.message_handler.render_and_send", new=AsyncMock(return_value=True)) as mock_render_text:
            result = await send_voice_reply("Your balance is 100", "447700900000", "t2")

        self.assertTrue(result)
        mock_send_voice.assert_not_awaited()
        mock_render_text.assert_awaited_once_with("Your balance is 100", "447700900000", "t2")

    async def test_falls_back_to_text_when_voice_send_fails(self) -> None:
        with patch(
            "app.services.message_handler.synthesize_voice_note",
            new=AsyncMock(return_value=(b"ogg-bytes", "audio/ogg; codecs=opus")),
        ), \
             patch("app.services.message_handler.send_voice_message", new=AsyncMock(return_value=False)), \
             patch("app.services.message_handler.render_and_send", new=AsyncMock(return_value=True)) as mock_render_text:
            result = await send_voice_reply("Your balance is 100", "447700900000", "t3")

        self.assertTrue(result)
        mock_render_text.assert_awaited_once_with("Your balance is 100", "447700900000", "t3")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock

from app.services import whatsapp


class _FakeResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


class _TrackingAsyncClient:
    last_request = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        type(self).last_request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return _FakeResponse(status_code=200, text="ok")


class SendTextMessageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._client_patcher = mock.patch.object(whatsapp.httpx, "AsyncClient", _TrackingAsyncClient)
        self._client_patcher.start()
        self.addCleanup(self._client_patcher.stop)
        self._phone_id_patcher = mock.patch.object(whatsapp, "PHONE_NUMBER_ID", "12345")
        self._phone_id_patcher.start()
        self.addCleanup(self._phone_id_patcher.stop)
        self._token_patcher = mock.patch.object(whatsapp, "ACCESS_TOKEN", "test-token")
        self._token_patcher.start()
        self.addCleanup(self._token_patcher.stop)
        _TrackingAsyncClient.last_request = None

    async def test_send_text_message_uses_full_chat_id_for_numeric_phone(self):
        result = await whatsapp.send_text_message("919080745760", "Hello", "trace-1")

        self.assertTrue(result)
        self.assertIsNotNone(_TrackingAsyncClient.last_request)
        self.assertEqual(
            _TrackingAsyncClient.last_request["url"], "https://graph.facebook.com/v25.0/12345/messages"
        )
        self.assertEqual(_TrackingAsyncClient.last_request["json"]["to"], "919080745760")
        self.assertEqual(_TrackingAsyncClient.last_request["json"]["type"], "text")
        self.assertEqual(_TrackingAsyncClient.last_request["json"]["text"], {"body": "Hello"})

    async def test_send_button_message_builds_expected_payload(self):
        result = await whatsapp.send_button_message(
            "919080745760", "Ready to send this?",
            [{"id": "1", "title": "Yes, send it"}, {"id": "2", "title": "Edit amount"}],
            "trace-2",
        )

        self.assertTrue(result)
        sent = _TrackingAsyncClient.last_request["json"]
        self.assertEqual(sent["type"], "interactive")
        self.assertEqual(sent["interactive"]["type"], "button")
        self.assertEqual(sent["interactive"]["body"]["text"], "Ready to send this?")
        self.assertEqual(
            sent["interactive"]["action"]["buttons"],
            [
                {"type": "reply", "reply": {"id": "1", "title": "Yes, send it"}},
                {"type": "reply", "reply": {"id": "2", "title": "Edit amount"}},
            ],
        )

    async def test_send_list_message_builds_expected_payload(self):
        sections = [
            {
                "title": "Loan types",
                "rows": [
                    {"id": "1", "title": "Personal Loan", "description": "Everyday personal borrowing"},
                    {"id": "2", "title": "Home Loan", "description": ""},
                ],
            }
        ]
        result = await whatsapp.send_list_message(
            "919080745760", "Choose a loan type", "Choose", sections, "trace-3"
        )

        self.assertTrue(result)
        sent = _TrackingAsyncClient.last_request["json"]
        self.assertEqual(sent["type"], "interactive")
        self.assertEqual(sent["interactive"]["type"], "list")
        self.assertEqual(sent["interactive"]["action"]["button"], "Choose")
        rows = sent["interactive"]["action"]["sections"][0]["rows"]
        self.assertEqual(rows[0], {"id": "1", "title": "Personal Loan", "description": "Everyday personal borrowing"})
        # Empty description is omitted rather than sent as an empty string.
        self.assertEqual(rows[1], {"id": "2", "title": "Home Loan"})

    async def test_send_button_message_failure_returns_false(self):
        class _FailingClient(_TrackingAsyncClient):
            async def post(self, *args, **kwargs):
                await super().post(*args, **kwargs)
                return _FakeResponse(status_code=500, text="server error")

        with mock.patch.object(whatsapp.httpx, "AsyncClient", _FailingClient):
            result = await whatsapp.send_button_message(
                "919080745760", "Hi", [{"id": "1", "title": "Yes"}], "trace-4"
            )
        self.assertFalse(result)


class GetInteractiveReplyTests(unittest.TestCase):
    def test_extracts_button_reply(self):
        payload = {"type": "interactive", "interactive": {"type": "button_reply", "button_reply": {"id": "1", "title": "Yes, send it"}}}
        self.assertEqual(whatsapp.get_interactive_reply(payload), {"id": "1", "title": "Yes, send it"})

    def test_extracts_list_reply(self):
        payload = {
            "type": "interactive",
            "interactive": {"type": "list_reply", "list_reply": {"id": "2", "title": "Home Loan", "description": ""}},
        }
        self.assertEqual(whatsapp.get_interactive_reply(payload), {"id": "2", "title": "Home Loan"})

    def test_non_interactive_payload_returns_none(self):
        self.assertIsNone(whatsapp.get_interactive_reply({"type": "text", "text": {"body": "hi"}}))

    def test_missing_id_returns_none(self):
        payload = {"type": "interactive", "interactive": {"type": "button_reply", "button_reply": {"title": "Yes"}}}
        self.assertIsNone(whatsapp.get_interactive_reply(payload))


class DetectMessageTypeInteractiveTests(unittest.TestCase):
    def test_interactive_type_detected(self):
        self.assertEqual(whatsapp.detect_message_type({"type": "interactive"}), "interactive")


if __name__ == "__main__":
    unittest.main()

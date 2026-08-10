import asyncio

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


def test_send_text_message_uses_full_chat_id_for_numeric_phone(monkeypatch):
    monkeypatch.setattr(whatsapp.httpx, "AsyncClient", _TrackingAsyncClient)
    monkeypatch.setattr(whatsapp, "PHONE_NUMBER_ID", "12345")
    monkeypatch.setattr(whatsapp, "ACCESS_TOKEN", "test-token")
    _TrackingAsyncClient.last_request = None

    result = asyncio.run(whatsapp.send_text_message("919080745760", "Hello", "trace-1"))

    assert result is True
    assert _TrackingAsyncClient.last_request is not None
    assert _TrackingAsyncClient.last_request["url"] == "https://graph.facebook.com/v17.0/12345/messages"
    assert _TrackingAsyncClient.last_request["json"]["to"] == "919080745760"
    assert _TrackingAsyncClient.last_request["json"]["type"] == "text"
    assert _TrackingAsyncClient.last_request["json"]["text"] == {"body": "Hello"}

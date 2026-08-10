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
    monkeypatch.setattr(whatsapp, "OPENWA_URL", "http://localhost:2785")
    monkeypatch.setattr(whatsapp, "OPENWA_API_KEY", "test-key")
    monkeypatch.setattr(whatsapp, "OPENWA_SESSION_ID", "test-session")
    _TrackingAsyncClient.last_request = None

    result = asyncio.run(whatsapp.send_text_message("919080745760", "Hello", "trace-1"))

    assert result is True
    assert _TrackingAsyncClient.last_request is not None
    assert _TrackingAsyncClient.last_request["json"]["chatId"] == "919080745760@c.us"

"""
Compatibility wrapper that exposes a small subset of the old
`groq.Groq` client's surface while delegating to `sarvamai.SarvamAI`.

This keeps the rest of the codebase (and tests) working with minimal
changes by providing `chat.completions.create(...)` and
`audio.transcriptions.create(...)` call sites backed by the Sarvam
client.
"""
from types import SimpleNamespace
import io
import os
from sarvamai import SarvamAI


class _ResponseChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_ResponseChoice(content)]


class ChatCompletions:
    def __init__(self, client: SarvamAI):
        self._client = client

    def create(self, **kwargs):
        # Map the old Groq chat.completions.create(...) to
        # sarvam_client.chat.completions(...). Keep the same kwargs.
        resp = self._client.chat.completions(**kwargs)
        # Try to extract text content from common response shapes.
        # If response has dict-like structure, try the nested path;
        # otherwise, fall back to str(resp).
        content = ""
        try:
            # sarvamai's responses may be object-like; attempt attribute access
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                first = choices[0]
                # Support both attribute and dict access
                content = getattr(getattr(first, "message", first), "content", None) or (
                    first.get("message", {}).get("content") if isinstance(first, dict) else None
                )
            elif isinstance(resp, dict):
                content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                # Fallback to string representation
                content = str(resp)
        except Exception:
            content = str(resp)

        return _Response(content)


class AudioTranscriptions:
    def __init__(self, client: SarvamAI):
        self._client = client

    def create(self, *, model: str, file, response_format: str = "json"):
        # Accept a file-like object. Sarvam's speech_to_text.transcribe
        # expects a file-like or bytes; call it and map the response to an
        # object with `.text` and `.language` attributes like Groq used to.
        # Prepare a file-like object if bytes were passed.
        if isinstance(file, (bytes, bytearray)):
            file_obj = io.BytesIO(file)
        else:
            file_obj = file

        resp = self._client.speech_to_text.transcribe(file=file_obj, model=model, mode="transcribe")

        # Map common fields
        text = ""
        language = ""
        try:
            if hasattr(resp, "text"):
                text = getattr(resp, "text")
            elif isinstance(resp, dict):
                text = resp.get("text") or resp.get("transcript") or ""
            else:
                text = str(resp)

            # language detection field may be present in different names
            if isinstance(resp, dict):
                language = resp.get("language") or resp.get("detected_language") or ""
            else:
                language = getattr(resp, "language", "")
        except Exception:
            text = str(resp)

        return SimpleNamespace(text=text, language=language)


class Groq:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self._client = SarvamAI(api_subscription_key=api_key)
        self.chat = SimpleNamespace(completions=ChatCompletions(self._client))
        self.audio = SimpleNamespace(transcriptions=AudioTranscriptions(self._client))

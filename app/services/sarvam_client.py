"""Single shared `sarvamai.SarvamAI` client, authenticated with
SARVAM_API_KEY. Every LLM/STT/TTS call in this app goes through this
client — there is no other provider (Groq, Gemini) wired in anymore.

`chat.completions(...)` and `speech_to_text.transcribe(...)` already return
plain, directly-usable response objects (`.choices[0].message.content`,
`.transcript`, `.language_code`, ...), so callers use the client directly
rather than going through a wrapper.
"""
import os

from sarvamai import SarvamAI

_client: SarvamAI | None = None


def get_sarvam_client() -> SarvamAI:
    global _client
    if _client is None:
        _client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY", ""))
    return _client

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


def get_fast_model() -> str:
    """Default chat model for nearly every LLM call in this app — intent
    understanding, conversation, translation, extraction. Benchmarked at
    ~2-3x faster than sarvam-105b with no observed quality loss, and
    unlike sarvam-105b it doesn't burn its token budget on hidden
    reasoning_content before producing a reply (sarvam-105b returned
    empty content at max_tokens=800 on a realistic agent-style prompt in
    testing; sarvam-105b-conversations did not). sarvam-30b, the other
    natural "fast" candidate, is deprecated and no longer callable."""
    return os.getenv("SARVAM_MODEL_FAST", "sarvam-105b-conversations")


def get_reasoning_model() -> str:
    """Reserved for the small set of calls that genuinely need deeper
    reasoning than get_fast_model() reliably provides. Callers using this
    model need real max_tokens headroom (1500+) — see get_fast_model()'s
    docstring for why a tight budget silently returns empty content."""
    return os.getenv("SARVAM_MODEL_REASONING", "sarvam-105b")

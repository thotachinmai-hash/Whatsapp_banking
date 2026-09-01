"""Language detection and translation for multilingual replies.

Scoped to English plus the Indian languages Sarvam's speech stack actually
covers — app/services/transcription.py (STT, saaras:v3) and
app/services/tts.py (TTS, bulbul:v3) — rather than an arbitrary broader
list, so a customer who writes/speaks in a "supported" language gets both
a translated text reply AND a voice reply that's actually spoken in that
language. See app/services/tts.py's _LANGUAGE_TO_TTS_CODE for the BCP-47
mapping bulbul:v3 needs.

Deliberately rule-free — language identification from arbitrary free text
isn't something a keyword list can do reliably, unlike the rest of this
app's rule-first style (see app/conversation/intent/rules.py). Both calls
here fail safe: any error, empty response, or unparseable output falls
back to English/the original text rather than raising or guessing.
"""

import asyncio
import hashlib
import re
import time
from typing import Optional

import redis
from dotenv import load_dotenv
from sarvamai import SarvamAI
from app.services.sarvam_client import get_sarvam_client
from app.memory import redis_client

from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# translate_text() is called on nearly every non-English reply, but the
# vast majority of this app's responses are one of a small, fixed set of
# template strings (menus, error messages, confirmation prompts) — see
# app/conversation/responses/*. Translating "Please choose 1 to confirm
# or 2 to edit" into Hindi always produces the same output, so caching by
# the exact (text, language) pair is a pure win: it transparently skips
# the LLM call for repeated template text while still translating fresh
# for genuinely unique text (a balance/transaction reply with real
# numbers in it) on every occurrence, since that text won't match a
# previous cache key. A week-long TTL bounds Redis memory rather than
# caching forever.
_TRANSLATION_CACHE_TTL = 7 * 24 * 3600


def _get_client() -> SarvamAI:
    return get_sarvam_client()


# ISO 639-1 code -> display name, used both to validate detect_language()'s
# output and to name the target language in the translate_text() prompt.
# Exactly the languages Sarvam TTS (bulbul:v3) can speak, so every
# supported language works for both text and voice replies.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

DEFAULT_LANGUAGE = "en"

# Below this length, language detection is unreliable (a bare "yes", "1",
# or "ok" carries no real linguistic signal) and not worth an LLM call —
# callers should keep using whatever language was already established for
# this conversation instead of re-detecting from noise.
MIN_DETECTABLE_LENGTH = 4


def _model() -> str:
    from app.services.sarvam_client import get_fast_model

    return get_fast_model()


def should_attempt_detection(text: str) -> bool:
    """Gate that decides whether a message is worth spending an LLM call
    on to detect its language — every incoming text message would
    otherwise pay that latency/cost, even the overwhelmingly common
    all-English case.

    Pure-ASCII text is treated as English without calling the model: every
    non-English language in SUPPORTED_LANGUAGES is written in its own
    non-Latin script (Devanagari for Hindi/Marathi, Bengali, Gujarati,
    Kannada, Malayalam, Odia, Gurmukhi for Punjabi, Tamil, Telugu), so this
    is a free, safe filter for the common case.

    Known limitation: romanized/transliterated non-English text typed in
    plain ASCII (e.g. Hindi written as "mera balance kya hai" rather than
    in Devanagari) is not caught by this filter and will be treated as
    English. Catching that would require calling the model on every
    message regardless of script, which is the latency/cost tradeoff this
    gate exists to avoid.
    """
    stripped = (text or "").strip()
    if len(stripped) < MIN_DETECTABLE_LENGTH:
        return False
    if not any(char.isalpha() for char in stripped):
        return False
    return any(ord(char) > 127 for char in stripped)


async def detect_language(text: str, trace_id: str = "") -> str:
    """Return an ISO 639-1 code from SUPPORTED_LANGUAGES, defaulting to
    "en" for anything unclear, unsupported, or on any failure."""
    if not should_attempt_detection(text):
        return DEFAULT_LANGUAGE

    start = time.time()
    try:
        # sarvam-105b is a reasoning model — it spends tokens on
        # reasoning_content before the actual answer, so max_tokens needs
        # real headroom (the old Groq-era value of 5 here would cut the
        # response off mid-reasoning with content=None). reasoning_effort
        # ="low" keeps that overhead down.
        #
        # The Sarvam SDK call itself is synchronous — run it in a thread
        # (same pattern as app/services/tts.py and transcription.py)
        # instead of blocking the event loop for the round-trip.
        response = await asyncio.to_thread(
            _get_client().chat.completions,
            model=_model(),
            temperature=0,
            max_tokens=800,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identify the language of the user's message. Reply with "
                        "ONLY its ISO 639-1 two-letter code (en, hi, bn, gu, kn, "
                        "ml, mr, or, pa, ta, te) and nothing else. If you are not "
                        "sure, reply 'en'."
                    ),
                },
                {"role": "user", "content": text[:300]},
            ],
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        code = "".join(ch for ch in raw if ch.isalpha())[:2]
        duration = (time.time() - start) * 1000
        if code in SUPPORTED_LANGUAGES:
            logger.info(f"[{trace_id}] Language detected | code={code} | duration={duration:.2f}ms")
            return code
        logger.info(f"[{trace_id}] Language detection unsupported/unclear | raw={raw!r} | defaulting to en")
        return DEFAULT_LANGUAGE
    except Exception as e:
        logger.error(f"[{trace_id}] Language detection failed | error={e}")
        return DEFAULT_LANGUAGE


# Display name -> ISO code, including a couple of common alternate/native
# spellings, used only to detect an EXPLICIT meta-request to change
# language ("reply in Tamil", "switch to Hindi") — as opposed to a message
# merely WRITTEN in that language, which should never be classified as a
# change request by this (already-ASCII, by construction — see
# should_attempt_detection) path.
_LANGUAGE_NAME_TO_CODE = {name.lower(): code for code, name in SUPPORTED_LANGUAGES.items()}
_LANGUAGE_NAME_TO_CODE.update({
    "english": "en",
    "oriya": "or",
    "gujrati": "gu",
})

_LANGUAGE_CHANGE_RE = re.compile(
    r"\b(?:reply|respond|speak|talk|switch|change)\b"
    r"(?:\s+to\s+me)?(?:\s+back)?(?:\s+(?:in|to))?\s+([a-z]+)\b",
    re.I,
)


def detect_explicit_language_change(message: str) -> Optional[str]:
    """Return an ISO 639-1 code only when `message` is a META-request about
    which language to use ("reply in Tamil", "switch to Hindi", "speak
    English please") — never when it's simply written in another language
    (that's should_attempt_detection/detect_language's job, and only fires
    on non-ASCII text). Regex-based rather than an LLM call: this pattern
    covers the natural phrasings for this request cheaply, and stays
    consistent with this app's rule-first style (see
    app/conversation/intent/rules.py). Returns None if no supported
    language is named."""
    if not message or not message.strip():
        return None
    match = _LANGUAGE_CHANGE_RE.search(message)
    if not match:
        return None
    return _LANGUAGE_NAME_TO_CODE.get(match.group(1).strip().lower())


def _translation_cache_key(text: str, target_language: str, max_length: int | None) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    budget = max_length if max_length is not None else "full"
    return f"cache:translation:{target_language}:{budget}:{digest}"


def _truncate_to_budget(text: str, max_length: int) -> str:
    """Last-resort safety net when the model still overshoots a requested
    character budget — cut at the last whitespace within budget so a
    multi-byte script character never gets split mid-cluster, falling
    back to a hard cut only if there's no whitespace to use."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    return truncated[:last_space].rstrip() if last_space > 0 else truncated


async def translate_text(
    text: str, target_language: str, trace_id: str = "", max_length: int | None = None
) -> str:
    """Translate `text` (assumed English — everything this app generates
    is authored in English) into `target_language`. Returns the original
    text unchanged if target_language is English/unsupported, or on any
    translation failure — a failed translation must never block the
    customer from getting a reply.

    `max_length`, when given, is a hard character budget for the
    translation — used for button/list labels, which WhatsApp itself caps
    (see app/conversation/renderer.py's MAX_BUTTON_TITLE_LEN etc.).
    Confirmed live: a literal English->target-language translation of a
    short label routinely runs longer than the source ("Edit amount" ->
    23 characters in Telugu, over the 20-character button-title limit),
    which silently dropped the interactive buttons/list entirely in favor
    of the plain-numbered-text fallback (see renderer.py's
    _fits_button_limits/_fits_list_limits) — every single time, not just
    on rare long labels. Left None for body text, which has no such
    limit."""
    if not text or target_language not in SUPPORTED_LANGUAGES or target_language == DEFAULT_LANGUAGE:
        return text

    cache_key = _translation_cache_key(text, target_language, max_length)
    try:
        cached = redis_client.get(cache_key)
        if cached is not None:
            return cached
    except redis.RedisError as e:
        logger.error(f"[{trace_id}] Translation cache read failed | error={e}")

    language_name = SUPPORTED_LANGUAGES[target_language]
    start = time.time()
    try:
        if max_length is not None:
            instruction = (
                f"Translate the user's message into {language_name}. This is a short "
                "WhatsApp button/menu label, not a sentence — reply with a natural, "
                f"idiomatic {language_name} label of at most {max_length} characters. "
                "Prefer a shorter, more colloquial phrasing over a longer literal one if "
                "needed to fit; abbreviate rather than dropping the core meaning. Preserve "
                "any number/ID/emoji exactly. Reply with ONLY the translated label, "
                "nothing else."
            )
        else:
            instruction = (
                f"Translate the user's message into {language_name}. Preserve "
                "every number, currency amount, ID/reference code (e.g. "
                "CHQ-XXXXXXXX, TRF-XXXXXXXX), emoji, and line break exactly "
                "as given — translate only the surrounding words. Reply with "
                "ONLY the translated text, nothing else."
            )
        # Sync SDK call — run off-thread, same reasoning as detect_language().
        response = await asyncio.to_thread(
            _get_client().chat.completions,
            model=_model(),
            temperature=0,
            max_tokens=2000,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
        )
        translated = (response.choices[0].message.content or "").strip()
        duration = (time.time() - start) * 1000
        if not translated:
            return text
        if max_length is not None and len(translated) > max_length:
            logger.info(
                f"[{trace_id}] Translated label exceeded budget, truncating | "
                f"language={target_language} | budget={max_length} | length={len(translated)}"
            )
            translated = _truncate_to_budget(translated, max_length)
        logger.info(f"[{trace_id}] Response translated | language={target_language} | duration={duration:.2f}ms")
        try:
            redis_client.setex(cache_key, _TRANSLATION_CACHE_TTL, translated)
        except redis.RedisError as e:
            logger.error(f"[{trace_id}] Translation cache write failed | error={e}")
        return translated
    except Exception as e:
        logger.error(f"[{trace_id}] Translation failed | language={target_language} | error={e}")
        return text

"""Language detection and translation for multilingual replies.

Scoped to the languages Meta officially lists as supported for Llama 3.3
(the model this app already uses for every LLM-answered turn — see
GROQ_MODEL in .env) — reliability outside this set is not guaranteed by
the underlying model, so detection is constrained to it rather than
guessing at broader support this app can't actually back up.

Deliberately rule-free — language identification from arbitrary free text
isn't something a keyword list can do reliably, unlike the rest of this
app's rule-first style (see app/conversation/intent/rules.py). Both calls
here fail safe: any error, empty response, or unparseable output falls
back to English/the original text rather than raising or guessing.
"""

import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    return _client


# ISO 639-1 code -> display name, used both to validate detect_language()'s
# output and to name the target language in the translate_text() prompt.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "es": "Spanish",
    "th": "Thai",
}

DEFAULT_LANGUAGE = "en"

# Below this length, language detection is unreliable (a bare "yes", "1",
# or "ok" carries no real linguistic signal) and not worth an LLM call —
# callers should keep using whatever language was already established for
# this conversation instead of re-detecting from noise.
MIN_DETECTABLE_LENGTH = 4


def _model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def should_attempt_detection(text: str) -> bool:
    """Gate that decides whether a message is worth spending an LLM call
    on to detect its language — every incoming text message would
    otherwise pay that latency/cost, even the overwhelmingly common
    all-English case.

    Pure-ASCII text is treated as English without calling the model: every
    non-English language in SUPPORTED_LANGUAGES is written with at least
    one character outside plain ASCII (Devanagari for Hindi, Thai script,
    or an accented Latin letter for French/German/Italian/Portuguese/
    Spanish), so this is a free, safe filter for the common case.

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


def detect_language(text: str, trace_id: str = "") -> str:
    """Return an ISO 639-1 code from SUPPORTED_LANGUAGES, defaulting to
    "en" for anything unclear, unsupported, or on any failure."""
    if not should_attempt_detection(text):
        return DEFAULT_LANGUAGE

    start = time.time()
    try:
        response = _get_client().chat.completions.create(
            model=_model(),
            temperature=0,
            max_tokens=5,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identify the language of the user's message. Reply with "
                        "ONLY its ISO 639-1 two-letter code (e.g. en, hi, es, fr, "
                        "de, it, pt, th) and nothing else. If you are not sure, "
                        "reply 'en'."
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


# Display name -> ISO code, including a couple of common alternate names,
# used only to detect an EXPLICIT meta-request to change language ("reply
# in Spanish", "switch to Hindi") — as opposed to a message merely WRITTEN
# in that language, which should never be classified as a change request
# by this (already-ASCII, by construction — see should_attempt_detection)
# path.
_LANGUAGE_NAME_TO_CODE = {name.lower(): code for code, name in SUPPORTED_LANGUAGES.items()}
_LANGUAGE_NAME_TO_CODE.update({"english": "en", "deutsch": "de", "espanol": "es", "español": "es"})

_LANGUAGE_CHANGE_RE = re.compile(
    r"\b(?:reply|respond|speak|talk|switch|change)\b"
    r"(?:\s+to\s+me)?(?:\s+back)?(?:\s+(?:in|to))?\s+([a-z]+)\b",
    re.I,
)


def detect_explicit_language_change(message: str) -> Optional[str]:
    """Return an ISO 639-1 code only when `message` is a META-request about
    which language to use ("reply in Spanish", "switch to Hindi", "speak
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


def translate_text(text: str, target_language: str, trace_id: str = "") -> str:
    """Translate `text` (assumed English — everything this app generates
    is authored in English) into `target_language`. Returns the original
    text unchanged if target_language is English/unsupported, or on any
    translation failure — a failed translation must never block the
    customer from getting a reply."""
    if not text or target_language not in SUPPORTED_LANGUAGES or target_language == DEFAULT_LANGUAGE:
        return text

    language_name = SUPPORTED_LANGUAGES[target_language]
    start = time.time()
    try:
        response = _get_client().chat.completions.create(
            model=_model(),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate the user's message into {language_name}. Preserve "
                        "every number, currency amount, ID/reference code (e.g. "
                        "CHQ-XXXXXXXX, TRF-XXXXXXXX), emoji, and line break exactly "
                        "as given — translate only the surrounding words. Reply with "
                        "ONLY the translated text, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        translated = (response.choices[0].message.content or "").strip()
        duration = (time.time() - start) * 1000
        if not translated:
            return text
        logger.info(f"[{trace_id}] Response translated | language={target_language} | duration={duration:.2f}ms")
        return translated
    except Exception as e:
        logger.error(f"[{trace_id}] Translation failed | language={target_language} | error={e}")
        return text

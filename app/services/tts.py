"""Text-to-speech via Sarvam's TTS API (SARVAM_TTS in .env, e.g.
bulbul:v3) — the text-out half of voice-to-voice. Voice-in already existed
(app/services/transcription.py, Sarvam speech-to-text).
"""

import asyncio
import base64
import os
import tempfile
import time

from dotenv import load_dotenv

from app.logger import get_logger
from app.services.sarvam_client import get_sarvam_client

load_dotenv()
logger = get_logger(__name__)

SARVAM_TTS = os.getenv("SARVAM_TTS", "bulbul:v3")

# The default speaker for bulbul:v3 per Sarvam's docs.
DEFAULT_VOICE = "shubh"

DEFAULT_LANGUAGE_CODE = "en-IN"

# ISO 639-1 (app.services.language.SUPPORTED_LANGUAGES) -> Sarvam TTS's
# BCP-47 language_code. Sarvam's TTS (bulbul) only covers English + Indian
# languages, unlike the broader set app.services.language translates text
# into — codes with no TTS equivalent fall back to DEFAULT_LANGUAGE_CODE
# (the reply is still spoken, just not in that language) rather than
# failing synthesis outright.
_LANGUAGE_TO_TTS_CODE = {
    "en": "en-IN",
    "hi": "hi-IN",
}

# Simple in-memory circuit breaker for Sarvam TTS quota/rate-limit
# exhaustion. Once a 429 is seen, further requests would just fail the
# same way until the window resets — retrying each one anyway means every
# voice reply pays a slow, doomed HTTP round trip and logs a fresh ERROR.
# Instead, stop calling out entirely for a cooldown window and fall back
# to text immediately, quietly, until it's worth trying again.
_QUOTA_COOLDOWN_SECONDS = 300.0
_quota_exhausted_until = 0.0


async def synthesize_speech(
    text: str,
    trace_id: str = "",
    voice: str = DEFAULT_VOICE,
    language: str | None = None,
) -> bytes | None:
    """Synthesize `text` to speech, returning WAV bytes, or None on any
    failure (missing key, network error, unexpected response shape) — a
    TTS failure must never block the customer from getting a reply; the
    caller falls back to sending text instead.

    `language` is an optional ISO 639-1 code (e.g. from
    app.services.language / voice transcription) used to pick the closest
    Sarvam TTS voice language; defaults to English when not given or not
    covered by Sarvam TTS.
    """
    if not os.getenv("SARVAM_API_KEY"):
        logger.warning(f"[{trace_id}] TTS skipped | SARVAM_API_KEY not configured")
        return None
    if not text or not text.strip():
        return None

    global _quota_exhausted_until
    if time.time() < _quota_exhausted_until:
        logger.info(f"[{trace_id}] TTS skipped | quota cooldown active, falling back to text")
        return None

    language_code = _LANGUAGE_TO_TTS_CODE.get(language or "", DEFAULT_LANGUAGE_CODE)
    start = time.time()
    try:
        response = await asyncio.to_thread(
            get_sarvam_client().text_to_speech.convert,
            text=text,
            language_code=language_code,
            speaker=voice,
            model=SARVAM_TTS,
            output_audio_codec="wav",
        )

        if not response.audios:
            logger.error(f"[{trace_id}] TTS response had no audio data")
            return None

        wav_bytes = base64.b64decode(response.audios[0])

        duration = (time.time() - start) * 1000
        logger.info(f"[{trace_id}] TTS synthesis complete | bytes={len(wav_bytes)} | duration={duration:.2f}ms")
        return wav_bytes

    except Exception as e:
        duration = (time.time() - start) * 1000
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            _quota_exhausted_until = time.time() + _QUOTA_COOLDOWN_SECONDS
            logger.warning(
                f"[{trace_id}] TTS quota exceeded — pausing voice replies for "
                f"{_QUOTA_COOLDOWN_SECONDS:.0f}s | error={e}"
            )
            return None
        logger.error(f"[{trace_id}] TTS synthesis failed | error={e} | duration={duration:.2f}ms")
        return None


async def _wav_to_ogg_opus(wav_bytes: bytes, trace_id: str = "") -> bytes | None:
    """Convert WAV to OGG/Opus via the ffmpeg binary already installed in
    this image (see Dockerfile) — no new Python dependency. The WhatsApp
    gateway's own send-audio docs specifically call out "audio/ogg;
    codecs=opus" as the format needed for a voice note (ptt) to play
    reliably; a bare WAV is not guaranteed to. Returns None on any
    failure so the caller can fall back to sending the WAV as a regular
    audio file instead of a voice note bubble."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
        wav_file.write(wav_bytes)
    ogg_path = wav_path.replace(".wav", ".ogg")

    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", ogg_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(f"[{trace_id}] ffmpeg WAV->OGG conversion failed | stderr={stderr.decode(errors='replace')[:300]}")
            return None
        with open(ogg_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[{trace_id}] ffmpeg WAV->OGG conversion error | error={e}")
        return None
    finally:
        for path in (wav_path, ogg_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


async def synthesize_voice_note(
    text: str,
    trace_id: str = "",
    voice: str = DEFAULT_VOICE,
    language: str | None = None,
) -> tuple[bytes, str] | None:
    """Synthesize `text` and return (audio_bytes, mimetype) ready to send
    as a WhatsApp voice note. Prefers OGG/Opus (what the gateway's
    send-audio API documents as reliable for `ptt`); falls back to the
    raw WAV (sent as a plain audio file, not a voice note bubble) if the
    ffmpeg conversion step fails. Returns None only if synthesis itself
    failed — see synthesize_speech()."""
    wav_bytes = await synthesize_speech(text, trace_id=trace_id, voice=voice, language=language)
    if wav_bytes is None:
        return None

    ogg_bytes = await _wav_to_ogg_opus(wav_bytes, trace_id=trace_id)
    if ogg_bytes is not None:
        return ogg_bytes, "audio/ogg; codecs=opus"

    logger.warning(f"[{trace_id}] Falling back to WAV audio (not a voice-note bubble) — OGG conversion unavailable")
    return wav_bytes, "audio/wav"

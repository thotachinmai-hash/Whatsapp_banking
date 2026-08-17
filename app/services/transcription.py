import os
import time
import tempfile
import httpx
from app.services.sarvam_client import get_sarvam_client
from dotenv import load_dotenv
from app.logger import get_logger
from app.services.language import MIN_DETECTABLE_LENGTH

load_dotenv()
logger = get_logger(__name__)

STT_MODEL = os.getenv("SARVAM_STT", "saaras:v3")


async def download_audio(media_url: str, api_key: str, trace_id: str) -> bytes | None:
    """Download audio file from OpenWA media URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                media_url,
                headers={"X-API-Key": api_key},
                timeout=30.0
            )
            if response.status_code == 200:
                logger.info(f"[{trace_id}] Audio downloaded | size={len(response.content)} bytes")
                return response.content
            else:
                logger.error(f"[{trace_id}] Audio download failed | status={response.status_code}")
                return None
    except Exception as e:
        logger.error(f"[{trace_id}] Audio download error | error={e}")
        return None


async def transcribe_audio(audio_data: bytes, trace_id: str) -> tuple[str | None, str | None]:
    """
    Transcribe audio using Sarvam's speech-to-text (SARVAM_STT in .env,
    e.g. saaras:v3).

    Returns (text, language_code) — language_code is an ISO 639-1 code
    from app.services.language.SUPPORTED_LANGUAGES if Sarvam's own
    detection matched one of them, else None (caller falls back to
    text-based detection on the transcript, same as a typed message).
    On failure, returns (None, None).
    """
    start = time.time()
    temp_path = None
    try:
        # Use the platform temp directory so this works on Windows and Linux.
        with tempfile.NamedTemporaryFile(
            prefix=f"audio_{trace_id}_", suffix=".ogg", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)

        with open(temp_path, "rb") as audio_file:
            transcription = get_sarvam_client().speech_to_text.transcribe(
                file=audio_file,
                model=STT_MODEL,
                mode="transcribe",
            )

        duration = (time.time() - start) * 1000
        text = (transcription.transcript or "").strip()
        # Sarvam reports the detected language as a BCP-47 tag (e.g.
        # "hi-IN", "en-IN") — take the leading subtag as the ISO 639-1 code
        # app.services.language.SUPPORTED_LANGUAGES uses everywhere else,
        # so a voice message and a typed message in the same language are
        # treated identically downstream. Odia is the one language where
        # Sarvam's tag ("od") doesn't match its real ISO 639-1 code ("or").
        raw_language = (transcription.language_code or "").strip().lower()
        language_code = raw_language.split("-")[0] or None
        if language_code == "od":
            language_code = "or"
        # Sarvam's own language guess is unreliable on a handful of words
        # ("Ok.", "Yes") — too little audio to carry real linguistic signal,
        # the same reason app/services/language.py won't run text-based
        # detection below MIN_DETECTABLE_LENGTH. Trusting it anyway risks a
        # one-word reply getting mis-tagged and then sticking for the rest
        # of the conversation. Drop the tag here so the caller falls back
        # to text-based detection on the transcript instead, which applies
        # that same length gate.
        if language_code and len(text) < MIN_DETECTABLE_LENGTH:
            logger.info(
                f"[{trace_id}] STT language tag ignored | reason=transcript_too_short | "
                f"language={raw_language} | text={text!r}"
            )
            language_code = None
        logger.info(
            f"[{trace_id}] Transcription complete | duration={duration:.2f}ms | "
            f"language={raw_language or 'unknown'} | text={text[:50]}"
        )
        return text or None, language_code

    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(f"[{trace_id}] Transcription failed | error={e} | duration={duration:.2f}ms")
        return None, None
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

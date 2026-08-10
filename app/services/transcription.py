import os
import time
import tempfile
import httpx
from groq import Groq
from dotenv import load_dotenv
from app.logger import get_logger
from app.services.language import MIN_DETECTABLE_LENGTH

load_dotenv()
logger = get_logger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# Whisper's verbose_json response names the detected language in full
# (lowercase English name), not as an ISO code — map onto the same
# SUPPORTED_LANGUAGES codes app/services/language.py uses everywhere else,
# so a voice message and a typed message in the same language are treated
# identically downstream.
_WHISPER_LANGUAGE_TO_CODE = {
    "english": "en", "german": "de", "french": "fr", "italian": "it",
    "portuguese": "pt", "hindi": "hi", "spanish": "es", "thai": "th",
}


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
    Transcribe audio using Groq whisper-large-v3-turbo.

    Returns (text, language_code) — language_code is an ISO 639-1 code
    from app.services.language.SUPPORTED_LANGUAGES if Whisper's own
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

        # verbose_json (rather than plain "text") also reports Whisper's
        # own detected spoken language, so voice messages don't need a
        # second, separate detection call the way typed text does.
        with open(temp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json"
            )

        duration = (time.time() - start) * 1000
        text = str(getattr(transcription, "text", "")).strip()
        language_name = str(getattr(transcription, "language", "")).strip().lower()
        language_code = _WHISPER_LANGUAGE_TO_CODE.get(language_name)
        # Whisper's own language guess is unreliable on a handful of words
        # ("Ok.", "Yes") — too little audio to carry real linguistic signal,
        # the same reason app/services/language.py won't run text-based
        # detection below MIN_DETECTABLE_LENGTH. Trusting it anyway is what
        # let a one-word "Ok." get mis-tagged Portuguese and then stick for
        # the rest of the conversation. Drop the tag here so the caller
        # falls back to text-based detection on the transcript instead,
        # which applies that same length gate.
        if language_code and len(text) < MIN_DETECTABLE_LENGTH:
            logger.info(
                f"[{trace_id}] Whisper language tag ignored | reason=transcript_too_short | "
                f"language={language_name} | text={text!r}"
            )
            language_code = None
        logger.info(
            f"[{trace_id}] Transcription complete | duration={duration:.2f}ms | "
            f"language={language_name or 'unknown'} | text={text[:50]}"
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

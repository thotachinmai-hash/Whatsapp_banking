import os
import time
import httpx
from groq import Groq
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")


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


async def transcribe_audio(audio_data: bytes, trace_id: str) -> str | None:
    """
    Transcribe audio using Groq whisper-large-v3-turbo.
    Returns transcribed text or None if failed.
    """
    start = time.time()
    try:
        # Write audio to temp file
        temp_path = f"/tmp/audio_{trace_id}.ogg"
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        # Transcribe with Groq Whisper
        with open(temp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="text"
            )

        duration = (time.time() - start) * 1000
        text = str(transcription).strip()
        logger.info(f"[{trace_id}] Transcription complete | duration={duration:.2f}ms | text={text[:50]}")

        # Cleanup temp file
        os.remove(temp_path)

        return text

    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(f"[{trace_id}] Transcription failed | error={e} | duration={duration:.2f}ms")
        return None

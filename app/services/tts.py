"""Text-to-speech via Gemini's TTS model (GEMINI_TTS_MODEL in .env) — the
text-out half of voice-to-voice. Voice-in already existed
(app/services/transcription.py, Groq Whisper); this is the previously
unused half, using the GEMINI_API_KEY that was already present in .env.

Called via plain REST (httpx, already a dependency) rather than adding
the google-generativeai/google-genai SDK — matches this repo's minimal-
dependency style (see app/rag/retriever.py's equivalent choice for RAG).
"""

import asyncio
import base64
import os
import struct
import tempfile
import time

import httpx
from dotenv import load_dotenv

from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# One prebuilt voice per supported language — Gemini's TTS voices are not
# themselves language-specific (a given voice can speak any language the
# text is written in), so this only needs to vary if a future need for
# distinct-sounding voices per language arises. A single well-reviewed
# voice is used for all languages for now.
DEFAULT_VOICE = "Kore"

# Gemini's inline audio data is raw 16-bit PCM at this rate/channel count
# (per its documented TTS output format), with no container — a WAV
# header has to be added before any standard player (or WhatsApp) can
# play it. See _pcm_to_wav().
_PCM_SAMPLE_RATE = 24000
_PCM_CHANNELS = 1
_PCM_BITS_PER_SAMPLE = 16

# Simple in-memory circuit breaker for Gemini TTS quota exhaustion. Once a
# 429 quota error is seen, further requests would just fail the same way
# until the quota window resets — retrying each one anyway means every
# voice reply pays a slow, doomed HTTP round trip and logs a fresh ERROR.
# Instead, stop calling out entirely for a cooldown window and fall back
# to text immediately, quietly, until it's worth trying again.
_QUOTA_COOLDOWN_SECONDS = 300.0
_quota_exhausted_until = 0.0


def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw 16-bit PCM in a minimal WAV header (stdlib-only, no new
    dependency) so the audio is a self-contained, playable file."""
    byte_rate = _PCM_SAMPLE_RATE * _PCM_CHANNELS * _PCM_BITS_PER_SAMPLE // 8
    block_align = _PCM_CHANNELS * _PCM_BITS_PER_SAMPLE // 8
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, _PCM_CHANNELS, _PCM_SAMPLE_RATE, byte_rate, block_align, _PCM_BITS_PER_SAMPLE,
        b"data", data_size,
    )
    return header + pcm_bytes


async def synthesize_speech(text: str, trace_id: str = "", voice: str = DEFAULT_VOICE) -> bytes | None:
    """Synthesize `text` to speech, returning WAV bytes, or None on any
    failure (missing key, network error, unexpected response shape) — a
    TTS failure must never block the customer from getting a reply; the
    caller falls back to sending text instead.
    """
    if not GEMINI_API_KEY:
        logger.warning(f"[{trace_id}] TTS skipped | GEMINI_API_KEY not configured")
        return None
    if not text or not text.strip():
        return None

    global _quota_exhausted_until
    if time.time() < _quota_exhausted_until:
        logger.info(f"[{trace_id}] TTS skipped | quota cooldown active, falling back to text")
        return None

    start = time.time()
    url = f"{GEMINI_BASE_URL}/models/{GEMINI_TTS_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30.0)

        if response.status_code == 429:
            _quota_exhausted_until = time.time() + _QUOTA_COOLDOWN_SECONDS
            logger.warning(
                f"[{trace_id}] TTS quota exceeded — pausing voice replies for "
                f"{_QUOTA_COOLDOWN_SECONDS:.0f}s | body={response.text[:200]}"
            )
            return None

        if response.status_code != 200:
            logger.error(f"[{trace_id}] TTS request failed | status={response.status_code} | body={response.text[:200]}")
            return None

        data = response.json()
        parts = data["candidates"][0]["content"]["parts"]
        inline_data = next((p["inlineData"] for p in parts if "inlineData" in p), None)
        if not inline_data or not inline_data.get("data"):
            logger.error(f"[{trace_id}] TTS response had no audio data")
            return None

        pcm_bytes = base64.b64decode(inline_data["data"])
        wav_bytes = _pcm_to_wav(pcm_bytes)

        duration = (time.time() - start) * 1000
        logger.info(f"[{trace_id}] TTS synthesis complete | bytes={len(wav_bytes)} | duration={duration:.2f}ms")
        return wav_bytes

    except Exception as e:
        duration = (time.time() - start) * 1000
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


async def synthesize_voice_note(text: str, trace_id: str = "", voice: str = DEFAULT_VOICE) -> tuple[bytes, str] | None:
    """Synthesize `text` and return (audio_bytes, mimetype) ready to send
    as a WhatsApp voice note. Prefers OGG/Opus (what the gateway's
    send-audio API documents as reliable for `ptt`); falls back to the
    raw WAV (sent as a plain audio file, not a voice note bubble) if the
    ffmpeg conversion step fails. Returns None only if synthesis itself
    failed — see synthesize_speech()."""
    wav_bytes = await synthesize_speech(text, trace_id=trace_id, voice=voice)
    if wav_bytes is None:
        return None

    ogg_bytes = await _wav_to_ogg_opus(wav_bytes, trace_id=trace_id)
    if ogg_bytes is not None:
        return ogg_bytes, "audio/ogg; codecs=opus"

    logger.warning(f"[{trace_id}] Falling back to WAV audio (not a voice-note bubble) — OGG conversion unavailable")
    return wav_bytes, "audio/wav"

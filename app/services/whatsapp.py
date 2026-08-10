import base64
import os
import re
import httpx
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

OPENWA_URL = os.getenv("OPENWA_URL", "http://localhost:2785")
OPENWA_API_KEY = os.getenv("OPENWA_API_KEY", "")
OPENWA_SESSION_ID = os.getenv("OPENWA_SESSION_ID")


def normalize_chat_id(phone_number: str) -> str:
    """Convert a phone number or existing chat identifier into an OpenWA chat ID."""
    value = (phone_number or "").strip()
    if not value:
        return value

    if "@" in value:
        return value

    digits_only = re.sub(r"\D", "", value)
    if digits_only:
        return f"{digits_only}@c.us"

    return value


async def send_text_message(phone_number: str, message: str, trace_id: str) -> bool:
    """
    Send text message back to WhatsApp user via OpenWA API.
    Accepts either a plain phone number or an existing chat ID.
    """
    chat_id = normalize_chat_id(phone_number)
    url = f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION_ID}/messages/send-text"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "X-API-Key": OPENWA_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "chatId": chat_id,
                    "text": message
                },
                timeout=15.0
            )

            if response.status_code in [200, 201]:
                logger.info(f"[{trace_id}] WhatsApp message sent | phone={phone_number[-4:]}")
                return True
            else:
                logger.error(f"[{trace_id}] WhatsApp send failed | status={response.status_code} | body={response.text[:100]}")
                return False

    except Exception as e:
        logger.error(f"[{trace_id}] WhatsApp send error | error={e}")
        return False

async def send_voice_message(phone_number: str, audio_bytes: bytes, mimetype: str, trace_id: str) -> bool:
    """
    Send a voice note back to WhatsApp user via OpenWA's real send-audio
    endpoint — confirmed against the running gateway's own OpenAPI spec
    (GET /api/docs-json on the OpenWA container): POST .../messages/
    send-audio with {chatId, base64, mimetype, ptt: true}. `base64` is
    the raw base64 payload (no "data:...;base64," prefix — mimetype is
    its own field). `ptt: true` renders it as a proper voice-note bubble;
    the gateway's own docs say this needs "audio/ogg; codecs=opus" bytes
    to play reliably — see app/services/tts.py::synthesize_voice_note for
    where that conversion happens.
    """
    chat_id = normalize_chat_id(phone_number)
    url = f"{OPENWA_URL}/api/sessions/{OPENWA_SESSION_ID}/messages/send-audio"
    encoded = base64.b64encode(audio_bytes).decode("ascii")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "X-API-Key": OPENWA_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "chatId": chat_id,
                    "base64": encoded,
                    "mimetype": mimetype,
                    "ptt": True,
                },
                timeout=30.0
            )

            if response.status_code in [200, 201]:
                logger.info(f"[{trace_id}] WhatsApp voice message sent | phone={phone_number[-4:]}")
                return True
            else:
                logger.error(f"[{trace_id}] WhatsApp voice send failed | status={response.status_code} | body={response.text[:200]}")
                return False

    except Exception as e:
        logger.error(f"[{trace_id}] WhatsApp voice send error | error={e}")
        return False


async def get_sender_phone(contact_id: str):
    url = (
        f"{OPENWA_URL}/api/sessions/"
        f"{OPENWA_SESSION_ID}/contacts/"
        f"{contact_id}/phone"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={
                    "X-API-Key": OPENWA_API_KEY
                },
                timeout=10.0,
            )

        logger.info(f"Phone lookup status: {response.status_code}")
        logger.info(f"Phone lookup body: {response.text}")

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict):
                for key in ("phone", "phoneNumber", "number"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
                if "data" in data and isinstance(data["data"], dict):
                    for key in ("phone", "phoneNumber", "number"):
                        value = data["data"].get(key)
                        if isinstance(value, str) and value.strip():
                            return value
            if isinstance(data, str) and data.strip():
                return data

        return None

    except Exception as e:
        logger.error(f"Phone lookup failed: {e}")
        return None

def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from WhatsApp chat ID format: 447812345678@c.us"""
    return chat_id.replace("@c.us", "").replace("@g.us", "").replace("@lid", "")


def detect_message_type(payload: dict) -> str:
    """
    Detect message type from OpenWA webhook payload.
    Returns: 'voice', 'text', or 'unsupported'
    """
    msg_type = str(payload.get("type") or payload.get("messageType") or payload.get("kind") or "").lower()
    media = payload.get("media", {}) or {}
    media_mime_type = str(
        media.get("mimetype") or media.get("mimeType")
        or payload.get("mimeType") or payload.get("mimetype") or ""
    ).lower()
    if msg_type in ["audio", "voice", "ptt"] or media_mime_type.startswith("audio/"):
        return "voice"
    elif msg_type == "text":
        return "text"

    elif msg_type in [
        "image",
        "document",
        "file",
        "pdf",
        "application"
    ]:
        return "document"

    # Some OpenWA versions send only mimeType
    mime_type = media_mime_type

    if mime_type.startswith("image/"):
        return "document"

    if mime_type in [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        return "document"

    return "unsupported"


def get_message_text(payload: dict) -> str:
    """Extract text content from webhook payload."""
    return payload.get("body", "") or payload.get("text", "") or ""


def get_media_url(payload: dict) -> str:
    """
    Extract media URL for voice/document/image files.
    """

    media = payload.get("media", {}) or {}

    return (
        media.get("url")
        or media.get("mediaUrl")
        or payload.get("mediaUrl")
        or payload.get("url")
        or ""
    )

def get_media_filename(payload: dict) -> str:
    """
    Extract uploaded filename.
    """

    media = payload.get("media", {}) or {}

    return (
        media.get("fileName")
        or media.get("filename")
        or payload.get("fileName")
        or payload.get("filename")
        or "uploaded_file"
    )

def get_media_mimetype(payload: dict) -> str:
    """
    Extract MIME type.
    """

    media = payload.get("media", {}) or {}

    return (
        media.get("mimetype")
        or media.get("mimeType")
        or payload.get("mimetype")
        or payload.get("mimeType")
        or ""
    )

def get_external_message_id(payload: dict) -> str:
    """
    Extract OpenWA's stable identifier for this message event, for
    inbound idempotency (see app/services/idempotency.py).

    OpenWA/WPPConnect message.received payloads typically carry the
    WhatsApp message id at data.id as a string (e.g.
    "true_447xxx@c.us_3EB0..."), but some builds nest it as
    data.id._serialized / data.id.id, or use "messageId"/"msgId"/
    "stanzaId" instead. Tries each in order.

    Deliberately never falls back to hashing the message body/media —
    two messages a user intentionally sends with identical text must
    keep producing two separate ids, only true webhook-retry duplicates
    share an id.
    """
    raw_id = payload.get("id")
    if isinstance(raw_id, dict):
        for key in ("_serialized", "id"):
            value = raw_id.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()

    for key in ("messageId", "msgId", "stanzaId", "message_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_media_data(payload: dict) -> str:
    """
    Extract Base64 media data from the OpenWA webhook payload.
    """

    media = payload.get("media", {}) or {}

    embedded_data = (
        media.get("data")
        or payload.get("mediaData")
        or payload.get("base64")
        or (payload.get("data") if isinstance(payload.get("data"), str) else "")
        or ""
    )
    return str(embedded_data)

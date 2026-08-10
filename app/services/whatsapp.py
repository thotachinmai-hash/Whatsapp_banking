import os
import re
import httpx
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v17.0")


def normalize_chat_id(phone_number: str) -> str:
    """Normalise a phone number by removing non-digit characters."""
    value = (phone_number or "").strip()
    if not value:
        return value

    digits_only = re.sub(r"\D", "", value)
    return digits_only


async def send_text_message(phone_number: str, message: str, trace_id: str) -> bool:
    """
    Send a text message back to WhatsApp user via WhatsApp Business Cloud API.
    Accepts a phone number in any common format and normalizes digits only.
    """
    to_phone = normalize_chat_id(phone_number)
    if not to_phone:
        logger.error(f"[{trace_id}] WhatsApp send failed | invalid recipient phone")
        return False

    if not PHONE_NUMBER_ID or not ACCESS_TOKEN:
        logger.error(
            f"[{trace_id}] WhatsApp send failed | missing PHONE_NUMBER_ID or ACCESS_TOKEN"
        )
        return False

    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{PHONE_NUMBER_ID}/messages"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {ACCESS_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": to_phone,
                    "type": "text",
                    "text": {"body": message}
                },
                timeout=15.0
            )

            if response.status_code in [200, 201]:
                logger.info(f"[{trace_id}] WhatsApp message sent | phone={to_phone[-4:]}")
                return True

            logger.error(
                f"[{trace_id}] WhatsApp send failed | status={response.status_code} | body={response.text[:200]}"
            )
            return False

    except Exception as e:
        logger.error(f"[{trace_id}] WhatsApp send error | error={e}")
        return False


async def send_voice_message(phone_number: str, audio_bytes: bytes, mimetype: str, trace_id: str) -> bool:
    """
    Voice note delivery is not currently implemented for the WhatsApp Cloud API path.
    The caller will fall back to text delivery if this returns False.
    """
    logger.warning(
        f"[{trace_id}] WhatsApp voice send is not supported by current Cloud API implementation"
    )
    return False


async def get_sender_phone(contact_id: str):
    logger.warning(
        "get_sender_phone() called, but WhatsApp Cloud API does not support OpenWA contact lookup"
    )
    return None


def get_media_id(payload: dict) -> str:
    """Extract the WhatsApp Cloud API media ID from a message payload."""
    msg_type = str(payload.get("type") or "").lower()
    if not msg_type:
        return ""

    media_payload = payload.get(msg_type, {}) or {}
    if isinstance(media_payload, dict):
        return media_payload.get("id") or media_payload.get("media_id") or ""

    return ""


async def download_media(media_id: str, trace_id: str) -> bytes | None:
    """Download media bytes from WhatsApp Business Cloud API given a media ID."""
    if not media_id:
        logger.error(f"[{trace_id}] download_media failed | missing media_id")
        return None

    if not ACCESS_TOKEN:
        logger.error(f"[{trace_id}] download_media failed | missing ACCESS_TOKEN")
        return None

    media_info_url = (
        f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{media_id}?fields=url"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                media_info_url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                timeout=15.0
            )

            if response.status_code != 200:
                logger.error(
                    f"[{trace_id}] Media metadata request failed | status={response.status_code} | body={response.text[:200]}"
                )
                return None

            media_url = response.json().get("url")
            if not media_url:
                logger.error(
                    f"[{trace_id}] Media metadata response missing url | body={response.text[:200]}"
                )
                return None

            response = await client.get(
                media_url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(
                    f"[{trace_id}] Media download failed | status={response.status_code} | body={response.text[:200]}"
                )
                return None

            logger.info(f"[{trace_id}] Media downloaded | size={len(response.content)} bytes")
            return response.content

    except Exception as e:
        logger.error(f"[{trace_id}] Media download error | error={e}")
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

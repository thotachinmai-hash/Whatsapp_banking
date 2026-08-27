import os
import httpx
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

OPENWA_URL = os.getenv("OPENWA_URL", "http://localhost:2785")
OPENWA_API_KEY = os.getenv("OPENWA_API_KEY", "")
OPENWA_SESSION_ID = os.getenv("OPENWA_SESSION_ID")


async def send_text_message(phone_number: str, message: str, trace_id: str) -> bool:
    """
    Send text message back to WhatsApp user via OpenWA API.
    phone_number format: 441111111111 (without @c.us)
    """
    chat_id = phone_number
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
            return response.json()

        return None

    except Exception as e:
        logger.error(f"Phone lookup failed: {e}")
        return None

def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from WhatsApp chat ID format: 441111111111@c.us"""
    return chat_id.replace("@c.us", "").replace("@g.us", "").replace("@lid", "")


def detect_message_type(payload: dict) -> str:
    """
    Detect message type from OpenWA webhook payload.
    Returns: 'voice', 'text', or 'unsupported'
    """
    msg_type = payload.get("type", "")
    if msg_type in ["audio", "ptt"]:  # ptt = push to talk (voice note)
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
    mime_type = (
        payload.get("mimeType")
        or payload.get("mimetype")
        or ""
    )

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

def get_media_data(payload: dict) -> str:
    """
    Extract Base64 media data from the OpenWA webhook payload.
    """

    media = payload.get("media", {}) or {}

    return (
        media.get("data")
        or payload.get("data")
        or ""
    )

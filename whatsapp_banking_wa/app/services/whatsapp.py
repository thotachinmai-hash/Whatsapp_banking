import os
import httpx
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v25.0")
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


async def send_text_message(phone_number: str, message: str, trace_id: str) -> bool:
    url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"

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
                    "to": phone_number,
                    "type": "text",
                    "text": {"body": message}
                },
                timeout=15.0
            )

        if response.status_code in [200, 201]:
            logger.info(f"[{trace_id}] WhatsApp message sent | phone={phone_number[-4:]}")
            return True

        logger.error(
            f"[{trace_id}] WhatsApp send failed | status={response.status_code} | body={response.text[:200]}"
        )
        return False

    except Exception as e:
        logger.error(f"[{trace_id}] WhatsApp send error | error={e}")
        return False


async def get_media_url(media_id: str, trace_id: str) -> str | None:
    url = f"{GRAPH_API_BASE}/{media_id}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                params={"fields": "url"},
                timeout=15.0
            )

        if response.status_code != 200:
            logger.error(
                f"[{trace_id}] Media URL lookup failed | status={response.status_code} | body={response.text[:200]}"
            )
            return None

        return response.json().get("url")

    except Exception as e:
        logger.error(f"[{trace_id}] Media URL lookup failed | error={e}")
        return None


async def download_whatsapp_media(media_id: str, trace_id: str) -> tuple[bytes | None, str]:
    media_url = await get_media_url(media_id, trace_id)
    if not media_url:
        return None, ""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                media_url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                timeout=60.0
            )

        if response.status_code != 200:
            logger.error(
                f"[{trace_id}] Media download failed | status={response.status_code}"
            )
            return None, ""

        content_type = response.headers.get("content-type", "application/octet-stream")
        logger.info(
            f"[{trace_id}] Media downloaded | size={len(response.content)} bytes | mime_type={content_type}"
        )
        return response.content, content_type

    except Exception as e:
        logger.error(f"[{trace_id}] Media download error | error={e}")
        return None, ""


def extract_phone_number(chat_id: str) -> str:
    if not chat_id:
        return ""
    return chat_id.replace("@c.us", "").replace("@g.us", "").replace("@lid", "").strip()


def detect_message_type(payload: dict) -> str:
    msg_type = payload.get("type", "")
    if msg_type in ["audio", "voice"]:
        return "voice"
    if msg_type == "text":
        return "text"
    if msg_type in ["image", "document"]:
        return "document"
    return "unsupported"


def get_message_text(payload: dict) -> str:
    text_data = payload.get("body") or payload.get("text", {})
    if isinstance(text_data, dict):
        return text_data.get("body", "") or ""
    return text_data or ""


def get_media_id(payload: dict) -> str:
    msg_type = payload.get("type", "")
    if msg_type == "audio":
        return payload.get("audio", {}).get("id", "")
    if msg_type == "image":
        return payload.get("image", {}).get("id", "")
    if msg_type == "document":
        return payload.get("document", {}).get("id", "")
    return ""


def get_media_filename(payload: dict) -> str:
    return (
        payload.get("document", {}).get("filename")
        or payload.get("image", {}).get("filename")
        or payload.get("audio", {}).get("filename")
        or "uploaded_file"
    )


def get_media_mimetype(payload: dict) -> str:
    return (
        payload.get("document", {}).get("mime_type")
        or payload.get("document", {}).get("mimeType")
        or payload.get("image", {}).get("mime_type")
        or payload.get("image", {}).get("mimeType")
        or payload.get("audio", {}).get("mime_type")
        or payload.get("audio", {}).get("mimeType")
        or ""
    )
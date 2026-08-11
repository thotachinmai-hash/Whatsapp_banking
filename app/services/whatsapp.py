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


async def send_button_message(phone_number: str, body_text: str, buttons: list[dict], trace_id: str) -> bool:
    """Send a WhatsApp Cloud API reply-buttons interactive message (up to
    3 tappable choices). Each `buttons` item is {"id": ..., "title": ...}.
    See https://developers.facebook.com/documentation/business-messaging/whatsapp/messages/interactive-reply-buttons-messages
    Caller (app/conversation/renderer.py) is responsible for enforcing
    WhatsApp's limits (≤3 buttons, ≤20-char titles) before calling this —
    this function does not validate or fall back itself."""
    url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": b["id"], "title": b["title"]}}
                    for b in buttons
                ]
            },
        },
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
                json=payload,
                timeout=15.0,
            )

        if response.status_code in [200, 201]:
            logger.info(f"[{trace_id}] WhatsApp button message sent | phone={phone_number[-4:]}")
            return True

        logger.error(
            f"[{trace_id}] WhatsApp button send failed | status={response.status_code} | body={response.text[:200]}"
        )
        return False

    except Exception as e:
        logger.error(f"[{trace_id}] WhatsApp button send error | error={e}")
        return False


async def send_list_message(
    phone_number: str, body_text: str, button_label: str, sections: list[dict], trace_id: str
) -> bool:
    """Send a WhatsApp Cloud API list interactive message (a single button
    that opens up to 10 rows across `sections`). Each section is
    {"title": ..., "rows": [{"id":..., "title":..., "description":...}]}.
    See https://developers.facebook.com/documentation/business-messaging/whatsapp/messages/interactive-list-messages
    Caller (app/conversation/renderer.py) is responsible for enforcing
    WhatsApp's limits (≤10 rows total, ≤24-char row titles, ≤20-char
    button label) before calling this."""
    url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body_text},
            "action": {
                "button": button_label,
                "sections": [
                    {
                        "title": section["title"],
                        "rows": [
                            {
                                "id": row["id"],
                                "title": row["title"],
                                **({"description": row["description"]} if row.get("description") else {}),
                            }
                            for row in section["rows"]
                        ],
                    }
                    for section in sections
                ],
            },
        },
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
                json=payload,
                timeout=15.0,
            )

        if response.status_code in [200, 201]:
            logger.info(f"[{trace_id}] WhatsApp list message sent | phone={phone_number[-4:]}")
            return True

        logger.error(
            f"[{trace_id}] WhatsApp list send failed | status={response.status_code} | body={response.text[:200]}"
        )
        return False

    except Exception as e:
        logger.error(f"[{trace_id}] WhatsApp list send error | error={e}")
        return False


def get_interactive_reply(payload: dict) -> dict | None:
    """Extract a tapped button/list reply from an inbound webhook payload
    as {"id": ..., "title": ...}, or None if this isn't an interactive
    reply. `id` is the value processors should treat as the customer's
    reply text (chosen to match a token the existing text parsers already
    accept — see app/workflows/nlu.py, app/workflows/processors/*);
    `title` is the human-readable label, kept only for logging."""
    interactive = payload.get("interactive")
    if not isinstance(interactive, dict):
        return None
    reply = interactive.get("button_reply") or interactive.get("list_reply")
    if not isinstance(reply, dict) or not reply.get("id"):
        return None
    return {"id": reply["id"], "title": reply.get("title", "")}


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


def get_media_data(payload: dict) -> str | None:
    """
    Compatibility helper: extract embedded Base64 media data when present.
    Some gateways include a `data` field or a data URI; Cloud API does not,
    but keep this for backward compatibility with existing handlers.
    """
    if not isinstance(payload, dict):
        return None
    # Check known keys where gateways sometimes embed base64 data
    for key in ("media", "image", "document", "audio", "video"):
        part = payload.get(key, {})
        if isinstance(part, dict):
            data = part.get("data") or part.get("base64") or part.get("file_data")
            if data:
                return data
    return None


async def download_media(media_id: str, trace_id: str) -> bytes | None:
    """
    Compatibility wrapper for earlier code: returns raw bytes for a media id.
    Internally uses `download_whatsapp_media` which returns (content, mime).
    """
    content, _mime = await download_whatsapp_media(media_id, trace_id)
    return content


async def send_voice_message(chat_id: str, audio_bytes: bytes, mimetype: str, trace_id: str) -> bool:
    """
    Upload audio bytes to the Graph API and send as an audio message.
    This follows the two-step Graph flow: upload media -> send message with media id.
    """
    upload_url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/media"
    send_url = f"{GRAPH_API_BASE}/{PHONE_NUMBER_ID}/messages"

    try:
        async with httpx.AsyncClient() as client:
            # Upload media as multipart/form-data
            files = {"file": ("audio.webm", audio_bytes, mimetype)}
            data = {"messaging_product": "whatsapp"}
            upload_resp = await client.post(
                upload_url,
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                data=data,
                files=files,
                timeout=30.0,
            )

        if upload_resp.status_code not in (200, 201):
            logger.error(f"[{trace_id}] Media upload failed | status={upload_resp.status_code} | body={upload_resp.text[:200]}")
            return False

        media_id = upload_resp.json().get("id")
        if not media_id:
            logger.error(f"[{trace_id}] Media upload returned no id | body={upload_resp.text[:200]}")
            return False

        # Send audio message referencing uploaded media id
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                send_url,
                headers={
                    "Authorization": f"Bearer {ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "messaging_product": "whatsapp",
                    "to": chat_id,
                    "type": "audio",
                    "audio": {"id": media_id},
                },
                timeout=15.0,
            )

        if resp.status_code in (200, 201):
            logger.info(f"[{trace_id}] Voice message sent | to={chat_id[-4:]}")
            return True

        logger.error(f"[{trace_id}] Voice message send failed | status={resp.status_code} | body={resp.text[:200]}")
        return False

    except Exception as e:
        logger.error(f"[{trace_id}] send_voice_message error | error={e}")
        return False


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
    if msg_type == "interactive":
        return "interactive"
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

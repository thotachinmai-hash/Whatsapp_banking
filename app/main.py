import time
import uuid
import os

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.logger import get_logger
from app.memory import get_redis_health
from app.metrics import get_metrics
from app.services.message_handler import handle_incoming_message
from app.services.document_parser import parse_document
from app.services.whatsapp import (
    extract_phone_number,
    get_media_id,
    get_message_text,
    get_media_filename,
    get_media_mimetype,
    download_whatsapp_media,
)
from app.services import idempotency
from app.conversation.renderer import render_and_send
from app.api.routes import router
from app.agent.agent import run_agent


load_dotenv()

logger = get_logger(__name__)


def _redact_media_for_log(value):
    """Keep webhook diagnostics useful without logging audio/document bytes."""
    if not isinstance(value, dict):
        return value
    safe = dict(value)
    media = safe.get("media")
    if isinstance(media, dict):
        safe_media = dict(media)
        if "data" in safe_media:
            safe_media["data"] = "[redacted]"
        safe["media"] = safe_media
    return safe


app = FastAPI(
    title="Finacle Banking WhatsApp Assistant",
    description="AI-powered banking assistant via WhatsApp — accepts voice and text messages",
    version="1.0.0"
)


# ── CORS Middleware ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/api")


# ── agent endpoint ──────────────────────────────────────────────
@app.post(
    "/",
    tags=["Agent"],
    summary="Receive WhatsApp Business Cloud API webhook"
)
async def whatsapp_webhook(request: Request):
    """
    Main endpoint — receives WhatsApp Business Cloud API webhook events.
    Handles both text and media messages.
    """

    webhook_trace_id = str(uuid.uuid4())[:8]

    try:
        payload = await request.json()
        logger.info(f"[{webhook_trace_id}] webhook.received")
        logger.info(f"payload received | payload={_redact_media_for_log(payload)}")

        entry = payload.get("entry")
        if not entry or not isinstance(entry, list):
            logger.info("message ignored | missing entry")
            return {"status": "ignored", "reason": "missing_entry"}

        change = entry[0].get("changes", [{}])[0]
        value = change.get("value", {})
        messages = value.get("messages") or []
        if not messages:
            logger.info("message ignored | no messages")
            return {"status": "ignored", "reason": "no_messages"}

        message = messages[0]
        sender_phone = message.get("from")
        if not sender_phone:
            logger.warning("message ignored | missing sender phone")
            return {"status": "ignored", "reason": "missing_sender"}

        message_type = message.get("type", "")
        media = message.get(message_type, {}) if message_type else {}
        # Cloud API: use the message's `id` as the external id
        external_message_id = message.get("id") or message.get("message_id") or ""

        # Normalize media details using whatsapp service helpers
        media_id = get_media_id(message)
        file_name = get_media_filename(message)
        mime_type = get_media_mimetype(message)

        logger.info(
            f"[{webhook_trace_id}] message.id={external_message_id} | sender={sender_phone} | type={message_type}"
        )

        claim_result = idempotency.claim(external_message_id) if external_message_id else idempotency.ClaimResult.CLAIMED
        if claim_result == idempotency.ClaimResult.DUPLICATE:
            logger.info(
                f"[{webhook_trace_id}] message.idempotency.duplicate | external_message_id={external_message_id}"
            )
            return {"status": "duplicate", "trace_id": webhook_trace_id}

        if claim_result == idempotency.ClaimResult.REDIS_UNAVAILABLE:
            logger.error(
                f"[{webhook_trace_id}] message.idempotency.redis_unavailable | external_message_id={external_message_id}"
            )
            await render_and_send(
                "I'm having trouble processing your request right now. Please try again shortly.",
                sender_phone,
                webhook_trace_id,
            )
            return {"status": "error", "reason": "idempotency_unavailable", "trace_id": webhook_trace_id}

        message_data = {
            "from": sender_phone,
            "to": value.get("metadata", {}).get("phone_number_id", ""),
            "body": message.get("text", {}).get("body", "") if message_type == "text" else get_message_text(message) if message_type == "text" else "",
            "type": message_type,
            "media": media,
            "media_id": media_id or (media.get("id") if isinstance(media, dict) else ""),
            "mimeType": mime_type or (media.get("mime_type") or media.get("mimeType", "")),
            "fileName": file_name or (media.get("filename") or media.get("file_name", "")),
            "raw": payload
        }

        # Expose the media object at top-level keys matching Graph API shape
        # so downstream helpers (get_media_id, get_media_data) can find it.
        if message_type in ("audio", "voice") and isinstance(media, dict):
            message_data["audio"] = media
        elif message_type in ("image", "document", "video") and isinstance(media, dict):
            message_data[message_type] = media
        elif message_type == "interactive" and isinstance(media, dict):
            message_data["interactive"] = media

        if value.get("statuses"):
            logger.info("message ignored | status update")
            return {"status": "ignored", "reason": "status_update"}

        lock_token = idempotency.acquire_conversation_lock(sender_phone)
        try:
            logger.info(f"[{webhook_trace_id}] message.processing.started | external_message_id={external_message_id}")
            result = await handle_incoming_message(message_data)
        finally:
            idempotency.release_conversation_lock(sender_phone, lock_token)

        if result.get("status") == "error":
            idempotency.mark_failed(external_message_id)
            logger.warning(
                f"[{webhook_trace_id}] message.processing.failed | external_message_id={external_message_id}"
            )
        else:
            idempotency.mark_completed(external_message_id)
            logger.info(
                f"[{webhook_trace_id}] message.processing.completed | external_message_id={external_message_id}"
            )

        return result

    except Exception as e:
        logger.error(f"Webhook error | error={e}")
        eid = locals().get("external_message_id")
        if eid:
            idempotency.mark_failed(eid)
        raise HTTPException(status_code=500, detail=str(e))


# ── Test endpoint ─────────────────────────────────────────────────
class TestMessageRequest(BaseModel):
    phone_number: str
    message: str
    message_type: str = "text"



@app.post(
    "/api/test/message",
    tags=["Testing"],
    summary="Test the agent without WhatsApp"
)
async def test_message(request: TestMessageRequest):
    """
    Send a test message directly to the agent without going through WhatsApp.
    """

    trace_id = str(uuid.uuid4())[:8]

    start = time.time()


    response = await run_agent(
        query=request.message,
        phone_number=request.phone_number,
        trace_id=trace_id
    )


    duration = (time.time() - start) * 1000


    return {
        "trace_id": trace_id,
        "phone_number": request.phone_number,
        "query": request.message,
        "response": response,
        "duration_ms": round(duration, 2)
    }

# ── Document parser test endpoint ────────────────────────────────

@app.post(
    "/api/test/document",
    tags=["Testing"],
    summary="Test document parser"
)
async def test_document(
    file: UploadFile = File(...)
):
    """
    Upload image/pdf/docx and test Qwen vision document extraction.
    """

    trace_id = str(uuid.uuid4())[:8]

    start = time.time()

    try:

        file_bytes = await file.read()

        logger.info(
            f"[{trace_id}] Document upload received | "
            f"name={file.filename} | "
            f"size={len(file_bytes)} bytes"
        )


        result = await parse_document(
            file_bytes=file_bytes,
            filename=file.filename,
            prompt="""
            Extract all information from this document.

            Rules:
            - Return ONLY valid JSON.
            - Do not summarize.
            - Preserve exact values.
            - Extract tables also.
            - Extract handwritten text if visible.
            """,
            trace_id=trace_id
        )


        duration = (time.time() - start) * 1000


        return {
            "trace_id": trace_id,
            "filename": file.filename,
            "duration_ms": round(duration, 2),
            "result": result
        }


    except Exception as e:

        logger.error(
            f"[{trace_id}] Document test failed | error={e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ── System endpoints ──────────────────────────────────────────────
@app.get(
    "/health",
    tags=["System"],
    summary="Health check"
)
async def health():
    """
    Check health of all system components.
    """

    from app.database import get_db_connection

    db_healthy = False

    try:
        conn = get_db_connection()
        conn.close()
        db_healthy = True

    except Exception:
        pass


    return {
        "status": "healthy",
        "components": {
            "api": "healthy",
            "redis": "connected" if get_redis_health() else "disconnected",
            "postgres": "connected" if db_healthy else "disconnected"
        }
    }



@app.get(
    "/metrics",
    tags=["System"],
    summary="System metrics"
)
async def metrics():
    """
    Real-time metrics for all system components.
    """

    return get_metrics()



@app.get(
    "/",
    tags=["System"]
)
async def root(request: Request):
    """
    Verify WhatsApp Business Cloud webhook or return service metadata.
    """
    query_params = dict(request.query_params)
    if query_params.get("hub.mode") == "subscribe":
        challenge = query_params.get("hub.challenge")
        verify_token = query_params.get("hub.verify_token")
        expected_token = os.getenv("VERIFY_TOKEN", "")
        if verify_token == expected_token:
            return PlainTextResponse(content=challenge or "", status_code=200)
        raise HTTPException(status_code=403, detail="Invalid verify_token")

    return {
        "service": "Finacle Banking WhatsApp Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "test": "POST /api/test/message",
        "test_document": "POST /api/test/document"
    }

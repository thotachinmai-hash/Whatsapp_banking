import time
import uuid

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.logger import get_logger
from app.memory import get_redis_health
from app.metrics import get_metrics
from app.services.message_handler import handle_incoming_message
from app.services.document_parser import parse_document
from app.services.whatsapp import extract_phone_number, get_sender_phone, get_external_message_id
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
    "/openwa/whatsapp",
    tags=["Agent"],
    summary="Receive messages from OpenWA WhatsApp Gateway"
)
async def whatsapp_webhook(request: Request):
    """
    Main endpoint — receives all WhatsApp messages from OpenWA.
    Handles both text and voice messages.
    """

    webhook_trace_id = str(uuid.uuid4())[:8]

    try:
        payload = await request.json()
        logger.info(f"[{webhook_trace_id}] webhook.received")
        logger.info(f"payload received | payload={_redact_media_for_log(payload)}")

        logger.info(
            f"payload received | event={payload.get('event', 'unknown')}"
        )

        # Only process message events
        event = payload.get("event", "")

        if event != "message.received":
            logger.info(
                f"message ignored | event={event}"
            )
            return {
                "status": "ignored",
                "event": event
            }

        # Extract OpenWA message data
        data = payload.get("data", {})
        logger.info(f"data={_redact_media_for_log(data)}")

        # Banking conversations are private. Ignore group messages so a
        # birthday wish or other group chat text cannot start onboarding or a
        # transfer for the group ID.
        if data.get("isGroup"):
            logger.info("message ignored | reason=group_chat")
            return {"status": "ignored", "reason": "group_chat"}

        chat_id = data.get("chatId")

        # ── Idempotency guard ──────────────────────────────────────
        # Claim the OpenWA message id BEFORE any expensive work (phone
        # resolution, transcription, OCR, intent classification, LLM,
        # workflow execution) so a webhook retry / duplicate delivery
        # of the SAME event can never start a second conversational turn
        # or a second banking action. See app/services/idempotency.py
        # and docs/current_architecture.md "Webhook Reliability &
        # Idempotency — Phase 6" for the full design.
        external_message_id = get_external_message_id(data)
        if not external_message_id:
            # No stable id in this payload build. Fall back to the
            # message's own delivery timestamp (stable across retries of
            # the same event, unlike processing time) rather than message
            # text/phone — a documented, reduced-reliability fallback.
            timestamp = data.get("t") or data.get("timestamp")
            if timestamp and chat_id:
                external_message_id = f"{chat_id}:{data.get('type', '')}:{timestamp}"
            else:
                logger.warning(
                    f"[{webhook_trace_id}] No external message id or timestamp available — "
                    f"idempotency cannot be enforced for this event"
                )

        claim_result = idempotency.claim(external_message_id) if external_message_id else idempotency.ClaimResult.CLAIMED

        if claim_result == idempotency.ClaimResult.DUPLICATE:
            logger.info(
                f"[{webhook_trace_id}] message.idempotency.duplicate | "
                f"external_message_id={external_message_id}"
            )
            return {"status": "duplicate", "trace_id": webhook_trace_id}

        if claim_result == idempotency.ClaimResult.REDIS_UNAVAILABLE:
            logger.error(
                f"[{webhook_trace_id}] message.idempotency.redis_unavailable | "
                f"external_message_id={external_message_id}"
            )
            # Fail-safe: we cannot guarantee this event hasn't already been
            # claimed by a concurrent/duplicate request, so we do not run
            # ConversationManager/workflows/LLM for it. This mirrors the
            # user-safe error responses already used elsewhere in the app.
            if chat_id:
                await render_and_send(
                    "I'm having trouble processing your request right now. Please try again shortly.",
                    chat_id,
                    webhook_trace_id,
                )
            return {"status": "error", "reason": "idempotency_unavailable", "trace_id": webhook_trace_id}

        logger.info(
            f"[{webhook_trace_id}] message.idempotency.claimed | "
            f"external_message_id={external_message_id}"
        )
        is_lid = bool(chat_id and "@lid" in chat_id)

        sender_phone = None
        if chat_id:
            # A normal WhatsApp chat ID already contains the real number.
            # Only LID identifiers require the OpenWA contact lookup.
            if not is_lid:
                sender_phone = extract_phone_number(chat_id)
            else:
                sender_phone = await get_sender_phone(chat_id)

        if is_lid and not sender_phone:
            # The LID's numeric portion is NOT a phone number — it must
            # never be used to look up or create a customer/account record.
            # If OpenWA couldn't resolve the real number behind this LID,
            # bail out instead of silently misidentifying the customer.
            logger.error(
                f"Could not resolve real phone number for LID chat — "
                f"refusing to use the LID for customer lookup | chat_id={chat_id}"
            )
            idempotency.mark_failed(external_message_id)
            await render_and_send(
                "Sorry, we couldn't verify your WhatsApp number right now. Please try again shortly.",
                chat_id,
                str(uuid.uuid4())[:8],
            )
            return {"status": "error", "reason": "phone_resolution_failed"}

        resolved_phone = sender_phone if is_lid else (sender_phone or chat_id)
        normalized_phone = extract_phone_number(resolved_phone) if resolved_phone else ""

        logger.info(f"Chat ID: {chat_id}")
        logger.info(f"Resolved sender phone: {sender_phone}")

        message_data = {
            "from": normalized_phone or data.get("chatId"),
            "to": data.get("to"),
            "body": data.get("body"),
            "type": data.get("type"),
            "media": data.get("media", {}),
            "mediaUrl": data.get("mediaUrl"),
            "fileName": data.get("fileName"),
            "mimeType": data.get("mimeType"),
            "raw": data
        }

        logger.info(
            f"Processed message | phone={message_data['from']} | type={message_data['type']}"
        )

        # Best-effort per-phone lock: serializes two near-simultaneous
        # messages for the same customer (e.g. "transfer money" then "500")
        # so they don't race on the same workflow:{phone} Redis key. Not a
        # queue — bounded wait, then proceeds without the lock rather than
        # dropping the message.
        lock_token = idempotency.acquire_conversation_lock(normalized_phone) if normalized_phone else None
        try:
            logger.info(f"[{webhook_trace_id}] message.processing.started | external_message_id={external_message_id}")
            result = await handle_incoming_message(message_data)
        finally:
            idempotency.release_conversation_lock(normalized_phone, lock_token)

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
        logger.error(
            f"Webhook error | error={e}"
        )
        eid = locals().get("external_message_id")
        if eid:
            idempotency.mark_failed(eid)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


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
async def root():

    return {
        "service": "Finacle Banking WhatsApp Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "webhook": "POST /webhook/whatsapp",
        "test": "POST /api/test/message",
        "test_document": "POST /api/test/document"
    }

import json
import os
import time
import uuid
from typing import Any, Dict
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.logger import get_logger
from app.memory import get_redis_health
from app.metrics import get_metrics
from app.services.message_handler import handle_incoming_message
from app.services.document_parser import parse_document
from app.services.whatsapp import get_sender_phone
from app.api.routes import router
from app.agent.agent import run_agent


load_dotenv()
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

logger = get_logger(__name__)


app = FastAPI(
    title="Finacle WhatsApp Banking Assistant",
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



#webhook endpoint for WA business API
@app.get("/")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("WEBHOOK VERIFIED")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    else:
        return PlainTextResponse(content="Forbidden", status_code=403)


@app.post("/")
async def receive_webhook(request: Request):
    raw_body = await request.body()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    print("Webhook POST received:", body)


    if raw_body:
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            # Not valid JSON, just log raw text
            body = {"raw_body": raw_body.decode("utf-8", errors="ignore")}
    else:
        body = {}

    print(json.dumps(body, indent=2))

    # Example: parse WhatsApp message if present
    try:
        change = body["entry"][0]["changes"][0]
        value = change["value"]

        if "messages" in value:
            message = value["messages"][0]
            sender = message["from"]
            message_type = message["type"]

            if message_type == "text":
                text = message["text"]["body"]
                print("Sender:", sender)
                print("Message:", text)

                # Send acknowledgement back
                send_message(sender, "✅ Got your message! Thanks for reaching out.")

    except Exception as e:
        print("Error parsing webhook:", e)

    return PlainTextResponse(content="OK", status_code=200)

def send_message(to, text):

    url = f"https://graph.facebook.com/v23.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": text
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    print(response.status_code)
    print(response.text)



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

    try:
        payload = await request.json()
        logger.info(f"payload received | payload={payload}")

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
        logger.info(f"data={data}")
        chat_id = data.get("chatId")

        sender_phone = None
        if chat_id:
            sender_phone = await get_sender_phone(chat_id)

        logger.info(f"Chat ID: {chat_id}")
        logger.info(f"Resolved sender phone: {sender_phone}")

        message_data = {
            "from": data.get("chatId"),
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

        result = await handle_incoming_message(message_data)

        return result


    except Exception as e:
        logger.error(
            f"Webhook error | error={e}"
        )

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
        "service": "HSBC WhatsApp Banking Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "webhook": "POST /webhook/whatsapp",
        "test": "POST /api/test/message",
        "test_document": "POST /api/test/document"
    }

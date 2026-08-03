import time
import uuid
from app.logger import get_logger
from app.metrics import (
    log_message_received,
    log_transcription,
    log_agent_call,
    log_whatsapp_send,
    log_request
)
from app.services.whatsapp import (
    extract_phone_number,
    detect_message_type,
    get_message_text,
    get_media_url,
    send_text_message,
    get_media_filename,
    get_media_mimetype,
    get_media_data
)
from app.services.transcription import download_audio, transcribe_audio
from app.services.document_parser import download_document, parse_document
from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_COLLECT_PAN,
    WORKFLOW_CHEQUE,
    WORKFLOW_LOAN,
    WORKFLOW_KYC,
)
from app.workflows.memory import get_workflow
from app.agent.agent import run_agent
import os
import base64

logger = get_logger(__name__)

OPENWA_API_KEY = os.getenv("OPENWA_API_KEY", "")


async def handle_incoming_message(payload: dict) -> dict:
    """
    Main handler for incoming WhatsApp messages.

    Flow:
    1. Generate trace ID
    2. Extract phone number and message type
    3. If voice — download and transcribe
    4. If text — use directly
    5. Run LangGraph agent
    6. Send response back to WhatsApp
    7. Log everything with trace ID
    """
    request_start = time.time()
    trace_id = str(uuid.uuid4())[:8]

    try:
        # Extract sender info
        sender = payload.get("from", "") or payload.get("sender", "")
        phone_number = extract_phone_number(sender)
        logger.info(f"[{trace_id}] Extracted phone number: {phone_number}")

        # Extract to person info
        to_person = payload.get("to", "")
        logger.info(f"[{trace_id}] Extracted to person: {to_person}")

        # Extract from person info
        from_person = payload.get("from", "")
        logger.info(f"[{trace_id}] Extracted from person: {from_person}")

        if not phone_number:
            logger.warning(f"[{trace_id}] No phone number in payload")
            return {"status": "error", "message": "No sender found"}

        # Detect message type
        msg_type = detect_message_type(payload)
        logger.info(f"[{trace_id}] Detected message type: {msg_type}")
        log_message_received(phone_number, msg_type, trace_id)

        logger.info(f"[{trace_id}] Message received | phone={phone_number[-4:]} | type={msg_type}")

        # Handle based on type
        query = ""
        parsed_document = None

        if msg_type == "text":
            query = get_message_text(payload)
            logger.info(f"[{trace_id}] Text message | content={query[:50]}")

        elif msg_type == "voice":
            logger.info(f"[{trace_id}] Voice message — transcribing")
            media_url = get_media_url(payload)

            if not media_url:
                logger.error(f"[{trace_id}] No media URL in voice message")
                await send_text_message(
                    from_person,
                    "Sorry, I couldn't process your voice message. Please try sending a text message.",
                    trace_id
                )
                return {"status": "error", "trace_id": trace_id}

            # Download audio
            audio_data = await download_audio(media_url, OPENWA_API_KEY, trace_id)
            if not audio_data:
                await send_text_message(
                    from_person,
                    "Sorry, I couldn't download your voice message. Please try again.",
                    trace_id
                )
                return {"status": "error", "trace_id": trace_id}

            # Transcribe with Groq Whisper
            transcribe_start = time.time()
            query = await transcribe_audio(audio_data, trace_id)
            transcribe_duration = (time.time() - transcribe_start) * 1000

            if not query:
                log_transcription(False, transcribe_duration, trace_id)
                await send_text_message(
                    from_person,
                    "Sorry, I couldn't understand your voice message. Please try again or send a text message.",
                    trace_id
                )
                return {"status": "error", "trace_id": trace_id}

            log_transcription(True, transcribe_duration, trace_id)
            logger.info(f"[{trace_id}] Transcribed: {query[:50]}")

        elif msg_type == "document":

            logger.info(f"[{trace_id}] Document received")

            media_url = get_media_url(payload)
            media_data = get_media_data(payload)
            filename = get_media_filename(payload)
            mime_type = get_media_mimetype(payload)

            if not media_url and not media_data:
                logger.error(f"[{trace_id}] No media found")

                await send_text_message(
                    from_person,
                    "Sorry, I couldn't access your uploaded document. Please try again.",
                    trace_id,
                )

                return {
                    "status": "error",
                    "trace_id": trace_id,
                }

            #
            # Download document
            #
            #
            # Get document bytes
            #
            if media_data:

                logger.info(
                    f"[{trace_id}] Using Base64 media from webhook"
                )

                try:
                    # OpenWA may provide either raw base64 or a data URI.
                    encoded_media = media_data.split(",", 1)[-1]
                    file_bytes = base64.b64decode(encoded_media)

                except Exception as e:

                    logger.error(
                        f"[{trace_id}] Failed to decode Base64 media | error={e}"
                    )

                    await send_text_message(
                        from_person,
                        "Sorry, I couldn't process your uploaded document.",
                        trace_id
                    )

                    return {
                        "status": "error",
                        "trace_id": trace_id
                    }

            else:

                logger.info(
                    f"[{trace_id}] Downloading document from media URL"
                )

                file_bytes = await download_document(
                    media_url=media_url,
                    api_key=OPENWA_API_KEY,
                    trace_id=trace_id
                )

            if not file_bytes:

                await send_text_message(
                    from_person,
                    "Sorry, I couldn't download your document. Please try again.",
                    trace_id
                )

                return {
                    "status": "error",
                    "trace_id": trace_id
                }

            #
            # Parse document
            #
            parse_start = time.time()

            logger.info(
                f"[{trace_id}] filename={filename}, mime_type={mime_type}, file_size={len(file_bytes)} bytes"
            )

            active_workflow = get_workflow(phone_number)
            document_prompt = """
                Extract every piece of information from this document.

                Return ONLY valid JSON.

                Do not summarize.

                Preserve all values exactly.
                """
            if active_workflow and active_workflow.get("step") == STEP_COLLECT_AADHAAR:
                document_prompt = """
                    Read the Aadhaar card image. Return ONLY valid JSON in this exact shape:
                    {"aadhaar_number": "12 digits, preserving the digits exactly"}
                    If the number is not visible or cannot be read, use an empty string.
                    Do not include any other text.
                    """
            elif active_workflow and active_workflow.get("step") == STEP_COLLECT_PAN:
                document_prompt = """
                    Read the PAN card image. Return ONLY valid JSON in this exact shape:
                    {"pan_number": "PAN number in the format ABCDE1234F, preserving it exactly"}
                    If the number is not visible or cannot be read, use an empty string.
                    Do not include any other text.
                    """
            elif active_workflow and active_workflow.get("type") == WORKFLOW_CHEQUE:
                document_prompt = """
                    Read this cheque image and return ONLY valid JSON in this exact shape:
                    {"bank_name":"", "branch":"", "payee":"", "amount_in_figures":"",
                     "amount_in_words":"", "numbers":"", "signatory_title":""}
                    Extract values exactly as printed. Use an empty string when a field
                    is missing, obscured, or unreadable. Do not guess or include other text.
                    """
            elif active_workflow and active_workflow.get("type") == WORKFLOW_LOAN:
                document_prompt = """
                    Read this loan application form and return ONLY valid JSON.
                    Extract applicant_name, monthly_income, employment_type,
                    requested_amount, tenure_months, and purpose. Preserve values
                    exactly. Use empty strings for missing or unreadable fields.
                    """
            elif active_workflow and active_workflow.get("type") == WORKFLOW_KYC:
                document_prompt = """
                    Read this KYC document and return ONLY valid JSON.
                    Extract full_name, date_of_birth, address, aadhaar_number,
                    and pan_number. Preserve values exactly. Use empty strings for
                    missing or unreadable fields.
                    """

            parsed_document = await parse_document(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=mime_type,
                prompt=document_prompt,
                trace_id=trace_id
            )

            logger.info(
                f"[{trace_id}] parse_document response={parsed_document}"
            )
            if not parsed_document.get("success"):

                logger.error(
                    f"[{trace_id}] Document parsing failed"
                )

                await send_text_message(
                    from_person,
                    "Sorry, I couldn't extract information from your document. Please try again.",
                    trace_id
                )

                return {
                    "status": "error",
                    "trace_id": trace_id
                }

            parse_duration = (time.time() - parse_start) * 1000

            logger.info(
                f"[{trace_id}] Document parsed successfully | duration={parse_duration:.2f}ms"
            )

            #
            # Convert extracted document into a query
            # for the existing LangGraph agent.
            #
            query = f"""
            Customer uploaded a document.

            Filename:
            {filename}

            Mime Type:
            {mime_type}

            Extracted Document:

            {parsed_document["content"]}
            """
        
        else:
            logger.info(f"[{trace_id}] Unsupported message type: {msg_type}")
            await send_text_message(
                from_person,
                "Sorry, I can only process text and voice messages. Please send a text or voice note.",
                trace_id
            )
            return {"status": "unsupported", "trace_id": trace_id}

        if not query or not query.strip():
            await send_text_message(
                from_person,
                "Sorry, I received an empty message. Please try again.",
                trace_id
            )
            return {"status": "error", "trace_id": trace_id}

        # Run LangGraph agent
        # Run LangGraph agent
        agent_start = time.time()

        response = await run_agent(
            query=query,
            phone_number=phone_number,
            trace_id=trace_id,
            parsed_document=parsed_document,
        )

        agent_duration = (time.time() - agent_start) * 1000
        log_agent_call(True, agent_duration, [], trace_id)

        # Send response back to WhatsApp
        send_success = await send_text_message(from_person, response, trace_id)
        log_whatsapp_send(send_success, trace_id)

        # Log total request time
        total_duration = (time.time() - request_start) * 1000
        log_request(total_duration, trace_id)

        logger.info(f"[{trace_id}] Request complete | total={total_duration:.2f}ms")

        return {
            "status": "success",
            "trace_id": trace_id,
            "phone": phone_number[-4:],
            "type": msg_type,
            "duration_ms": round(total_duration, 2)
        }

    except Exception as e:
        total_duration = (time.time() - request_start) * 1000
        logger.error(f"[{trace_id}] Request failed | error={e} | duration={total_duration:.2f}ms")
        return {"status": "error", "trace_id": trace_id, "error": str(e)}

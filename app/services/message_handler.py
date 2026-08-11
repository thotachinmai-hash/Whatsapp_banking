import time
import uuid
import time
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
    get_interactive_reply,
    get_message_text,
    get_media_filename,
    get_media_mimetype,
    get_media_data,
    get_media_id,
    download_media,
)
from app.conversation.renderer import as_structured_response, render_and_send
from app.services.transcription import download_audio, transcribe_audio
from app.services.tts import synthesize_voice_note
from app.services.whatsapp import send_voice_message
from app.services.document_parser import download_document, parse_document
from app.workflows.constants import (
    STEP_COLLECT_AADHAAR,
    STEP_COLLECT_PAN,
    STEP_SELECT_LOAN_TYPE,
    STEP_UPLOAD_CHEQUE,
    STEP_UPLOAD_KYC_FORM,
    WORKFLOW_CHEQUE,
    WORKFLOW_LOAN,
    WORKFLOW_KYC,
)
from app.workflows.memory import create_workflow, create_workflow_model, get_workflow
from app.workflows.document_detect import detect_workflow_type
from app.workflows.processors.cheque import ChequeWorkflowProcessor
from app.workflows.processors.kyc import KYCWorkflowHandler
from app.database import get_customer_by_phone
from app.agent.agent import run_agent
from app.conversation.responses import errors as error_templates
import os
import base64

logger = get_logger(__name__)


def build_document_prompt(active_workflow: dict | None, filename: str) -> str:
    """Build the OCR prompt for onboarding documents so profile fields can be validated."""
    # No active workflow — the customer just sent an image/document cold,
    # with no prior "I want to deposit a cheque" text. A vague "extract
    # everything" prompt can't reliably tell a cheque from a KYC document
    # from a loan form, so ask the model to classify AND extract in one
    # call — see app/workflows/document_detect.py, which reads
    # "document_type" from this response to auto-start the right workflow
    # without discarding the already-parsed content.
    document_prompt = """
        Look at this document/image and identify what it is, then extract
        its fields. Return ONLY valid JSON in this exact shape:
        {"document_type": "cheque" | "kyc" | "loan_form" | "other",
         "bank_name": "", "branch": "", "payee": "", "amount_in_figures": "",
         "amount_in_words": "", "numbers": "", "signatory_title": "",
         "date_written": "", "drawer_name": "",
         "aadhaar_number": "", "pan_number": "", "full_name": "",
         "date_of_birth": "", "address": "", "guardian_name": "",
         "applicant_name": "", "monthly_income": "", "employment_type": "",
         "requested_amount": "", "tenure_months": "", "purpose": ""}

        document_type rules:
        - "cheque" if it is a bank cheque (payee, amount, cheque number visible)
        - "kyc" if it is an identity/address document (Aadhaar, PAN, passport, voter ID, driving licence)
        - "loan_form" if it is a loan application form
        - "other" if it does not clearly match any of the above

        Only fill in fields that are actually visible on THIS document —
        leave every other field as an empty string. Preserve values
        exactly as printed. Do not guess, summarize, or include any text
        outside the JSON object.
        """
    if active_workflow and active_workflow.get("step") == STEP_COLLECT_AADHAAR:
        document_prompt = """
            Read the Aadhaar card image. Return ONLY valid JSON in this exact shape:
            {"aadhaar_number": "12 digits, preserving the digits exactly",
             "full_name": "full name as printed",
             "date_of_birth": "date of birth as printed",
             "address": "address as printed",
             "guardian_name": "father/spouse/guardian name as printed"}
            If a field is not visible or cannot be read, use an empty string.
            Do not include any other text.
            """
    elif active_workflow and active_workflow.get("step") == STEP_COLLECT_PAN:
        document_prompt = """
            Read the PAN card image. Return ONLY valid JSON in this exact shape:
            {"pan_number": "PAN number in the format ABCDE1234F, preserving it exactly",
             "full_name": "full name as printed",
             "date_of_birth": "date of birth as printed",
             "address": "address as printed",
             "guardian_name": "father/spouse/guardian name as printed"}
            If a field is not visible or cannot be read, use an empty string.
            Do not include any other text.
            """
    elif active_workflow and active_workflow.get("type") == WORKFLOW_CHEQUE:
        document_prompt = """
            Read this cheque image and return ONLY valid JSON in this exact shape:
            {"bank_name":"", "branch":"", "payee":"", "amount_in_figures":"",
             "amount_in_words":"", "numbers":"", "signatory_title":"",
             "date_written":"", "drawer_name":""}
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

    return document_prompt


async def _handle_bare_document_upload(
    detected_type: str,
    phone_number: str,
    parsed_document: dict,
    trace_id: str = "",
) -> str | None:
    """A cheque/KYC/loan-form image arrived with no active workflow and no
    prior "I want to deposit a cheque"-style text. Auto-start the matching
    workflow and hand it the already-parsed content in this same turn,
    instead of discarding it and asking the customer to upload again.

    Only called for a REGISTERED customer (checked by the caller) — an
    unregistered customer's cold upload still goes through the normal
    registration path first, since there's nowhere safe to carry image
    data across the registration conversation.

    Returns the response text if handled here, or None to fall through to
    the existing generic-query/LLM path (loan forms need a loan type
    chosen first, so that case just starts the selection step and stashes
    the parsed content for app/workflows/processors/loan.py to pick up
    once a type is chosen).
    """
    if detected_type == "cheque":
        workflow = create_workflow_model(WORKFLOW_CHEQUE, STEP_UPLOAD_CHEQUE)
        create_workflow(phone_number, workflow)
        logger.info(f"[{trace_id}] Auto-detected cheque upload | phone={phone_number[-4:]}")
        result = await ChequeWorkflowProcessor().handle(
            workflow=workflow, phone_number=phone_number, query="",
            parsed_document=parsed_document, trace_id=trace_id,
        )
        return result.get("response")

    if detected_type == "kyc":
        workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
        create_workflow(phone_number, workflow)
        logger.info(f"[{trace_id}] Auto-detected KYC document upload | phone={phone_number[-4:]}")
        result = await KYCWorkflowHandler().handle(
            workflow=workflow, phone_number=phone_number, query="",
            parsed_document=parsed_document, trace_id=trace_id,
        )
        return result.get("response")

    if detected_type == "loan_form":
        # Which loan product this is for isn't on the form itself — ask,
        # and stash the already-extracted fields so the very next reply
        # (the loan type) completes the form instead of requesting the
        # same document again. See loan.py's _select_type pending-content
        # pickup.
        workflow = create_workflow_model(
            WORKFLOW_LOAN, STEP_SELECT_LOAN_TYPE,
            data={"pending_document_content": parsed_document.get("content", {})},
        )
        create_workflow(phone_number, workflow)
        logger.info(f"[{trace_id}] Auto-detected loan form upload | phone={phone_number[-4:]}")
        return (
            "📝 That looks like a loan application form. Which loan is this for?\n\n"
            "1. Personal Loan\n2. Home Loan\n3. Vehicle Loan\n4. Education Loan\n\n"
            "Reply with a number or name. Reply *Cancel* to stop."
        )

    return None


async def send_voice_reply(response, chat_id: str, trace_id: str = "") -> bool:
    """The voice-out half of voice-to-voice: synthesize the response text
    and send it back as a voice note, mirroring how the customer reached
    out. Falls back to a plain text reply (via render_and_send) whenever
    synthesis or sending audio fails, since a voice-in customer still
    needs a reply even if TTS is unavailable.

    `response` may be a plain string or a StructuredResponse carrying
    WhatsApp interactive buttons/list metadata (see
    app/conversation/renderer.py) — only the body text is spoken; a
    voice-in customer can still just say "yes"/"1" back, so the tappable
    options aren't needed on this path.
    """
    response_text = as_structured_response(response).text
    synthesized = await synthesize_voice_note(response_text, trace_id=trace_id)
    if synthesized is None:
        logger.info(f"[{trace_id}] Voice reply unavailable — falling back to text")
        return await render_and_send(response_text, chat_id, trace_id)

    audio_bytes, mimetype = synthesized
    sent = await send_voice_message(chat_id, audio_bytes, mimetype, trace_id)
    if not sent:
        logger.info(f"[{trace_id}] Voice send failed — falling back to text")
        return await render_and_send(response_text, chat_id, trace_id)
    return True


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
        detected_language = None
        is_voice_message = msg_type == "voice"

        if msg_type == "text":
            query = get_message_text(payload)
            logger.info(f"[{trace_id}] Text message | content={query[:50]}")

        elif msg_type == "interactive":
            reply = get_interactive_reply(payload)
            if not reply:
                logger.warning(f"[{trace_id}] Interactive message with no recognizable reply")
                await render_and_send(
                    "Sorry, I couldn't read that selection. Please try again.",
                    from_person,
                    trace_id,
                )
                return {"status": "error", "trace_id": trace_id}
            # `id` is what every text parser (interpret_confirmation,
            # digit/menu matching) already expects — see
            # app/conversation/renderer.py's module docstring for the id
            # convention. `title` is logged only, never parsed as input.
            query = reply["id"]
            logger.info(f"[{trace_id}] Interactive reply | id={reply['id']!r} | title={reply['title']!r}")

        elif msg_type == "voice":
            logger.info(f"[{trace_id}] Voice message — transcribing")
            media_id = get_media_id(payload)
            media_data = get_media_data(payload)

            if media_data:
                try:
                    encoded_audio = media_data.split(",", 1)[-1]
                    encoded_audio += "=" * (-len(encoded_audio) % 4)
                    audio_data = base64.b64decode(encoded_audio)
                    logger.info(f"[{trace_id}] Using embedded voice audio | size={len(audio_data)} bytes")
                except (ValueError, base64.binascii.Error) as exc:
                    logger.error(f"[{trace_id}] Embedded voice audio decode failed | error={exc}")
                    audio_data = None
            elif media_id:
                audio_data = await download_media(media_id, trace_id)
            else:
                logger.error(f"[{trace_id}] No voice media data or media id")
                audio_data = None

            if not audio_data:
                await render_and_send(
                    error_templates.render_voice_unavailable_error(),
                    from_person,
                    trace_id
                )
                return {"status": "error", "trace_id": trace_id}

            # Transcribe with Groq Whisper
            transcribe_start = time.time()
            query, detected_language = await transcribe_audio(audio_data, trace_id)
            transcribe_duration = (time.time() - transcribe_start) * 1000

            if not query:
                log_transcription(False, transcribe_duration, trace_id)
                await render_and_send(
                    error_templates.render_transcription_error(),
                    from_person,
                    trace_id
                )
                return {"status": "error", "trace_id": trace_id}

            log_transcription(True, transcribe_duration, trace_id)
            logger.info(f"[{trace_id}] Voice transcription ready | query={query[:50]}")

        elif msg_type == "document":

            logger.info(f"[{trace_id}] Document received")

            media_id = get_media_id(payload)
            media_data = get_media_data(payload)
            filename = get_media_filename(payload)
            mime_type = get_media_mimetype(payload)

            if not media_id and not media_data:
                logger.error(f"[{trace_id}] No media found")

                await render_and_send(
                    "Sorry, I couldn't access your uploaded document. Please try again.",
                    from_person,
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

                    await render_and_send(
                        "Sorry, I couldn't process your uploaded document.",
                        from_person,
                        trace_id
                    )

                    return {
                        "status": "error",
                        "trace_id": trace_id
                    }

            else:

                logger.info(
                    f"[{trace_id}] Downloading document from WhatsApp media ID"
                )

                file_bytes = await download_media(
                    media_id,
                    trace_id=trace_id
                )

            if not file_bytes:

                await render_and_send(
                    "Sorry, I couldn't download your document. Please try again.",
                    from_person,
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
            document_prompt = build_document_prompt(active_workflow, filename)

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

                await render_and_send(
                    "Sorry, I couldn't extract information from your document. Please try again.",
                    from_person,
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

            if not active_workflow and get_customer_by_phone(phone_number):
                detected_type = detect_workflow_type(parsed_document.get("content", {}))
                if detected_type:
                    auto_response = await _handle_bare_document_upload(
                        detected_type, phone_number, parsed_document, trace_id
                    )
                    if auto_response is not None:
                        send_success = await render_and_send(auto_response, from_person, trace_id)
                        log_whatsapp_send(send_success, trace_id)
                        total_duration = (time.time() - request_start) * 1000
                        log_request(total_duration, trace_id)
                        return {
                            "status": "success",
                            "trace_id": trace_id,
                            "phone": phone_number[-4:],
                            "type": msg_type,
                            "duration_ms": round(total_duration, 2),
                        }

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
            await render_and_send(
                "Sorry, I can only process text and voice messages. Please send a text or voice note.",
                from_person,
                trace_id
            )
            return {"status": "unsupported", "trace_id": trace_id}

        if not query or not query.strip():
            await render_and_send(
                "Sorry, I received an empty message. Please try again.",
                from_person,
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
            detected_language=detected_language,
        )

        agent_duration = (time.time() - agent_start) * 1000
        log_agent_call(True, agent_duration, [], trace_id)

        # Send response back to WhatsApp — as a voice note (with a text
        # fallback if speech synthesis fails) when the customer sent voice,
        # matching how they reached out; text otherwise.
        if is_voice_message:
            send_success = await send_voice_reply(response, from_person, trace_id)
        else:
            send_success = await render_and_send(response, from_person, trace_id)
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

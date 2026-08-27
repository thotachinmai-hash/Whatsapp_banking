import asyncio
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
    get_interactive_reply,
    get_message_text,
    get_media_filename,
    get_media_mimetype,
    get_media_data,
    get_media_id,
    download_media,
)
from app.conversation.renderer import ResponseKind, as_structured_response, render_and_send, spoken_choices_hint
from app.conversation.responses.loan import render_loan_menu
from app.services.language import translate_text
from app.services.transcription import download_audio, transcribe_audio
from app.services.tts import synthesize_voice_note
from app.services.whatsapp import send_voice_message, send_document_message
from app.services.document_parser import GENERIC_DOCUMENT_SCHEMA, download_document, parse_document
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
         "id_type": "aadhaar" | "pan" | "passport" | "voter_id" | "driving_license" | "other",
         "id_number": "", "full_name": "",
         "date_of_birth": "", "address": "", "guardian_name": "",
         "applicant_name": "", "monthly_income": "", "employment_type": "",
         "requested_amount": "", "tenure_months": "", "purpose": ""}

        document_type rules:
        - "cheque" if it is a bank cheque (payee, amount, cheque number visible)
        - "kyc" if it is a government-issued identity document — an Aadhaar
          card, PAN card, passport, voter ID (EPIC), or driving licence
        - "loan_form" if it is a loan application form
        - "other" if it does not clearly match any of the above

        If document_type is "kyc", also set id_type to which of those five
        it is, and id_number to that document's own ID number (the Aadhaar
        number, PAN, passport number, EPIC/voter ID number, or driving
        licence number — whichever applies), preserving it exactly as
        printed. If it's an identity-looking document but not one of those
        five (e.g. a ration card, a bank statement), set id_type to "other".

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
            Identify which government-issued ID this is, then extract its
            details. Return ONLY valid JSON in this exact shape:
            {"id_type": "aadhaar" | "pan" | "passport" | "voter_id" |
                         "driving_license" | "other",
             "id_number": "", "full_name": "", "date_of_birth": "",
             "address": ""}

            id_type must be exactly one of those five values. Only use
            "aadhaar", "pan", "passport", "voter_id", or "driving_license"
            when the image is genuinely that document (an Aadhaar card, PAN
            card, passport, voter ID/EPIC card, or driving licence) —
            anything else (a ration card, utility bill, bank statement,
            student ID, or a blurry/unreadable image) must be "other".

            id_number is that document's own ID number — the Aadhaar
            number, PAN, passport number, EPIC/voter ID number, or driving
            licence number, whichever applies — preserved exactly as
            printed (keep the original spacing/formatting).

            Use an empty string for any field that isn't visible or can't
            be read. Do not guess, summarize, or include any text outside
            the JSON object.
            """

    return document_prompt


def _string_field(description: str) -> dict:
    return {"type": "string", "description": description}


def build_document_schema(active_workflow: dict | None) -> dict:
    """The JSON Schema counterpart to build_document_prompt(), for image
    uploads only — see app/services/document_parser.py's
    _extract_via_doc_ai. Mirrors the same (workflow, step) branches so the
    two never name a different set of fields for the same context; PDF/
    DOCX uploads keep using build_document_prompt()'s natural-language
    text instead (see parse_document()'s docstring for why)."""
    if active_workflow and active_workflow.get("step") == STEP_COLLECT_AADHAAR:
        return {
            "type": "object",
            "properties": {
                "aadhaar_number": _string_field("The 12-digit Aadhaar number, preserving the digits exactly"),
                "full_name": _string_field("Full name as printed"),
                "date_of_birth": _string_field("Date of birth as printed"),
                "address": _string_field("Address as printed"),
                "guardian_name": _string_field("Father/spouse/guardian name as printed"),
            },
        }
    if active_workflow and active_workflow.get("step") == STEP_COLLECT_PAN:
        return {
            "type": "object",
            "properties": {
                "pan_number": _string_field("The PAN in the format ABCDE1234F, preserving it exactly"),
                "full_name": _string_field("Full name as printed"),
                "date_of_birth": _string_field("Date of birth as printed"),
                "address": _string_field("Address as printed"),
                "guardian_name": _string_field("Father/spouse/guardian name as printed"),
            },
        }
    if active_workflow and active_workflow.get("type") == WORKFLOW_CHEQUE:
        return {
            "type": "object",
            "properties": {
                "bank_name": _string_field("Bank name printed on the cheque"),
                "branch": _string_field("Bank branch printed on the cheque"),
                "payee": _string_field("Who the cheque is payable to (after 'Pay')"),
                "amount_in_figures": _string_field("The cheque amount in numeric figures"),
                "amount_in_words": _string_field("The cheque amount written out in words"),
                "numbers": _string_field("The cheque number"),
                "signatory_title": _string_field("The signatory's title"),
                "date_written": _string_field("The date written on the cheque"),
                "drawer_name": _string_field("Name of the cheque's drawer/issuer"),
            },
        }
    if active_workflow and active_workflow.get("type") == WORKFLOW_LOAN:
        return {
            "type": "object",
            "properties": {
                "applicant_name": _string_field("The loan applicant's full name"),
                "monthly_income": _string_field("The applicant's monthly income"),
                "employment_type": _string_field("The applicant's employment type"),
                "requested_amount": _string_field("The loan amount requested"),
                "tenure_months": _string_field("The requested loan tenure in months"),
                "purpose": _string_field("The purpose of the loan"),
            },
        }
    if active_workflow and active_workflow.get("type") == WORKFLOW_KYC:
        return {
            "type": "object",
            "properties": {
                "id_type": {
                    "type": "string",
                    "enum": ["aadhaar", "pan", "passport", "voter_id", "driving_license", "other"],
                    "description": (
                        "Which government ID this is. Only aadhaar/pan/passport/voter_id/"
                        "driving_license when the image is genuinely that document — anything "
                        "else (a ration card, utility bill, bank statement, student ID, or a "
                        "blurry/unreadable image) is 'other'."
                    ),
                },
                "id_number": _string_field(
                    "That document's own ID number — the Aadhaar number, PAN, passport "
                    "number, EPIC/voter ID number, or driving licence number, whichever "
                    "applies — preserved exactly as printed"
                ),
                "full_name": _string_field("Full name as printed"),
                "date_of_birth": _string_field("Date of birth as printed"),
                "address": _string_field("Address as printed"),
            },
        }
    return GENERIC_DOCUMENT_SCHEMA


def _handle_bare_document_upload(
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
        result = ChequeWorkflowProcessor().handle(
            workflow=workflow, phone_number=phone_number, query="",
            parsed_document=parsed_document, trace_id=trace_id,
        )
        return result.get("response")

    if detected_type == "kyc":
        workflow = create_workflow_model(WORKFLOW_KYC, STEP_UPLOAD_KYC_FORM)
        create_workflow(phone_number, workflow)
        logger.info(f"[{trace_id}] Auto-detected KYC document upload | phone={phone_number[-4:]}")
        result = KYCWorkflowHandler().handle(
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


async def _send_receipt_if_any(response, chat_id: str, trace_id: str = "") -> None:
    """Send the generated PDF receipt (see app/services/receipts.py) a
    workflow processor attached to its success response, if any. Never
    raises and never blocks the turn — the text/voice confirmation with
    the request ID has already been sent by the time this runs, so a
    failure here just means the customer doesn't get the bonus PDF, not
    that their request silently failed."""
    structured = as_structured_response(response)
    if not structured.pdf_bytes:
        return
    try:
        sent = await send_document_message(
            chat_id, structured.pdf_bytes, structured.pdf_filename or "receipt.pdf",
            "Here's your receipt.", trace_id,
        )
        if not sent:
            logger.warning(f"[{trace_id}] Receipt PDF send failed | to={chat_id[-4:]}")
    except Exception as e:
        logger.error(f"[{trace_id}] Receipt PDF send error | error={e}")


# Maps StructuredResponse.voice_menu markers (set by
# app/agent/agent.py::_run_llm_agent) to the existing plain-text menu
# template a voice-in customer should also see. Never builds new menu
# text here — only reuses templates the corresponding text flow already
# shows elsewhere.
_VOICE_QUERY_MENUS = {
    "loan_type": render_loan_menu,
}


async def send_voice_reply(response, chat_id: str, trace_id: str = "", language: str | None = None) -> bool:
    """The voice-out half of voice-to-voice: synthesize the response text
    and send it back as a voice note, mirroring how the customer reached
    out. Falls back to a plain text reply (via render_and_send) whenever
    synthesis or sending audio fails, since a voice-in customer still
    needs a reply even if TTS is unavailable.

    `response` may be a plain string or a StructuredResponse carrying
    WhatsApp interactive buttons/list metadata (see
    app/conversation/renderer.py). The voice note itself can't carry
    tappable UI, so spoken_choices_hint() folds the button/row titles into
    the spoken sentence too (e.g. "You can say Yes, send it, or Edit
    amount.") — otherwise a voice-in customer would hear the summary with
    no indication of how to reply at all. The actual tappable
    buttons/list — or a matching plain-text menu via `voice_menu` — is
    then sent as a separate follow-up message once the voice note is out,
    so the customer isn't limited to speaking/typing the choice back.

    `language` is a fallback ISO 639-1 code (e.g. Sarvam STT's per-turn
    detection — see transcribe_audio) used only when `response` carries no
    resolved language of its own. Prefer letting `response` carry it: a
    StructuredResponse's `.language` (set by ConversationManager._finish)
    is the language `.text` was ACTUALLY translated into, which can differ
    from this turn's raw STT hint (e.g. a short "yes" voice reply has no
    detectable language of its own, but the conversation's established
    language still applies) — using the wrong one plays back audio that
    doesn't match the language of the text it's reading.
    """
    structured = as_structured_response(response)
    response_text = structured.text
    resolved_language = structured.language or language
    choices_hint = spoken_choices_hint(structured)
    if choices_hint:
        # spoken_choices_hint() is authored in English (button/list titles
        # are never translated — see ConversationManager._finish), so it
        # needs the same translation pass `.text` already went through,
        # or a Hindi/etc. reply would end with a jarring English sentence.
        if resolved_language:
            choices_hint = await translate_text(choices_hint, resolved_language, trace_id=trace_id)
        response_text = f"{response_text} {choices_hint}"
    synthesized = await synthesize_voice_note(response_text, trace_id=trace_id, language=resolved_language)
    voice_sent = False
    if synthesized is None:
        logger.info(f"[{trace_id}] Voice reply unavailable — falling back to text")
        sent = await render_and_send(response, chat_id, trace_id)
    else:
        audio_bytes, mimetype = synthesized
        voice_sent = await send_voice_message(chat_id, audio_bytes, mimetype, trace_id)
        if not voice_sent:
            logger.info(f"[{trace_id}] Voice send failed — falling back to text")
            sent = await render_and_send(response, chat_id, trace_id)
        else:
            sent = True

    # The voice note only speaks `response_text` — a customer can't tap or
    # read a menu from audio alone. When audio actually went out, send any
    # existing menu this reply carries as its own follow-up message: an
    # interactive list/buttons (e.g. the main menu) as-is, or a matching
    # plain-text menu (e.g. the loan type list) via voice_menu. The text
    # fallback above already sent `response` — interactive kind and all —
    # so it's skipped there to avoid sending the same menu twice.
    if voice_sent:
        if structured.kind in (ResponseKind.LIST, ResponseKind.BUTTONS):
            await render_and_send(structured, chat_id, trace_id)
        elif structured.voice_menu:
            menu_render = _VOICE_QUERY_MENUS.get(structured.voice_menu)
            if menu_render:
                await render_and_send(menu_render(), chat_id, trace_id)
    return sent


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
                # A voice-in customer gets a voice reply here too — the
                # language isn't known yet (nothing was transcribed), so
                # this speaks in the default (English).
                await send_voice_reply(
                    error_templates.render_voice_unavailable_error(),
                    from_person,
                    trace_id
                )
                return {"status": "error", "trace_id": trace_id}

            # Transcribe with Sarvam STT
            transcribe_start = time.time()
            query, detected_language = await transcribe_audio(audio_data, trace_id)
            transcribe_duration = (time.time() - transcribe_start) * 1000

            if not query:
                log_transcription(False, transcribe_duration, trace_id)
                await send_voice_reply(
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

            active_workflow = await asyncio.to_thread(get_workflow, phone_number)
            document_prompt = build_document_prompt(active_workflow, filename)
            document_schema = build_document_schema(active_workflow)

            parsed_document = await parse_document(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=mime_type,
                prompt=document_prompt,
                trace_id=trace_id,
                schema=document_schema,
            )

            logger.info(
                f"[{trace_id}] parse_document response={parsed_document}"
            )
            if not parsed_document.get("success"):

                logger.error(
                    f"[{trace_id}] Document parsing failed | error={parsed_document.get('error')}"
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

            if not active_workflow and await asyncio.to_thread(get_customer_by_phone, phone_number):
                detected_type = detect_workflow_type(parsed_document.get("content", {}))
                if detected_type:
                    auto_response = await asyncio.to_thread(
                        _handle_bare_document_upload,
                        detected_type, phone_number, parsed_document, trace_id
                    )
                    if auto_response is not None:
                        send_start = time.time()
                        send_success = await render_and_send(auto_response, from_person, trace_id)
                        send_duration = (time.time() - send_start) * 1000
                        log_whatsapp_send(send_success, trace_id, send_duration)
                        await _send_receipt_if_any(auto_response, from_person, trace_id)
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
            is_voice=is_voice_message,
        )

        agent_duration = (time.time() - agent_start) * 1000
        log_agent_call(True, agent_duration, [], trace_id)

        # Send response back to WhatsApp — as a voice note (with a text
        # fallback if speech synthesis fails) when the customer sent voice,
        # matching how they reached out; text otherwise.
        send_start = time.time()
        if is_voice_message:
            send_success = await send_voice_reply(response, from_person, trace_id, language=detected_language)
        else:
            send_success = await render_and_send(response, from_person, trace_id)
        send_duration = (time.time() - send_start) * 1000
        log_whatsapp_send(send_success, trace_id, send_duration)

        # A successful cheque/loan/KYC/transfer submission carries a
        # generated PDF receipt (see app/services/receipts.py) — send it
        # as a document right after the confirmation, voice or text alike.
        await _send_receipt_if_any(response, from_person, trace_id)

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

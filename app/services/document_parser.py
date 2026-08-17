import asyncio
import json
import mimetypes
import os
import tempfile
import time

import httpx
from dotenv import load_dotenv
from app.services.sarvam_client import get_sarvam_client
from docx import Document
from pypdf import PdfReader

from app.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

CHAT_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")

# Images go through Sarvam's Document Intelligence API (doc_ai.extract) —
# an async job (submit -> poll status -> fetch results), not the chat/
# vision completions endpoint. Sarvam's public chat-completions API
# rejects image_url content blocks on this account/model ("Input should
# be a valid string") even though the SDK's request schema declares
# support for them, so that path never worked; doc_ai.extract is the
# real, working one, confirmed against a live cheque image (correct
# field extraction + document-type classification in ~2-5s for a single
# page).
_DOC_AI_POLL_INTERVAL_SECONDS = 1.0
_DOC_AI_POLL_MAX_ATTEMPTS = 25  # ~25s cap on a single document
_DOC_AI_TERMINAL_STATUSES = {"completed", "partially_completed", "failed", "rejected"}

# Fallback schema for callers (e.g. the /api/test/document debug endpoint)
# that don't supply their own — the same fields the "cold upload, no
# active workflow" classify-and-extract prompt covers, so a generic
# upload with no context still gets a reasonable best-effort extraction.
GENERIC_DOCUMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["cheque", "kyc", "loan_form", "other"],
            "description": "What kind of document this is",
        },
        "bank_name": {"type": "string", "description": "Bank name printed on a cheque"},
        "branch": {"type": "string", "description": "Bank branch printed on a cheque"},
        "payee": {"type": "string", "description": "Who a cheque is payable to"},
        "amount_in_figures": {"type": "string", "description": "Cheque amount in numeric figures"},
        "amount_in_words": {"type": "string", "description": "Cheque amount written out in words"},
        "numbers": {"type": "string", "description": "The cheque number"},
        "signatory_title": {"type": "string", "description": "Signatory's title on a cheque"},
        "date_written": {"type": "string", "description": "Date written on a cheque"},
        "drawer_name": {"type": "string", "description": "Name of the cheque's drawer/issuer"},
        "id_type": {
            "type": "string",
            "enum": ["aadhaar", "pan", "passport", "voter_id", "driving_license", "other"],
            "description": "Which government ID this is, if it is a KYC identity document",
        },
        "id_number": {"type": "string", "description": "The ID number on a government identity document"},
        "full_name": {"type": "string", "description": "Full name as printed on an identity document"},
        "date_of_birth": {"type": "string", "description": "Date of birth as printed"},
        "address": {"type": "string", "description": "Address as printed"},
        "guardian_name": {"type": "string", "description": "Father/spouse/guardian name as printed"},
        "applicant_name": {"type": "string", "description": "Loan applicant's full name"},
        "monthly_income": {"type": "string", "description": "Applicant's monthly income"},
        "employment_type": {"type": "string", "description": "Applicant's employment type"},
        "requested_amount": {"type": "string", "description": "Loan amount requested"},
        "tenure_months": {"type": "string", "description": "Requested loan tenure in months"},
        "purpose": {"type": "string", "description": "Purpose of the loan"},
    },
}


async def _extract_via_doc_ai(file_bytes: bytes, filename: str, mime_type: str, schema: dict) -> dict:
    """Submit an image/PDF to Sarvam's Document Intelligence extract job,
    poll until it reaches a terminal status, and return the extracted
    fields (missing/unreadable fields as "" rather than None, matching
    the empty-string convention every workflow processor already checks
    with `_is_present`)."""
    client = get_sarvam_client()
    job = await asyncio.to_thread(
        client.doc_ai.extract,
        file=[(filename or "document", file_bytes, mime_type)],
        schema=json.dumps(schema),
    )

    status = job.status
    for _ in range(_DOC_AI_POLL_MAX_ATTEMPTS):
        if status in _DOC_AI_TERMINAL_STATUSES:
            break
        await asyncio.sleep(_DOC_AI_POLL_INTERVAL_SECONDS)
        status = (await asyncio.to_thread(client.doc_ai.get_status, job.job_id)).status
    else:
        raise TimeoutError(f"Document AI job {job.job_id} did not finish within {_DOC_AI_POLL_MAX_ATTEMPTS}s")

    if status in ("failed", "rejected"):
        raise RuntimeError(f"Document AI job {job.job_id} ended with status={status}")

    results = await asyncio.to_thread(client.doc_ai.get_results, job.job_id)
    return {key: ("" if value is None else value) for key, value in (results.result or {}).items()}


async def download_document(
    media_url: str,
    api_key: str,
    trace_id: str
) -> bytes | None:
    """
    Download any uploaded document from OpenWA.

    Supports:
        - Images
        - PDF
        - DOCX

    Returns:
        bytes if successful
        None otherwise.
    """

    start = time.time()

    try:

        async with httpx.AsyncClient() as client:

            response = await client.get(
                media_url,
                headers={
                    "X-API-Key": api_key
                },
                timeout=60.0
            )

        duration = (time.time() - start) * 1000

        if response.status_code == 200:

            logger.info(
                f"[{trace_id}] "
                f"Document downloaded | "
                f"size={len(response.content)} bytes | "
                f"duration={duration:.2f}ms"
            )

            return response.content

        logger.error(
            f"[{trace_id}] "
            f"Document download failed | "
            f"status={response.status_code} | "
            f"duration={duration:.2f}ms"
        )

        return None

    except Exception as e:

        duration = (time.time() - start) * 1000

        logger.error(
            f"[{trace_id}] "
            f"Document download error | "
            f"error={e} | "
            f"duration={duration:.2f}ms"
        )

        return None


async def parse_document(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    prompt: str,
    trace_id: str,
    schema: dict | None = None,
) -> dict:
    """
    Generic document parser.

    Supports:
        • Images
        • PDF
        • DOCX

    `schema` is a JSON Schema dict (used only for images, via Sarvam's
    Document Intelligence extract job — see _extract_via_doc_ai) naming
    exactly which fields to pull out and what each means; falls back to
    GENERIC_DOCUMENT_SCHEMA when the caller doesn't have workflow context
    to build a narrower one from (e.g. the /api/test/document endpoint).
    `prompt` is unused for images — PDF/DOCX still go through it via a
    plain chat completion over their already-extracted text.
    """

    start = time.time()

    try:

        if not mime_type:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        if mime_type.startswith("image/"):

            parsed = await _parse_image(
                file_bytes,
                mime_type,
                filename,
                schema or GENERIC_DOCUMENT_SCHEMA,
            )

        elif mime_type == "application/pdf":

            parsed = await _parse_pdf(
                file_bytes,
                filename,
                prompt
            )

        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

            parsed = await _parse_docx(
                file_bytes,
                prompt
            )

        else:

            return {
                "success": False,
                "error": f"Unsupported file type: {mime_type}"
            }

        duration = (time.time() - start) * 1000

        logger.info(
            f"[{trace_id}] "
            f"Document parsed | "
            f"{filename} | "
            f"duration={duration:.2f}ms"
        )

        return {
            "success": True,
            "filename": filename,
            "mime_type": mime_type,
            "content": parsed
        }

    except Exception as e:

        duration = (time.time() - start) * 1000

        logger.error(
            f"[{trace_id}] "
            f"Document parsing failed | "
            f"{e} | "
            f"duration={duration:.2f}ms"
        )

        return {
            "success": False,
            "error": str(e)
        }


async def _parse_image(
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    schema: dict,
) -> dict:
    return await _extract_via_doc_ai(file_bytes, filename, mime_type, schema)


async def _parse_pdf(
    file_bytes: bytes,
    filename: str,
    prompt: str
) -> dict:

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    try:
        reader = PdfReader(temp_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    response = get_sarvam_client().chat.completions(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=1500,
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": f"""
{prompt}

Document Content:

{text}
"""
            }
        ]
    )

    return _parse_llm_response(
        response.choices[0].message.content
    )


async def _parse_docx(
    file_bytes: bytes,
    prompt: str
) -> dict:

    with tempfile.NamedTemporaryFile(
        suffix=".docx",
        delete=False
    ) as temp:

        temp.write(file_bytes)
        temp_path = temp.name

    try:

        document = Document(temp_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    response = get_sarvam_client().chat.completions(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=1500,
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": f"""
{prompt}

Document Content:

{text}
"""
            }
        ]
    )

    return _parse_llm_response(
        response.choices[0].message.content
    )


def _parse_llm_response(
    response_text: str
) -> dict:

    response_text = response_text.strip()

    try:
        return json.loads(response_text)

    except Exception:
        pass


    # Remove Qwen thinking section
    if "<think>" in response_text:

        response_text = response_text.split(
            "</think>"
        )[-1].strip()


    # Remove markdown json block

    if "```json" in response_text:

        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )


    try:
        return json.loads(response_text)

    except Exception:

        return {
            "raw_text": response_text
        }

import base64
import json
import mimetypes
import os
import tempfile
import time

import httpx
from dotenv import load_dotenv
from groq import Groq
from docx import Document

from app.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27B"
)


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
    trace_id: str
) -> dict:
    """
    Generic document parser.

    Supports:
        • Images
        • PDF
        • DOCX
    """

    start = time.time()

    try:

        if not mime_type:
            mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        if mime_type.startswith("image/"):

            parsed = await _parse_image(
                file_bytes,
                mime_type,
                prompt
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
    prompt: str
) -> dict:

    image = base64.b64encode(file_bytes).decode()

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image}"
                        }
                    }
                ]
            }
        ]
    )

    return _parse_llm_response(
        response.choices[0].message.content
    )


async def _parse_pdf(
    file_bytes: bytes,
    filename: str,
    prompt: str
) -> dict:

    pdf = base64.b64encode(file_bytes).decode()

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": f"data:application/pdf;base64,{pdf}"
                        }
                    }
                ]
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

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
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

"""PDF receipt generation for successfully submitted requests (cheque
deposit, loan application, KYC update, money transfer) — sent as a
WhatsApp document alongside the existing text confirmation, so a
completed request feels like a real bank issuing a receipt.

SENSITIVE-DATA RULE: same as the text summaries this mirrors (see
app/conversation/responses/{cheque,loan,kyc,transfer}.py) — callers must
never pass a raw Aadhaar/PAN/ID number, account number, OTP, or similar
into `fields`; pass "Provided ✅" or a masked value instead. This module
renders whatever it's given as-is.
"""

import io
import re
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from app.conversation.renderer import StructuredResponse
from app.logger import get_logger

logger = get_logger(__name__)

BANK_NAME = "FINACLE BANK"
_PAGE_WIDTH, _PAGE_HEIGHT = A4
_MARGIN = 20 * mm


def generate_receipt_pdf(document_title: str, request_id: str, fields: list[tuple[str, str]]) -> bytes | None:
    """Render a one-page PDF receipt: bank letterhead, document title,
    request ID, a generated timestamp, then each (label, value) in
    `fields` as its own line. Never raises — a receipt is a nice-to-have
    on top of the text confirmation that already carries the request ID,
    so a rendering failure must never block the actual confirmation.
    Returns None on failure.
    """
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        y = _PAGE_HEIGHT - _MARGIN

        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(_MARGIN, y, BANK_NAME)
        y -= 8 * mm
        pdf.setLineWidth(1)
        pdf.line(_MARGIN, y, _PAGE_WIDTH - _MARGIN, y)
        y -= 12 * mm

        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(_MARGIN, y, document_title)
        y -= 10 * mm

        pdf.setFont("Helvetica", 10)
        generated_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        pdf.drawString(_MARGIN, y, f"Generated: {generated_at}")
        y -= 10 * mm

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(_MARGIN, y, f"Request ID: {request_id}")
        y -= 12 * mm

        pdf.setLineWidth(0.5)
        pdf.line(_MARGIN, y, _PAGE_WIDTH - _MARGIN, y)
        y -= 10 * mm

        label_x = _MARGIN
        value_x = _MARGIN + 55 * mm
        for label, value in fields:
            if y < _MARGIN:
                pdf.showPage()
                y = _PAGE_HEIGHT - _MARGIN
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(label_x, y, f"{label}:")
            pdf.setFont("Helvetica", 10)
            pdf.drawString(value_x, y, str(value) if value else "—")
            y -= 8 * mm

        y -= 4 * mm
        pdf.setLineWidth(0.5)
        pdf.line(_MARGIN, y, _PAGE_WIDTH - _MARGIN, y)
        y -= 8 * mm
        pdf.setFont("Helvetica-Oblique", 8)
        pdf.drawString(
            _MARGIN, y,
            "This is a system-generated receipt and does not require a signature.",
        )

        pdf.save()
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Receipt PDF generation failed | title={document_title} | request_id={request_id} | error={e}")
        return None


def build_receipt_response(
    text: str, document_title: str, request_id: str, fields: list[tuple[str, str]]
) -> StructuredResponse:
    """The text confirmation, with a generated PDF receipt attached (see
    StructuredResponse.pdf_bytes) — the one call site every workflow
    processor's success path uses. Falls back to a plain text-only
    response if PDF generation fails; the customer must never lose their
    request-ID confirmation just because the receipt couldn't be built."""
    pdf_bytes = generate_receipt_pdf(document_title, request_id, fields)
    if pdf_bytes is None:
        return StructuredResponse.plain(text)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", request_id) or "receipt"
    return StructuredResponse(text=text, pdf_bytes=pdf_bytes, pdf_filename=f"{safe_id}.pdf")

"""Response Renderer — Task 10, Part 1.

Hides OpenWA-specific delivery details from every banking workflow /
ConversationManager / guidance layer. Those layers only ever produce
already-rendered text (via app/conversation/responses/*, the centralized
template layer from Phase 4) or a StructuredResponse naming which
template produced it — never an OpenWA HTTP request, never a chat-ID
format, never a session-ID. This module is the single choke point that
turns either into an actual outbound WhatsApp send.

    Banking Workflow -> Structured Response -> Response Renderer -> OpenWA -> WhatsApp

No new OpenWA client is created here — app/services/whatsapp.py already
owns the only OpenWA HTTP client (`send_text_message`) and this module
calls it, exactly as message_handler.py/main.py already did before this
task. What changes is *where* that call happens: previously every
call site (main.py's error paths, message_handler.py's error paths and
final response) called `send_text_message` directly; now they all go
through `render_and_send()`.

Supported response kinds: TEXT, TEMPLATE, BUTTONS, and LIST. TEXT/TEMPLATE
render identically — as plain text. BUTTONS/LIST render as WhatsApp Cloud
API native interactive messages (reply buttons / list messages — see
https://developers.facebook.com/documentation/business-messaging/whatsapp/messages/interactive-reply-buttons-messages)
so a customer can tap a choice instead of typing a digit/keyword back.
Every interactive send has a plain-text fallback (`_fallback_text`) built
from the same body text + button/row titles, used whenever WhatsApp's
strict payload limits (≤3 buttons, ≤20-char button titles, ≤10 rows,
≤24-char row titles) are violated, or the interactive send itself fails
for any reason — a customer must never lose a turn just because the
richer UI couldn't be sent. The underlying text-based reply parsers
(app/workflows/processors/*, app/workflows/nlu.py) are unchanged by any
of this: a button/list tap's `id` is chosen to match a token those
parsers already accept (see docs/current_architecture.md), and a
customer can still just type a reply either way.
"""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel

from app.logger import get_logger
from app.services.whatsapp import send_button_message, send_list_message, send_text_message

logger = get_logger(__name__)

MAX_BUTTONS = 3
MAX_BUTTON_TITLE_LEN = 20
MAX_LIST_ROWS = 10
MAX_LIST_ROW_TITLE_LEN = 24
MAX_LIST_BUTTON_LABEL_LEN = 20


class ResponseKind(str, Enum):
    TEXT = "text"
    TEMPLATE = "template"
    BUTTONS = "buttons"
    LIST = "list"


class InteractiveButton(BaseModel):
    id: str
    title: str


class InteractiveListRow(BaseModel):
    id: str
    title: str
    description: str = ""


class InteractiveListSection(BaseModel):
    title: str
    rows: list[InteractiveListRow]


class StructuredResponse(BaseModel):
    """A response ready to be sent, with its provenance attached.

    `text` is always the final, already-rendered message body — this is
    also what gets translated (app/services/language.py) and logged/
    persisted (ConversationContext.last_assistant_message, session
    history), so it must stand alone as a sensible reply even when
    `buttons`/`list_sections` can't be sent for some reason. `kind` says
    which interactive shape (if any) to attach on top of that body text.
    """

    kind: ResponseKind = ResponseKind.TEXT
    text: str
    template_name: Optional[str] = None
    buttons: list[InteractiveButton] = []
    list_button_label: Optional[str] = None
    list_sections: list[InteractiveListSection] = []

    @classmethod
    def template(cls, text: str, template_name: str) -> "StructuredResponse":
        return cls(kind=ResponseKind.TEMPLATE, text=text, template_name=template_name)

    @classmethod
    def plain(cls, text: str) -> "StructuredResponse":
        return cls(kind=ResponseKind.TEXT, text=text)

    @classmethod
    def buttons_of(cls, text: str, buttons: list[InteractiveButton]) -> "StructuredResponse":
        return cls(kind=ResponseKind.BUTTONS, text=text, buttons=buttons)

    @classmethod
    def list_of(
        cls, text: str, list_button_label: str, list_sections: list[InteractiveListSection]
    ) -> "StructuredResponse":
        return cls(
            kind=ResponseKind.LIST, text=text,
            list_button_label=list_button_label, list_sections=list_sections,
        )


ResponseLike = Union[str, StructuredResponse]


def as_structured_response(response: ResponseLike) -> StructuredResponse:
    if isinstance(response, StructuredResponse):
        return response
    return StructuredResponse.plain(str(response))


def _fits_button_limits(structured: StructuredResponse) -> bool:
    return (
        0 < len(structured.buttons) <= MAX_BUTTONS
        and all(len(b.title) <= MAX_BUTTON_TITLE_LEN for b in structured.buttons)
    )


def _fits_list_limits(structured: StructuredResponse) -> bool:
    if not structured.list_button_label or len(structured.list_button_label) > MAX_LIST_BUTTON_LABEL_LEN:
        return False
    rows = [row for section in structured.list_sections for row in section.rows]
    return (
        0 < len(rows) <= MAX_LIST_ROWS
        and all(len(row.title) <= MAX_LIST_ROW_TITLE_LEN for row in rows)
    )


def _flattened_fallback_text(structured: StructuredResponse) -> str:
    """A plain numbered-text rendering of the same choices, used whenever
    the interactive shape can't be sent — matches this app's pre-buttons
    UX exactly, so nothing is lost, only the tap-to-reply convenience."""
    if structured.kind == ResponseKind.BUTTONS:
        options = "\n".join(f"{i+1}. {b.title}" for i, b in enumerate(structured.buttons))
        return f"{structured.text}\n\n{options}"
    if structured.kind == ResponseKind.LIST:
        lines = []
        for section in structured.list_sections:
            for row in section.rows:
                lines.append(f"{row.id}. {row.title}" if row.id.isdigit() else f"- {row.title}")
        return f"{structured.text}\n\n" + "\n".join(lines)
    return structured.text


async def render_and_send(response: ResponseLike, phone_number: str, trace_id: str) -> bool:
    """The only function in this codebase that should call
    app.services.whatsapp.send_* directly outside of this module itself.

    Never raises: a delivery failure is logged (safely — no OpenWA
    response body/headers, no stack trace) and returns False, matching
    send_text_message()'s own existing contract. Conversation state is
    untouched either way — ConversationManager/the workflow that produced
    `response` has already persisted its own state by the time this runs.
    """
    structured = as_structured_response(response)
    try:
        if structured.kind == ResponseKind.BUTTONS and _fits_button_limits(structured):
            sent = await send_button_message(
                phone_number, structured.text,
                [b.model_dump() for b in structured.buttons], trace_id,
            )
        elif structured.kind == ResponseKind.LIST and _fits_list_limits(structured):
            sent = await send_list_message(
                phone_number, structured.text, structured.list_button_label,
                [s.model_dump() for s in structured.list_sections], trace_id,
            )
        else:
            if structured.kind in (ResponseKind.BUTTONS, ResponseKind.LIST):
                logger.warning(
                    f"[{trace_id}] response.render.interactive_limits_exceeded | "
                    f"phone={phone_number[-4:]} | kind={structured.kind.value} — falling back to text"
                )
            sent = await send_text_message(phone_number, _flattened_fallback_text(structured), trace_id)
    except Exception as e:
        logger.error(
            f"[{trace_id}] response.render.send_failed | phone={phone_number[-4:]} | "
            f"kind={structured.kind.value} | error={e}"
        )
        if structured.kind in (ResponseKind.BUTTONS, ResponseKind.LIST):
            try:
                sent = await send_text_message(phone_number, _flattened_fallback_text(structured), trace_id)
            except Exception as fallback_error:
                logger.error(
                    f"[{trace_id}] response.render.fallback_send_failed | phone={phone_number[-4:]} | "
                    f"error={fallback_error}"
                )
                return False
        else:
            return False

    logger.info(
        f"[{trace_id}] response.render.sent | phone={phone_number[-4:]} | "
        f"kind={structured.kind.value}"
        + (f" | template={structured.template_name}" if structured.template_name else "")
        + f" | success={sent}"
    )
    return sent

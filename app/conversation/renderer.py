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

Supported response kinds (per this task's explicit scope): TEXT and
TEMPLATE. Both currently render identically — as plain text — because
the actual OpenWA capability inspection for this task (see
docs/current_architecture.md, "Response UX Architecture — Phase 8")
found no interactive button/list mechanism worth adopting; TEMPLATE exists
so callers can express "this text came from a centralized template
function" (for logging/observability, and so a future interactive
capability could special-case it) without every call site needing to
change again later.
"""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel

from app.logger import get_logger
from app.services.whatsapp import send_text_message

logger = get_logger(__name__)


class ResponseKind(str, Enum):
    TEXT = "text"
    TEMPLATE = "template"


class StructuredResponse(BaseModel):
    """A response ready to be sent, with its provenance attached.

    `text` is always the final, already-rendered message — this class
    does not defer rendering or accept a template name to look up later;
    every template in app/conversation/responses/* already renders to a
    plain string when called, and re-deriving that from a name+vars pair
    here would just be a second, parallel template system. `kind` is
    metadata only, for logging/observability.
    """

    kind: ResponseKind = ResponseKind.TEXT
    text: str
    template_name: Optional[str] = None

    @classmethod
    def template(cls, text: str, template_name: str) -> "StructuredResponse":
        return cls(kind=ResponseKind.TEMPLATE, text=text, template_name=template_name)

    @classmethod
    def plain(cls, text: str) -> "StructuredResponse":
        return cls(kind=ResponseKind.TEXT, text=text)


ResponseLike = Union[str, StructuredResponse]


def as_structured_response(response: ResponseLike) -> StructuredResponse:
    if isinstance(response, StructuredResponse):
        return response
    return StructuredResponse.plain(str(response))


async def render_and_send(response: ResponseLike, phone_number: str, trace_id: str) -> bool:
    """The only function in this codebase that should call
    app.services.whatsapp.send_text_message directly outside of this
    module itself.

    Never raises: a delivery failure is logged (safely — no OpenWA
    response body/headers, no stack trace) and returns False, matching
    send_text_message()'s own existing contract. Conversation state is
    untouched either way — ConversationManager/the workflow that produced
    `response` has already persisted its own state by the time this runs.
    """
    structured = as_structured_response(response)
    try:
        sent = await send_text_message(phone_number, structured.text, trace_id)
    except Exception as e:
        logger.error(
            f"[{trace_id}] response.render.send_failed | phone={phone_number[-4:]} | "
            f"kind={structured.kind.value} | error={e}"
        )
        return False

    logger.info(
        f"[{trace_id}] response.render.sent | phone={phone_number[-4:]} | "
        f"kind={structured.kind.value}"
        + (f" | template={structured.template_name}" if structured.template_name else "")
        + f" | success={sent}"
    )
    return sent

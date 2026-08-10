"""User-facing error templates.

These never accept or echo a raw exception, stack trace, database driver
error, Redis error, or Groq/API error — every function here takes at most
a short, already-safe hint string. If the caller only has a raw exception,
it should log that (with the trace ID, as the app already does everywhere)
and call one of these with no argument instead of formatting the
exception into user-facing text.
"""

from typing import Optional


def render_invalid_input(hint: Optional[str] = None) -> str:
    base = "I couldn't verify that. Please try again."
    return f"{base} {hint}" if hint else base


def render_missing_information(fields: Optional[list[str]] = None) -> str:
    if fields:
        return f"I still need: {', '.join(fields)}. Please provide these to continue."
    return "I still need a bit more information to continue."


def render_document_error() -> str:
    return "I couldn't read that document clearly. Please upload another clear photo."


def render_transcription_error() -> str:
    return "Sorry, I couldn't understand your voice message. Please try again or send a text message."


def render_voice_unavailable_error() -> str:
    return "Sorry, I couldn't access that voice message. Please try again or type your request."


def render_service_error() -> str:
    return "I couldn't complete that request right now. Please try again shortly."


def render_rate_limit_error() -> str:
    return "I'm sorry, the service is temporarily busy. Please try again in a few minutes."


def render_database_error() -> str:
    return "I couldn't save that right now. Please try again shortly."


def render_unknown_error() -> str:
    return "Something went wrong on my end. Please try again, or contact support if this continues."


def render_agent_error() -> str:
    """The final safety-net response for an unhandled failure during a
    conversation turn (see app/conversation/manager.py's outer except) —
    kept as its own function, with its pre-existing exact wording, rather
    than reusing render_unknown_error()'s different phrasing."""
    return "I'm sorry, I encountered an error processing your request. Please try again."

"""Conversation-level context model.

This is a conversation-orchestration abstraction, not a banking data store.
`workflow:{phone_number}` (see app/workflows/memory.py) remains the source
of truth for workflow execution; ConversationContext only mirrors the
non-sensitive parts of it for conversation-layer use.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

# Workflow-data keys that must never be copied into ConversationContext.
# ConversationContext is for conversational orchestration, not banking PII
# storage — see docs/current_architecture.md, "Conversation Context — Phase 1".
SENSITIVE_WORKFLOW_KEYS = {
    "aadhaar_number",
    "pan_number",
    "id_number",
    "otp",
    "password",
    "api_key",
    "account_number",
    "beneficiary_account",
    "source_account",
    "card_number",
    "cvv",
    "pin",
}


def sanitize_workflow_data(data: dict[str, Any] | None) -> dict[str, Any]:
    """Drop known-sensitive fields before they ever reach ConversationContext."""
    if not data:
        return {}
    return {key: value for key, value in data.items() if key not in SENSITIVE_WORKFLOW_KEYS}


class ConversationContext(BaseModel):
    """Represents one customer's current conversation.

    Deliberately thin: it does not duplicate the full workflow JSON
    (`workflow:{phone_number}`) or the message history
    (`session:{phone_number}`) — both remain the source of truth for their
    respective concerns. This model exists so future conversation-layer
    logic has one place to read/write conversation-level metadata instead
    of reaching into workflow/session state directly.
    """

    phone_number: str
    customer_id: Optional[int] = None
    is_registered: bool = False

    current_workflow: Optional[str] = None
    current_step: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_data: dict[str, Any] = Field(default_factory=dict)

    last_intent: Optional[str] = None
    intent_confidence: Optional[float] = None

    # ISO 639-1 codes — see app/services/language.py. Voice and text each
    # get their own sticky language, tracked independently: a language
    # established on a voice call must never carry over to a text reply
    # (or vice versa) — each channel is answered in whatever language THAT
    # channel is currently using. See ConversationManager._update_language.
    voice_language: str = "en"
    text_language: str = "en"

    # The language resolved for THIS turn's response — always a copy of
    # either voice_language or text_language depending on which channel
    # the current message came in on (set fresh every turn by
    # ConversationManager._update_language). Persisted so a short reply
    # that carries no detectable language signal of its own still gets
    # answered in whatever language its own channel already established,
    # rather than silently reverting to English.
    detected_language: str = "en"

    last_user_message: Optional[str] = None
    last_assistant_message: Optional[str] = None
    conversation_summary: Optional[str] = None

    last_error: Optional[str] = None
    retry_count: int = 0

    pending_action: Optional[str] = None
    allowed_actions: list[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def touch(self) -> None:
        """Refresh updated_at. Called by the store immediately before persisting."""
        self.updated_at = datetime.utcnow().isoformat()

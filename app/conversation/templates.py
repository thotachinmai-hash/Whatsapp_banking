"""Compatibility wrapper — Phase 4 moved these into
app/conversation/responses/common.py (see docs/current_architecture.md,
"Response Template Layer — Phase 4"). Kept so existing imports
(app/agent/agent.py) don't need to change. New code should import from
app.conversation.responses directly.
"""

from app.conversation.responses.common import (
    render_clarification,
    render_low_confidence,
    render_out_of_scope,
)
from app.conversation.responses.common import render_unsupported_request as render_unsupported_banking_request

__all__ = [
    "render_out_of_scope",
    "render_clarification",
    "render_low_confidence",
    "render_unsupported_banking_request",
]

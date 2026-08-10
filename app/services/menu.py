"""Compatibility wrapper — Phase 4 moved these into
app/conversation/responses/ (see docs/current_architecture.md, "Response
Template Layer — Phase 4"). Kept so existing callers (registration_gate.py,
workflows/manager.py, workflows/processors/onboarding.py) don't need to
change. New code should import from app.conversation.responses directly.
"""

from app.conversation.responses.onboarding import render_onboarding_welcome as build_onboarding_welcome_message
from app.conversation.responses.common import render_main_menu as build_menu_response
from app.conversation.responses.status import render_account_summary as build_accounts_summary

__all__ = ["build_onboarding_welcome_message", "build_menu_response", "build_accounts_summary"]

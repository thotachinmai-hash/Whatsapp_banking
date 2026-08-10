"""Centralized response/template layer — Phase 4.

BUSINESS LOGIC → response template → consistent WhatsApp message.

Every module in this package formats presentation text only. None of them
execute a transaction, query a database, change workflow state, decide
loan eligibility, or authorize a financial action — see
docs/current_architecture.md, "Response Template Layer — Phase 4".

Import the domain submodule you need, e.g.:

    from app.conversation.responses import common, transfer, loan
    common.render_main_menu(name)
    transfer.render_transfer_summary(...)
"""

from app.conversation.responses import (
    cheque,
    common,
    errors,
    kyc,
    loan,
    onboarding,
    status,
    transfer,
)

__all__ = ["common", "onboarding", "transfer", "loan", "cheque", "kyc", "status", "errors"]

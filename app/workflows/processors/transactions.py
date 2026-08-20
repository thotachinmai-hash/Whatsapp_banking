"""Deterministic "View transactions" menu flow.

Mirrors the transfer workflow's source-account picker
(app/workflows/processors/transfer.py::_source_prompt/_source_account):
zero accounts gets a plain message, exactly one account skips straight to
its last 5 transactions with no question asked, and 2+ accounts show a
tappable list first. This exists so tapping "View transactions" on the
main menu is a deterministic UI flow instead of depending on the LLM+tools
agent to decide to call get_last_transactions on its own (see tools.py's
tool_get_last_transactions) — that path was confirmed live to sometimes
answer "no account linked" without calling the tool at all.
"""

from typing import Any

from app.database import get_account_by_number, get_accounts_by_phone, get_transactions
from app.logger import get_logger
from app.workflows.constants import STEP_SELECT_TRANSACTIONS_ACCOUNT, WORKFLOW_VIEW_TRANSACTIONS
from app.workflows.memory import complete_workflow, create_workflow, create_workflow_model
from app.conversation.responses.common import format_currency, with_nav_buttons
from app.conversation.renderer import InteractiveListRow, InteractiveListSection, StructuredResponse

logger = get_logger(__name__)

_TRANSACTIONS_LIMIT = 5


def _render_transactions(account: dict, trace_id: str = "") -> StructuredResponse:
    # get_accounts_by_phone's row has no "id" column — re-fetch the full
    # account record (which does) before querying its ledger.
    full_account = get_account_by_number(account["account_number"]) or account
    transactions = get_transactions(full_account["id"], limit=_TRANSACTIONS_LIMIT)
    currency = account.get("currency", "INR")
    label = f"{str(account['account_type']).title()} · {account['account_number']}"
    logger.info(
        f"[{trace_id}] View-transactions rendered | account={account['account_number']} | count={len(transactions)}"
    )
    if not transactions:
        return with_nav_buttons(f"\U0001F4C4 *{label}*\n\nNo transactions found on this account yet.")

    lines = [f"\U0001F4C4 *Last {len(transactions)} transactions — {label}*\n"]
    for t in transactions:
        sign = "+" if t["transaction_type"] == "credit" else "-"
        amount_label = format_currency(t["amount"], currency)
        date_label = str(t["created_at"])[:10]
        lines.append(f"{date_label} | {sign}{amount_label} | {t['description']}")
    return with_nav_buttons("\n".join(lines))


def _account_prompt(accounts: list[dict]) -> StructuredResponse:
    rows = [
        InteractiveListRow(
            id=f"vtxn_{index}", title=f"{str(a['account_type']).title()} · {a['account_number']}"[:24],
            description=f"Balance {format_currency(a['balance'], a.get('currency', 'INR'))}",
        )
        for index, a in enumerate(accounts, 1)
    ]
    return StructuredResponse.list_of(
        "\U0001F4C4 Which account would you like to see transactions for?",
        "Choose account", [InteractiveListSection(title="Your accounts", rows=rows)],
    )


def start_view_transactions(phone_number: str, trace_id: str = "") -> dict[str, Any]:
    """Entry point called from WorkflowManager.start_requested() for the
    "View transactions" menu row / keyword trigger."""
    accounts = get_accounts_by_phone(phone_number)
    if not accounts:
        return {
            "handled": True,
            "response": (
                "It looks like there's no active account linked to your mobile number yet. "
                "You can link one through the app or by visiting a branch."
            ),
        }
    if len(accounts) == 1:
        return {"handled": True, "response": _render_transactions(accounts[0], trace_id)}

    workflow = create_workflow_model(WORKFLOW_VIEW_TRANSACTIONS, STEP_SELECT_TRANSACTIONS_ACCOUNT)
    create_workflow(phone_number, workflow)
    return {"handled": True, "response": _account_prompt(accounts)}


class ViewTransactionsWorkflowHandler:
    """Single-step handler: resolve the tapped/typed account choice, then
    complete the workflow after showing its last 5 transactions."""

    async def handle(self, workflow: dict[str, Any], phone_number: str, query: str, trace_id: str = "") -> dict[str, Any]:
        # Not persisted in the workflow record itself — Decimal balances
        # aren't JSON-serializable for Redis storage (see create_workflow),
        # so re-fetch fresh instead, same as transfer.py's own fallback.
        accounts = get_accounts_by_phone(phone_number)
        text = query.strip()
        if text.lower().startswith("vtxn_"):
            text = text[len("vtxn_"):]
        choice = text.strip().lower()

        account = None
        if choice.isdigit() and 1 <= int(choice) <= len(accounts):
            account = accounts[int(choice) - 1]
        if account is None:
            account = next((a for a in accounts if a["account_number"].lower() == choice), None)
        if account is None:
            type_matches = [a for a in accounts if str(a["account_type"]).lower() in choice]
            if len(type_matches) == 1:
                account = type_matches[0]

        if not account:
            return {"handled": True, "response": _account_prompt(accounts)}

        complete_workflow(phone_number)
        return {"handled": True, "response": _render_transactions(account, trace_id)}

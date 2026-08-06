import time
from app.database import (
    get_account_by_number,
    get_transactions,
    get_spend_summary,
    get_cheque_request_by_id,
)
from app.logger import get_logger
from app.workflows.memory import (
    create_workflow,
    create_workflow_model,
    get_workflow,
)

from app.workflows.constants import (
    WORKFLOW_CHEQUE,
    STEP_UPLOAD_CHEQUE,
)

logger = get_logger(__name__)


def tool_get_account_balance(account_number: str, trace_id: str = "") -> dict:
    """
    Get account balance for a given account number.
    Queries PostgreSQL accounts table.
    Returns balance, account type, account holder name, currency.
    """
    start = time.time()
    try:
        account = get_account_by_number(account_number.strip())
        duration = (time.time() - start) * 1000

        if not account:
            logger.info(f"[{trace_id}] TOOL | get_account_balance | not_found | account={account_number}")
            return {
                "found": False,
                "message": f"Account {account_number} not found or inactive."
            }

        logger.info(f"[{trace_id}] TOOL | get_account_balance | success | duration={duration:.2f}ms")
        return {
            "found": True,
            "account_number": account["account_number"],
            "account_holder": account["account_holder"],
            "account_type": account["account_type"],
            "balance": float(account["balance"]),
            "currency": account["currency"],
            "status": account["status"]
        }
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(f"[{trace_id}] TOOL | get_account_balance | error={e} | duration={duration:.2f}ms")
        return {"found": False, "message": f"Error retrieving account: {str(e)}"}


def tool_get_last_transactions(
    account_number: str,
    limit: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    transaction_type: str | None = None,
    category: str | None = None,
    trace_id: str = "",
) -> dict:
    """
    Get transactions for a given account number.

    By default returns the last 5 transactions. Optionally filter by
    start_date/end_date (YYYY-MM-DD), transaction_type ("credit"/"debit"),
    and/or category (e.g. "groceries", "bills", "rent", "salary",
    "transport", "entertainment", "shopping", "isa", "bonus", "interest",
    "transfer", "other").
    """
    start = time.time()
    try:
        account = get_account_by_number(account_number.strip())
        if not account:
            return {
                "found": False,
                "message": f"Account {account_number} not found or inactive."
            }

        transactions = get_transactions(
            account["id"],
            limit=limit,
            start_date=start_date,
            end_date=end_date,
            transaction_type=transaction_type,
            category=category,
        )
        duration = (time.time() - start) * 1000

        logger.info(f"[{trace_id}] TOOL | get_last_transactions | success | count={len(transactions)} | duration={duration:.2f}ms")

        formatted = []
        for t in transactions:
            formatted.append({
                "type": t["transaction_type"],
                "category": t["category"],
                "amount": float(t["amount"]),
                "description": t["description"],
                "date": str(t["created_at"])[:10],
                "balance_after": float(t["balance_after"])
            })

        return {
            "found": True,
            "account_number": account_number,
            "account_holder": account["account_holder"],
            "transactions": formatted,
            "total": len(formatted)
        }
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | get_last_transactions | error={e}")
        return {"found": False, "message": f"Error retrieving transactions: {str(e)}"}


def tool_get_spend_summary(
    account_number: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    trace_id: str = "",
) -> dict:
    """
    Get a spend summary (total spent and transaction count, grouped by
    category) for a given account number. Use for questions like
    "how much did I spend on bills this month" or "what's my spending
    breakdown". Optionally filter by start_date/end_date (YYYY-MM-DD) and/or
    a specific category.
    """
    start = time.time()
    try:
        account = get_account_by_number(account_number.strip())
        if not account:
            return {
                "found": False,
                "message": f"Account {account_number} not found or inactive."
            }

        summary = get_spend_summary(
            account["id"],
            start_date=start_date,
            end_date=end_date,
            category=category,
        )
        duration = (time.time() - start) * 1000

        logger.info(f"[{trace_id}] TOOL | get_spend_summary | success | duration={duration:.2f}ms")

        formatted = [
            {
                "category": row["category"],
                "total": float(row["total"]),
                "count": row["count"],
            }
            for row in summary
        ]

        return {
            "found": True,
            "account_number": account_number,
            "account_holder": account["account_holder"],
            "summary": formatted,
            "total_spent": sum(row["total"] for row in formatted),
        }
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | get_spend_summary | error={e}")
        return {"found": False, "message": f"Error retrieving spend summary: {str(e)}"}


def tool_check_cheque_status(request_id: str, trace_id: str = "") -> dict:
    """
    Check the status of a previously submitted cheque deposit request by its
    request ID (format CHQ-XXXXXXXX).
    """
    start = time.time()
    try:
        cheque_request = get_cheque_request_by_id(request_id.strip().upper())
        duration = (time.time() - start) * 1000

        if not cheque_request:
            logger.info(f"[{trace_id}] TOOL | check_cheque_status | not_found | request_id={request_id}")
            return {
                "found": False,
                "message": f"No cheque request found with ID {request_id}."
            }

        logger.info(f"[{trace_id}] TOOL | check_cheque_status | success | duration={duration:.2f}ms")

        return {
            "found": True,
            "request_id": cheque_request["request_id"],
            "status": cheque_request["status"],
            "payee": cheque_request["payee"],
            "amount": cheque_request["amount_in_figures"],
            "bank_name": cheque_request["bank_name"],
            "created_at": str(cheque_request["created_at"]),
        }
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | check_cheque_status | error={e}")
        return {"found": False, "message": f"Error retrieving cheque request: {str(e)}"}


def tool_start_cheque_workflow(phone_number: str) -> str:
    """
    Start a cheque deposit workflow.

    Use this tool when the customer wants to:
    - deposit a cheque
    - cash a cheque
    - submit a cheque

    Do not use this tool for balance or transaction requests.
    """

    existing_workflow = get_workflow(phone_number)

    if existing_workflow:
        return (
            "You already have an active workflow in progress. "
            "Please complete it before starting a new one."
        )

    workflow = create_workflow_model(
        workflow_type=WORKFLOW_CHEQUE,
        step=STEP_UPLOAD_CHEQUE,
    )

    create_workflow(phone_number, workflow)

    return (
        "Cheque deposit started successfully.\n\n"
        "Please upload a clear image of the cheque to continue."
    )

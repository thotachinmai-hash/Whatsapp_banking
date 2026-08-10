import re
import time
from datetime import date, timedelta
from app.database import (
    get_account_by_number,
    get_accounts_by_phone,
    get_transactions,
    get_spend_summary,
    get_cheque_request_by_id,
    get_cheque_requests_by_phone,
    get_loan_request_by_id,
    get_loan_requests_by_phone,
    get_transfer_by_reference,
    get_transfers_by_phone,
    get_beneficiaries_by_phone,
    get_loan_product,
    get_all_loan_products,
)
from app.conversation.responses.common import mask_account_number
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
from app.rag.retriever import search as rag_search

logger = get_logger(__name__)

# The LLM (and customers) naturally describe spending in everyday words that
# don't match the exact category values stored in the transactions table.
# Map the common synonyms onto the real category before querying, so "how
# much did I spend on food" actually finds the "groceries" rows instead of
# silently matching nothing and reporting £0.
CATEGORY_SYNONYMS = {
    "food": "groceries", "foods": "groceries", "grocery": "groceries",
    "groceries": "groceries", "supermarket": "groceries", "eating": "groceries",
    "bill": "bills", "bills": "bills", "utilities": "bills", "utility": "bills",
    "rent": "rent", "housing": "rent",
    "transport": "transport", "transportation": "transport", "travel": "transport",
    "commute": "transport", "fuel": "transport", "petrol": "transport",
    "entertainment": "entertainment", "fun": "entertainment", "leisure": "entertainment",
    "shopping": "shopping", "clothes": "shopping", "clothing": "shopping",
    "salary": "salary", "income": "salary", "wages": "salary", "pay": "salary",
    "isa": "isa", "savings": "isa",
    "bonus": "bonus",
    "interest": "interest",
    "transfer": "transfer", "transfers": "transfer",
}


def _normalize_category(category: str | None) -> str | None:
    if not category:
        return category
    key = category.strip().lower()
    return CATEGORY_SYNONYMS.get(key, category.strip())


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RELATIVE_UNITS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}


def _normalize_date(value: str | None, trace_id: str = "") -> str | None:
    """The LLM is asked for YYYY-MM-DD but sometimes sends everyday phrases
    like "1 month ago" or "last week" instead — passed straight to
    Postgres those crash the query. Translate the common ones to a real
    date, and drop (rather than crash on) anything else unparseable so the
    tool still returns results instead of failing the whole turn."""
    if not value:
        return None
    text = value.strip().lower()
    if _DATE_RE.match(text):
        return text

    today = date.today()
    if text in ("today",):
        return today.isoformat()
    if text in ("yesterday",):
        return (today - timedelta(days=1)).isoformat()
    if text in ("this month",):
        return today.replace(day=1).isoformat()
    if text in ("this week",):
        return (today - timedelta(days=today.weekday())).isoformat()
    if text in ("this year",):
        return today.replace(month=1, day=1).isoformat()
    if text in ("last month",):
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        return last_month_end.replace(day=1).isoformat()
    if text in ("last week",):
        start_of_this_week = today - timedelta(days=today.weekday())
        return (start_of_this_week - timedelta(days=7)).isoformat()
    if text in ("last year",):
        return today.replace(year=today.year - 1, month=1, day=1).isoformat()

    match = re.fullmatch(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*ago", text)
    if match:
        count = int(match.group(1))
        unit_days = _RELATIVE_UNITS[match.group(2)]
        return (today - timedelta(days=count * unit_days)).isoformat()

    logger.warning(f"[{trace_id}] Unrecognized date filter ignored | value={value!r}")
    return None


def tool_get_account_balance(account_number: str = "", phone_number: str = "", trace_id: str = "") -> dict:
    """
    Get account balance for a given account number.
    Queries PostgreSQL accounts table.
    Returns balance, account type, account holder name, currency.
    """
    start = time.time()
    try:
        account_number = (account_number or "").strip()
        if account_number:
            account = get_account_by_number(account_number)
        else:
            accounts = get_accounts_by_phone(phone_number)
            if not accounts:
                return {"found": False, "message": "No active accounts are linked to this mobile number."}
            if len(accounts) > 1:
                return {"found": True, "accounts": [
                    {"account_number": item["account_number"], "account_type": item["account_type"],
                     "balance": float(item["balance"]), "currency": item["currency"]}
                    for item in accounts
                ]}
            account = get_account_by_number(accounts[0]["account_number"])
        duration = (time.time() - start) * 1000

        if not account:
            logger.info(f"[{trace_id}] TOOL | get_account_balance | not_found | account={account_number}")
            return {
                "found": False,
                "message": f"Account {account_number or 'linked to this mobile number'} was not found or is inactive."
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
    account_number: str = "",
    limit: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    transaction_type: str | None = None,
    category: str | None = None,
    trace_id: str = "",
    phone_number: str = "",
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
        account_number = (account_number or "").strip()
        if not account_number:
            accounts = get_accounts_by_phone(phone_number)
            if not accounts:
                return {"found": False, "message": "No active accounts are linked to this mobile number."}
            account_number = accounts[0]["account_number"]
        account = get_account_by_number(account_number)
        if not account:
            return {
                "found": False,
                "message": f"Account {account_number} not found or inactive."
            }

        transactions = get_transactions(
            account["id"],
            limit=limit,
            start_date=_normalize_date(start_date, trace_id),
            end_date=_normalize_date(end_date, trace_id),
            transaction_type=transaction_type,
            category=_normalize_category(category),
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
    account_number: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    trace_id: str = "",
    phone_number: str = "",
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
        account_number = (account_number or "").strip()
        if not account_number:
            accounts = get_accounts_by_phone(phone_number)
            if not accounts:
                return {"found": False, "message": "No active accounts are linked to this mobile number."}
            account_number = accounts[0]["account_number"]
        account = get_account_by_number(account_number)
        if not account:
            return {
                "found": False,
                "message": f"Account {account_number} not found or inactive."
            }

        summary = get_spend_summary(
            account["id"],
            start_date=_normalize_date(start_date, trace_id),
            end_date=_normalize_date(end_date, trace_id),
            category=_normalize_category(category),
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


_LOAN_TYPE_ALIASES = {
    "personal": "personal", "personal loan": "personal",
    "home": "home", "home loan": "home", "mortgage": "home", "housing": "home",
    "vehicle": "vehicle", "vehicle loan": "vehicle", "car": "vehicle", "auto": "vehicle",
    "education": "education", "education loan": "education", "student": "education",
}


def tool_get_loan_product_info(loan_type: str = "", trace_id: str = "") -> dict:
    """
    Get the bank's published loan terms — interest rate range, min/max
    amount, tenure range, and processing fee — for a loan type (personal,
    home, vehicle, education). Use this for questions like "what's the
    interest rate on a personal loan" or "how much can I borrow for a car".

    This is the bank's rate card, not a personal eligibility decision — it
    never confirms what rate or amount a specific customer would get; that
    still depends on their application. If no loan_type is given, returns
    the full rate card for all loan types.
    """
    start = time.time()
    try:
        normalized = _LOAN_TYPE_ALIASES.get((loan_type or "").strip().lower())

        if not loan_type:
            products = get_all_loan_products()
            duration = (time.time() - start) * 1000
            logger.info(f"[{trace_id}] TOOL | get_loan_product_info | all | duration={duration:.2f}ms")
            return {"found": bool(products), "products": [_format_loan_product(p) for p in products]}

        if not normalized:
            return {
                "found": False,
                "message": f"'{loan_type}' isn't a loan type this bank offers. Available types: personal, home, vehicle, education.",
            }

        product = get_loan_product(normalized)
        duration = (time.time() - start) * 1000
        if not product:
            logger.info(f"[{trace_id}] TOOL | get_loan_product_info | not_found | type={normalized}")
            return {"found": False, "message": f"No published rate card for {normalized} loans right now."}

        logger.info(f"[{trace_id}] TOOL | get_loan_product_info | success | type={normalized} | duration={duration:.2f}ms")
        return {"found": True, "product": _format_loan_product(product)}
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | get_loan_product_info | error={e}")
        return {"found": False, "message": f"Error retrieving loan product info: {str(e)}"}


def _format_loan_product(product: dict) -> dict:
    return {
        "loan_type": product["loan_type"],
        "display_name": product["display_name"],
        "interest_rate_min": float(product["interest_rate_min"]),
        "interest_rate_max": float(product["interest_rate_max"]),
        "min_amount": float(product["min_amount"]),
        "max_amount": float(product["max_amount"]),
        "min_tenure_months": product["min_tenure_months"],
        "max_tenure_months": product["max_tenure_months"],
        "processing_fee_percent": float(product["processing_fee_percent"]),
        "currency": product["currency"],
        "notes": product["notes"],
    }


def tool_search_bank_documents(query: str = "", trace_id: str = "") -> dict:
    """
    Search the bank's own indexed policy/product documents — required
    documents for each loan type, KYC document rules, cheque deposit
    rules, and general how-it-works FAQs. Use this for any "what documents
    do I need", "how does KYC work", "why was my cheque rejected", or
    similar informational/policy question that isn't covered by a more
    specific tool (e.g. loan rates still use get_loan_product_info).

    Returns only what is actually indexed — never invents an answer when
    nothing relevant is found. Contains no account-specific or customer
    data; it is safe to call for both registered and unregistered users.
    """
    start = time.time()
    try:
        query = (query or "").strip()
        if not query:
            return {"found": False, "message": "No search query was provided."}
        results = rag_search(query, top_k=3)
        duration = (time.time() - start) * 1000
        logger.info(f"[{trace_id}] TOOL | search_bank_documents | results={len(results)} | duration={duration:.2f}ms")
        if not results:
            return {"found": False, "message": "Nothing relevant was found in the bank's documents for this question."}
        return {"found": True, "results": [{"title": r["title"], "text": r["text"]} for r in results]}
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | search_bank_documents | error={e}")
        return {"found": False, "message": f"Error searching documents: {str(e)}"}


def tool_check_cheque_status(request_id: str = "", phone_number: str = "", trace_id: str = "") -> dict:
    """
    Check the status of a previously submitted cheque deposit request by its
    request ID (format CHQ-XXXXXXXX).
    """
    start = time.time()
    try:
        request_id = (request_id or "").strip().upper()
        if request_id:
            cheque_request = get_cheque_request_by_id(request_id)
            requests = [cheque_request] if cheque_request else []
        else:
            requests = get_cheque_requests_by_phone(phone_number)
        duration = (time.time() - start) * 1000

        if not requests:
            logger.info(f"[{trace_id}] TOOL | check_cheque_status | not_found | request_id={request_id}")
            return {
                "found": False,
                "message": "No cheque requests were found for this customer."
            }

        logger.info(f"[{trace_id}] TOOL | check_cheque_status | success | duration={duration:.2f}ms")

        return {"found": True, "count": len(requests), "cheques": [
            {
                "request_id": cheque["request_id"],
                "status": cheque["status"],
                "payee": cheque.get("payee"),
                "amount": cheque.get("amount_in_figures"),
                "bank_name": cheque.get("bank_name"),
                "cheque_number": cheque.get("cheque_number"),
                "created_at": str(cheque.get("created_at")),
            }
            for cheque in requests
        ]}
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | check_cheque_status | error={e}")
        return {"found": False, "message": f"Error retrieving cheque request: {str(e)}"}


def tool_check_loan_status(request_id: str = "", phone_number: str = "", trace_id: str = "") -> dict:
    """Check the status of a submitted loan application."""
    try:
        request_id = (request_id or "").strip().upper()
        if request_id:
            request = get_loan_request_by_id(request_id)
            requests = [request] if request else []
        else:
            requests = get_loan_requests_by_phone(phone_number)
        if not requests:
            return {"found": False, "message": "No loan applications were found for this customer."}
        return {"found": True, "count": len(requests), "applications": [
            {
                "request_id": item["request_id"],
                "loan_type": item["loan_type"],
                "status": item["status"],
                "details": item.get("details", {}),
                "created_at": str(item.get("created_at")),
            }
            for item in requests
        ]}
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | check_loan_status | error={e}")
        return {"found": False, "message": f"Error retrieving loan application: {str(e)}"}


def tool_check_transfer_status(request_id: str = "", phone_number: str = "", trace_id: str = "") -> dict:
    """
    Check the status of a money transfer by its transaction ID (format
    TRF-XXXXXXXX), or list the customer's transfers (most recent first) if
    no ID is given — use this for "what was my last transfer" too.
    """
    start = time.time()
    try:
        request_id = (request_id or "").strip().upper()
        if request_id:
            transfer = get_transfer_by_reference(request_id)
            transfers = [transfer] if transfer else []
        else:
            transfers = get_transfers_by_phone(phone_number)
        duration = (time.time() - start) * 1000

        if not transfers:
            logger.info(f"[{trace_id}] TOOL | check_transfer_status | not_found | request_id={request_id}")
            return {"found": False, "message": "No transfers were found for this customer."}

        logger.info(f"[{trace_id}] TOOL | check_transfer_status | success | duration={duration:.2f}ms")

        return {"found": True, "count": len(transfers), "transfers": [
            {
                "reference": item["reference"],
                "status": item["status"],
                "beneficiary_name": item.get("beneficiary_name"),
                "beneficiary_account": item.get("beneficiary_account"),
                "amount": float(item["amount"]) if item.get("amount") is not None else None,
                "source_account": item.get("source_account"),
                "created_at": str(item.get("created_at")),
            }
            for item in transfers
        ]}
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | check_transfer_status | error={e}")
        return {"found": False, "message": f"Error retrieving transfer: {str(e)}"}


def tool_list_beneficiaries(phone_number: str = "", trace_id: str = "") -> dict:
    """
    List the customer's saved transfer beneficiaries (name and masked
    account number). Use this for "list my beneficiaries", "who do I have
    saved", "show my saved payees", or similar — never invent a
    beneficiary list from memory or a previous transfer.
    """
    start = time.time()
    try:
        beneficiaries = get_beneficiaries_by_phone(phone_number)
        duration = (time.time() - start) * 1000
        logger.info(f"[{trace_id}] TOOL | list_beneficiaries | count={len(beneficiaries)} | duration={duration:.2f}ms")
        if not beneficiaries:
            return {"found": False, "message": "No saved beneficiaries yet."}
        return {"found": True, "count": len(beneficiaries), "beneficiaries": [
            {
                "name": item["beneficiary_name"],
                "account_number_masked": mask_account_number(item["account_number"]),
                "bank_name": item.get("bank_name"),
            }
            for item in beneficiaries
        ]}
    except Exception as e:
        logger.error(f"[{trace_id}] TOOL | list_beneficiaries | error={e}")
        return {"found": False, "message": f"Error retrieving beneficiaries: {str(e)}"}


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

from fastapi import APIRouter, HTTPException
from app.database import execute_query, get_account_by_number, get_cheque_request_by_id
from app.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/customers", tags=["Customers"], summary="List all registered customers")
async def list_customers():
    """List all registered customers — for testing."""
    customers = execute_query(
        "SELECT phone_number, full_name, aadhaar_number, pan_number, status, created_at "
        "FROM customers ORDER BY id"
    )
    return {"customers": customers, "total": len(customers)}


@router.get("/cheque-requests/{request_id}", tags=["Cheques"], summary="Get cheque request by ID")
async def get_cheque_request(request_id: str):
    """Get a cheque deposit request by its request ID."""
    cheque_request = get_cheque_request_by_id(request_id.strip().upper())
    if not cheque_request:
        raise HTTPException(status_code=404, detail="Cheque request not found")
    return {"cheque_request": cheque_request}


@router.get("/accounts", tags=["Accounts"], summary="List all accounts")
async def list_accounts():
    """List all bank accounts — for testing."""
    accounts = execute_query(
        "SELECT account_number, account_holder, account_type, balance, currency, status FROM accounts ORDER BY id"
    )
    return {"accounts": accounts, "total": len(accounts)}


@router.get("/accounts/{account_number}", tags=["Accounts"], summary="Get account by number")
async def get_account(account_number: str):
    """Get account details by account number."""
    account = get_account_by_number(account_number)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    return {"account": account}


@router.get("/accounts/{account_number}/transactions", tags=["Accounts"], summary="Get transactions")
async def get_transactions(account_number: str, limit: int = 10):
    """Get transactions for an account."""
    account = get_account_by_number(account_number)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    from app.database import get_transactions as db_get_transactions
    transactions = db_get_transactions(account["id"], limit=limit)
    return {
        "account_number": account_number,
        "transactions": transactions,
        "total": len(transactions)
    }

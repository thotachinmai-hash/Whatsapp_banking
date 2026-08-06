import os
import secrets
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_db_connection():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    logger.info("Database connection established")
    return conn


def execute_query(query: str, params: tuple = None) -> list:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        return results
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def execute_write(query: str, params: tuple = None) -> bool:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Write failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def execute_write_returning(query: str, params: tuple = None) -> dict | None:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.commit()
        return dict(row) if row else None
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Write-returning failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_account_by_number(account_number: str) -> dict | None:
    results = execute_query(
        "SELECT * FROM accounts WHERE account_number = %s AND status = 'active'",
        (account_number,)
    )
    return results[0] if results else None


def get_accounts_by_phone(phone_number: str) -> list[dict]:
    """Return all active accounts linked to a customer's mobile number."""
    return execute_query(
        """SELECT account_number, account_type, balance, currency, status
           FROM accounts
           WHERE phone_number = %s AND status = 'active'
           ORDER BY id""",
        (phone_number,),
    )


def get_transactions(
    account_id: int,
    limit: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    transaction_type: str | None = None,
    category: str | None = None,
) -> list:
    query = "SELECT * FROM transactions WHERE account_id = %s"
    params: list = [account_id]

    if start_date:
        query += " AND created_at >= %s"
        params.append(start_date)
    if end_date:
        query += " AND created_at <= %s"
        params.append(end_date)
    if transaction_type:
        query += " AND transaction_type = %s"
        params.append(transaction_type)
    if category:
        query += " AND category = %s"
        params.append(category)

    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    return execute_query(query, tuple(params))


def get_spend_summary(
    account_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
) -> list:
    query = """
        SELECT category, SUM(amount) AS total, COUNT(*) AS count
        FROM transactions
        WHERE account_id = %s AND transaction_type = 'debit'
    """
    params: list = [account_id]

    if start_date:
        query += " AND created_at >= %s"
        params.append(start_date)
    if end_date:
        query += " AND created_at <= %s"
        params.append(end_date)
    if category:
        query += " AND category = %s"
        params.append(category)

    query += " GROUP BY category ORDER BY total DESC"
    return execute_query(query, tuple(params))


def get_customer_by_phone(phone_number: str) -> dict | None:
    results = execute_query(
        "SELECT * FROM customers WHERE phone_number = %s",
        (phone_number,)
    )
    return results[0] if results else None


def create_customer(
    phone_number: str,
    full_name: str,
    aadhaar_number: str,
    pan_number: str,
) -> dict | None:
    return execute_write_returning(
        """INSERT INTO customers (phone_number, full_name, aadhaar_number, pan_number)
           VALUES (%s, %s, %s, %s)
           RETURNING *""",
        (phone_number, full_name, aadhaar_number, pan_number)
    )


ACCOUNT_TYPES = ("savings", "current", "salary")


def create_zero_balance_account(
    phone_number: str,
    account_holder: str,
    account_type: str,
) -> dict | None:
    """Create an active zero-balance account with a collision-safe number."""
    normalized_type = account_type.strip().lower()
    if normalized_type not in ACCOUNT_TYPES:
        raise ValueError(f"Unsupported account type: {account_type}")

    for _ in range(3):
        account_number = f"INFI{secrets.token_hex(8).upper()}"
        try:
            return execute_write_returning(
                """INSERT INTO accounts
                   (account_number, account_holder, phone_number, account_type,
                    balance, currency, status)
                   VALUES (%s, %s, %s, %s, 0.00, 'GBP', 'active')
                   RETURNING *""",
                (account_number, account_holder, phone_number, normalized_type),
            )
        except psycopg2.errors.UniqueViolation:
            logger.warning("Account number collision; retrying")

    raise RuntimeError("Unable to generate a unique account number")


def create_cheque_request(
    request_id: str,
    phone_number: str,
    bank_name: str | None,
    branch: str | None,
    payee: str | None,
    amount_in_figures: str | None,
    amount_in_words: str | None,
    cheque_number: str | None,
    signatory: str | None,
    status: str = "PENDING",
) -> dict | None:
    return execute_write_returning(
        """INSERT INTO cheque_requests
           (request_id, phone_number, bank_name, branch, payee,
            amount_in_figures, amount_in_words, cheque_number, signatory, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (request_id, phone_number, bank_name, branch, payee,
         amount_in_figures, amount_in_words, cheque_number, signatory, status)
    )


def get_cheque_request_by_id(request_id: str) -> dict | None:
    results = execute_query(
        "SELECT * FROM cheque_requests WHERE request_id = %s",
        (request_id,)
    )
    return results[0] if results else None


def ensure_application_tables() -> None:
    """Create workflow request tables for installations initialized before this feature."""
    execute_write(
        """CREATE TABLE IF NOT EXISTS loan_requests (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(20) UNIQUE NOT NULL,
            phone_number VARCHAR(20) NOT NULL,
            loan_type VARCHAR(50) NOT NULL,
            details JSONB NOT NULL,
            status VARCHAR(20) DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS kyc_requests (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(20) UNIQUE NOT NULL,
            phone_number VARCHAR(20) NOT NULL,
            details JSONB NOT NULL,
            status VARCHAR(20) DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT NOW()
        );"""
    )


def create_loan_request(
    request_id: str,
    phone_number: str,
    loan_type: str,
    details: dict,
    status: str = "PENDING",
) -> dict | None:
    ensure_application_tables()
    return execute_write_returning(
        """INSERT INTO loan_requests
           (request_id, phone_number, loan_type, details, status)
           VALUES (%s, %s, %s, %s::jsonb, %s)
           RETURNING *""",
        (request_id, phone_number, loan_type, json.dumps(details), status),
    )


def create_kyc_request(
    request_id: str,
    phone_number: str,
    details: dict,
    status: str = "PENDING",
) -> dict | None:
    ensure_application_tables()
    return execute_write_returning(
        """INSERT INTO kyc_requests (request_id, phone_number, details, status)
           VALUES (%s, %s, %s::jsonb, %s)
           RETURNING *""",
        (request_id, phone_number, json.dumps(details), status),
    )

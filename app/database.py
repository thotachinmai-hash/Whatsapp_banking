import os
import secrets
import json
import psycopg2
import psycopg2.errors
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
    date_of_birth: str = "",
    guardian_name: str = "",
    address: str = "",
) -> dict | None:
    return execute_write_returning(
        """INSERT INTO customers (
            phone_number, full_name, aadhaar_number, pan_number,
            date_of_birth, guardian_name, address
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (phone_number, full_name, aadhaar_number, pan_number, date_of_birth, guardian_name, address)
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
    date_written: str | None = None,
    drawer_name: str | None = None,
    status: str = "PENDING",
) -> dict | None:
    return execute_write_returning(
        """INSERT INTO cheque_requests
           (request_id, phone_number, bank_name, branch, payee,
            amount_in_figures, amount_in_words, cheque_number, signatory,
            date_written, drawer_name, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (request_id, phone_number, bank_name, branch, payee,
         amount_in_figures, amount_in_words, cheque_number, signatory,
         date_written, drawer_name, status)
    )


def get_cheque_request_by_id(request_id: str) -> dict | None:
    results = execute_query(
        "SELECT * FROM cheque_requests WHERE request_id = %s",
        (request_id,)
    )
    return results[0] if results else None


def get_cheque_requests_by_phone(phone_number: str) -> list[dict]:
    """Return all cheque requests belonging to the signed-in customer."""
    return execute_query(
        """SELECT * FROM cheque_requests
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (phone_number,),
    )


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
        );
        ALTER TABLE cheque_requests
            ADD COLUMN IF NOT EXISTS date_written VARCHAR(50);
        ALTER TABLE cheque_requests
            ADD COLUMN IF NOT EXISTS drawer_name VARCHAR(255);"""
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


def get_loan_request_by_id(request_id: str) -> dict | None:
    results = execute_query(
        "SELECT * FROM loan_requests WHERE request_id = %s",
        (request_id.strip().upper(),),
    )
    return results[0] if results else None


def get_loan_requests_by_phone(phone_number: str) -> list[dict]:
    """Return all loan applications belonging to the signed-in customer."""
    ensure_application_tables()
    return execute_query(
        """SELECT * FROM loan_requests
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (phone_number,),
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


def ensure_beneficiaries_table() -> None:
    """Create the beneficiaries table for installations initialized before this feature."""
    execute_write(
        """CREATE TABLE IF NOT EXISTS beneficiaries (
            id SERIAL PRIMARY KEY,
            phone_number VARCHAR(20) NOT NULL,
            beneficiary_name VARCHAR(255) NOT NULL,
            account_number VARCHAR(30) NOT NULL,
            bank_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (phone_number, account_number)
        );"""
    )


def get_beneficiaries_by_phone(phone_number: str) -> list[dict]:
    """Return all saved beneficiaries for a customer, oldest first."""
    ensure_beneficiaries_table()
    return execute_query(
        """SELECT * FROM beneficiaries
           WHERE phone_number = %s
           ORDER BY id""",
        (phone_number,),
    )


def create_beneficiary(
    phone_number: str,
    beneficiary_name: str,
    account_number: str,
    bank_name: str | None = None,
) -> dict | None:
    """
    Save a new beneficiary, or update the name/bank on an existing one with
    the same account number for this customer (re-adding is harmless).
    """
    ensure_beneficiaries_table()
    return execute_write_returning(
        """INSERT INTO beneficiaries (phone_number, beneficiary_name, account_number, bank_name)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (phone_number, account_number)
           DO UPDATE SET beneficiary_name = EXCLUDED.beneficiary_name,
                          bank_name = COALESCE(EXCLUDED.bank_name, beneficiaries.bank_name)
           RETURNING *""",
        (phone_number, beneficiary_name, account_number, bank_name),
    )


def ensure_transfers_table() -> None:
    """Create the transfers table for installations initialized before this feature."""
    execute_write(
        """CREATE TABLE IF NOT EXISTS transfers (
            id SERIAL PRIMARY KEY,
            reference VARCHAR(20) UNIQUE NOT NULL,
            phone_number VARCHAR(20) NOT NULL,
            source_account VARCHAR(30) NOT NULL,
            beneficiary_name VARCHAR(255) NOT NULL,
            beneficiary_account VARCHAR(30) NOT NULL,
            amount DECIMAL(15, 2) NOT NULL,
            status VARCHAR(20) DEFAULT 'INITIATED',
            created_at TIMESTAMP DEFAULT NOW()
        );"""
    )


def create_transfer(
    reference: str,
    phone_number: str,
    source_account: str,
    beneficiary_name: str,
    beneficiary_account: str,
    amount,
    status: str = "INITIATED",
) -> dict | None:
    ensure_transfers_table()
    return execute_write_returning(
        """INSERT INTO transfers
           (reference, phone_number, source_account, beneficiary_name, beneficiary_account, amount, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (reference, phone_number, source_account, beneficiary_name, beneficiary_account, amount, status),
    )


def get_transfer_by_reference(reference: str) -> dict | None:
    ensure_transfers_table()
    results = execute_query(
        "SELECT * FROM transfers WHERE reference = %s",
        (reference.strip().upper(),),
    )
    return results[0] if results else None


def get_transfers_by_phone(phone_number: str) -> list[dict]:
    """Return all transfers initiated by this customer, most recent first."""
    ensure_transfers_table()
    return execute_query(
        """SELECT * FROM transfers
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (phone_number,),
    )


def update_transfer_status(reference: str, status: str) -> dict | None:
    """Reflect the bank side completing (or failing) a transfer."""
    ensure_transfers_table()
    return execute_write_returning(
        """UPDATE transfers SET status = %s WHERE reference = %s RETURNING *""",
        (status, reference.strip().upper()),
    )


def get_loan_product(loan_type: str) -> dict | None:
    """The bank's published rate-card entry for one loan type — real,
    static reference data (see infra/postgres/init.sql), not a customer's
    application. Used so the assistant can answer rate/fee/tenure
    questions with an actual figure instead of inventing or refusing."""
    results = execute_query(
        "SELECT * FROM loan_products WHERE loan_type = %s",
        (loan_type.strip().lower(),),
    )
    return results[0] if results else None


def get_all_loan_products() -> list[dict]:
    return execute_query("SELECT * FROM loan_products ORDER BY loan_type")

import os
import re
import secrets
import json
from decimal import Decimal
import psycopg2
import psycopg2.errors
import psycopg2.extras
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def normalize_phone_number(phone_number: str | None) -> str:
    """Normalize WhatsApp-style phone forms to the canonical digits-only
    format used in the bank database.

    This handles values such as "+91 98765 43210", "911111111111",
    "447818658034", and numbers with leading zeros or formatting noise.
    """
    if phone_number is None:
        return ""

    raw = str(phone_number).strip()
    if not raw:
        return ""

    digits = re.sub(r"\D+", "", raw)
    if not digits:
        return raw.strip()

    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 10:
        digits = digits[1:]
    return digits


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
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return []
    return execute_query(
        """SELECT account_number, account_type, balance, currency, status
           FROM accounts
           WHERE phone_number = %s AND status = 'active'
           ORDER BY id""",
        (normalized,),
    )


def get_frequently_used_account(phone_number: str, accounts: list[dict] | None = None) -> dict | None:
    """The account linked to this phone with the most transaction history,
    used to suggest a default instead of always asking which account to
    use (see app/workflows/processors/loan.py's account-confirmation
    step). Ties (including the common "no transactions yet" case) fall
    back to the first account on file, same ordering get_accounts_by_phone
    already uses.

    `accounts`, if given, is used instead of a fresh get_accounts_by_phone
    lookup — callers that already fetched the list (or, in tests, that
    monkeypatch their own import of get_accounts_by_phone rather than this
    module's) get one consistent answer instead of two independent DB
    calls that could disagree."""
    if accounts is None:
        accounts = get_accounts_by_phone(phone_number)
    if not accounts:
        return None
    if len(accounts) == 1:
        return accounts[0]
    normalized = normalize_phone_number(phone_number)
    rows = execute_query(
        """SELECT a.account_number, COUNT(t.id) AS tx_count
           FROM accounts a
           LEFT JOIN transactions t ON t.account_id = a.id
           WHERE a.phone_number = %s AND a.status = 'active'
           GROUP BY a.account_number, a.id
           ORDER BY tx_count DESC, a.id ASC""",
        (normalized,),
    )
    if not rows:
        return accounts[0]
    top_account_number = rows[0]["account_number"]
    return next((a for a in accounts if a["account_number"] == top_account_number), accounts[0])


def get_transactions(
    account_id: int,
    limit: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    transaction_type: str | None = None,
    category: str | None = None,
    keyword: str | None = None,
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
    if keyword:
        # Free-text match against the transaction description — lets a
        # counterparty/purpose search ("payments to my landlord") work
        # without needing a dedicated counterparty column.
        query += " AND description ILIKE %s"
        params.append(f"%{keyword}%")

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
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return None
    results = execute_query(
        "SELECT * FROM customers WHERE phone_number = %s",
        (normalized,)
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
    normalized_phone = normalize_phone_number(phone_number)
    return execute_write_returning(
        """INSERT INTO customers (
            phone_number, full_name, aadhaar_number, pan_number,
            date_of_birth, guardian_name, address
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
           RETURNING *""",
        (normalized_phone, full_name, aadhaar_number, pan_number, date_of_birth, guardian_name, address)
    )


ACCOUNT_TYPES = ("savings", "current", "salary")


def build_default_account_transaction_history() -> list[dict]:
    """Return the default 15-transaction history for a newly created account.

    The ledger starts from zero and ends at the account's default balance of
    20000.00, with the final balance_after matching the seeded account balance.
    """
    default_balance = Decimal("20000.00")
    entries = [
        ("credit", "salary", Decimal("2500.00"), "Salary credit", "INIT-001"),
        ("debit", "rent", Decimal("1200.00"), "Rent payment", "INIT-002"),
        ("debit", "groceries", Decimal("300.00"), "Groceries", "INIT-003"),
        ("credit", "bonus", Decimal("1500.00"), "Bonus payout", "INIT-004"),
        ("debit", "bills", Decimal("700.00"), "Utility bill", "INIT-005"),
        ("debit", "transport", Decimal("450.00"), "Transport", "INIT-006"),
        ("credit", "salary", Decimal("2200.00"), "Salary credit", "INIT-007"),
        ("debit", "shopping", Decimal("600.00"), "Shopping", "INIT-008"),
        ("debit", "entertainment", Decimal("350.00"), "Streaming services", "INIT-009"),
        ("credit", "transfer", Decimal("1200.00"), "Internal transfer", "INIT-010"),
        ("debit", "groceries", Decimal("200.00"), "Groceries", "INIT-011"),
        ("credit", "bonus", Decimal("4500.00"), "Performance bonus", "INIT-012"),
        ("debit", "rent", Decimal("1100.00"), "Rent payment", "INIT-013"),
        ("credit", "transfer", Decimal("9000.00"), "Account top-up", "INIT-014"),
        ("credit", "other", Decimal("4000.00"), "Account opening balance", "INIT-015"),
    ]

    running_balance = Decimal("0.00")
    history = []
    for transaction_type, category, amount, description, reference in entries:
        if transaction_type == "credit":
            running_balance += amount
        else:
            running_balance -= amount
        history.append({
            "transaction_type": transaction_type,
            "category": category,
            "amount": amount,
            "description": description,
            "reference": reference,
            "balance_after": running_balance.quantize(Decimal("0.01")),
        })

    if history[-1]["balance_after"] != default_balance:
        raise ValueError(f"Default transaction history must end at {default_balance}, got {history[-1]['balance_after']}")

    return history


def create_zero_balance_account(
    phone_number: str,
    account_holder: str,
    account_type: str,
) -> dict | None:
    """Create an active account with the default bank opening balance."""
    normalized_phone = normalize_phone_number(phone_number)
    normalized_type = account_type.strip().lower()
    if normalized_type not in ACCOUNT_TYPES:
        raise ValueError(f"Unsupported account type: {account_type}")

    for _ in range(3):
        account_number = f"INFI{secrets.token_hex(8).upper()}"
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                """INSERT INTO accounts
                   (account_number, account_holder, phone_number, account_type,
                    balance, currency, status)
                   VALUES (%s, %s, %s, %s, %s, 'INR', 'active')
                   RETURNING *""",
                (account_number, account_holder, normalized_phone, normalized_type, Decimal("20000.00")),
            )
            account = dict(cursor.fetchone())
            for item in build_default_account_transaction_history():
                cursor.execute(
                    """INSERT INTO transactions
                       (account_id, transaction_type, category, amount, description,
                        reference, balance_after, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (
                        account["id"],
                        item["transaction_type"],
                        item["category"],
                        item["amount"],
                        item["description"],
                        item["reference"],
                        item["balance_after"],
                    ),
                )
            conn.commit()
            return account
        except psycopg2.errors.UniqueViolation:
            logger.warning("Account number collision; retrying")
            if conn:
                conn.rollback()
        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

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
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return []
    return execute_query(
        """SELECT * FROM cheque_requests
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (normalized,),
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
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return []
    return execute_query(
        """SELECT * FROM loan_requests
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (normalized,),
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


def get_kyc_request_by_id(request_id: str) -> dict | None:
    ensure_application_tables()
    results = execute_query(
        "SELECT * FROM kyc_requests WHERE request_id = %s",
        (request_id.strip().upper(),),
    )
    return results[0] if results else None


def get_kyc_requests_by_phone(phone_number: str) -> list[dict]:
    """Return all KYC update requests belonging to the signed-in customer."""
    ensure_application_tables()
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return []
    return execute_query(
        """SELECT * FROM kyc_requests
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (normalized,),
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
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return []
    return execute_query(
        """SELECT * FROM beneficiaries
           WHERE phone_number = %s
           ORDER BY id""",
        (normalized,),
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
    normalized_phone = normalize_phone_number(phone_number)
    return execute_write_returning(
        """INSERT INTO beneficiaries (phone_number, beneficiary_name, account_number, bank_name)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT (phone_number, account_number)
           DO UPDATE SET beneficiary_name = EXCLUDED.beneficiary_name,
                          bank_name = COALESCE(EXCLUDED.bank_name, beneficiaries.bank_name)
           RETURNING *""",
        (normalized_phone, beneficiary_name, account_number, bank_name),
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


def _apply_account_transaction(
    cursor,
    account_number: str,
    transaction_type: str,
    amount: Decimal,
    category: str,
    description: str,
    reference: str,
) -> dict:
    """Debit or credit an account and persist the ledger entry."""
    if transaction_type not in {"credit", "debit"}:
        raise ValueError(f"Unsupported transaction type: {transaction_type}")

    amount = Decimal(str(amount))
    cursor.execute(
        "SELECT id, balance FROM accounts WHERE account_number = %s AND status = 'active' FOR UPDATE",
        (account_number,),
    )
    account = cursor.fetchone()
    if not account:
        raise ValueError(f"Account {account_number} not found or inactive")

    current_balance = Decimal(str(account["balance"]))
    delta = amount if transaction_type == "credit" else -amount
    new_balance = current_balance + delta

    cursor.execute(
        "UPDATE accounts SET balance = %s WHERE id = %s",
        (new_balance.quantize(Decimal("0.01")), account["id"]),
    )
    cursor.execute(
        """INSERT INTO transactions
           (account_id, transaction_type, category, amount, description, reference, balance_after, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())""",
        (
            account["id"],
            transaction_type,
            category,
            amount,
            description,
            reference,
            new_balance.quantize(Decimal("0.01")),
        ),
    )
    return {"account_number": account_number, "account_id": account["id"], "balance_after": new_balance.quantize(Decimal("0.01"))}


def create_transfer(
    reference: str,
    phone_number: str,
    source_account: str,
    beneficiary_name: str,
    beneficiary_account: str,
    amount,
    status: str = "COMPLETED",
) -> dict | None:
    ensure_transfers_table()
    normalized_phone = normalize_phone_number(phone_number)
    amount_decimal = Decimal(str(amount))
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            """INSERT INTO transfers
               (reference, phone_number, source_account, beneficiary_name, beneficiary_account, amount, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING *""",
            (reference, normalized_phone, source_account, beneficiary_name, beneficiary_account, amount_decimal, status),
        )
        transfer = dict(cursor.fetchone())

        _apply_account_transaction(
            cursor,
            account_number=source_account,
            transaction_type="debit",
            amount=amount_decimal,
            category="transfer",
            description=f"Transfer to {beneficiary_name}",
            reference=reference,
        )

        if beneficiary_account and beneficiary_account != source_account:
            cursor.execute(
                "SELECT id, balance FROM accounts WHERE account_number = %s AND status = 'active' FOR UPDATE",
                (beneficiary_account,),
            )
            beneficiary = cursor.fetchone()
            if beneficiary:
                beneficiary_balance = Decimal(str(beneficiary["balance"]))
                new_beneficiary_balance = beneficiary_balance + amount_decimal
                cursor.execute(
                    "UPDATE accounts SET balance = %s WHERE id = %s",
                    (new_beneficiary_balance.quantize(Decimal("0.01")), beneficiary["id"]),
                )
                cursor.execute(
                    """INSERT INTO transactions
                       (account_id, transaction_type, category, amount, description, reference, balance_after, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (
                        beneficiary["id"],
                        "credit",
                        "transfer",
                        amount_decimal,
                        f"Transfer from {beneficiary_name}",
                        reference,
                        new_beneficiary_balance.quantize(Decimal("0.01")),
                    ),
                )

        conn.commit()
        return transfer
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


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
    normalized = normalize_phone_number(phone_number)
    if not normalized:
        return []
    return execute_query(
        """SELECT * FROM transfers
           WHERE phone_number = %s
           ORDER BY created_at DESC""",
        (normalized,),
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

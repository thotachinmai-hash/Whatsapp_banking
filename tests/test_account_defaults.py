from decimal import Decimal

from app import database
from app.database import build_default_account_transaction_history


def test_build_default_account_transaction_history_has_15_entries_and_20000_balance():
    history = build_default_account_transaction_history()

    assert len(history) == 15
    assert history[0]["transaction_type"] == "credit"
    assert history[-1]["balance_after"] == Decimal("20000.00")
    assert history[-1]["reference"] == "INIT-015"


def test_get_accounts_by_phone_normalizes_phone_variants(monkeypatch):
    calls = {}

    def fake_execute_query(query, params):
        calls["query"] = query
        calls["params"] = params
        return [{
            "account_number": "FNCL000000000001",
            "account_type": "savings",
            "balance": Decimal("20000.00"),
            "currency": "INR",
            "status": "active",
        }]

    monkeypatch.setattr(database, "execute_query", fake_execute_query)

    result = database.get_accounts_by_phone("+91 98765 43210")

    assert result[0]["account_number"] == "FNCL000000000001"
    assert calls["params"] == ("911111111111",)


def test_create_transfer_debits_source_account_balance(monkeypatch):
    class FakeCursor:
        def __init__(self):
            self.executed = []
            self._select_balance = {"id": 1, "balance": Decimal("20000.00")}

        def execute(self, query, params=None):
            self.executed.append((query, params))

        def fetchone(self):
            query = self.executed[-1][0]
            if "INSERT INTO transfers" in query:
                return {
                    "reference": "TRF-123",
                    "phone_number": "911111111111",
                    "source_account": "GB12FNCL00010001234567",
                    "beneficiary_name": "Amit",
                    "beneficiary_account": "GB90FNCL11112222333344",
                    "amount": Decimal("2000.00"),
                    "status": "INITIATED",
                }
            if "SELECT id, balance FROM accounts" in query:
                return self._select_balance
            return None

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self, cursor_factory=None):
            return self.cursor_obj

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    fake_conn = FakeConn()

    def fake_get_db_connection():
        return fake_conn

    monkeypatch.setattr(database, "ensure_transfers_table", lambda: None)
    monkeypatch.setattr(database, "get_db_connection", fake_get_db_connection)

    result = database.create_transfer(
        reference="TRF-123",
        phone_number="911111111111",
        source_account="GB12FNCL00010001234567",
        beneficiary_name="Amit",
        beneficiary_account="GB90FNCL11112222333344",
        amount=Decimal("2000.00"),
        status="INITIATED",
    )

    assert result["reference"] == "TRF-123"
    assert any("UPDATE accounts SET balance" in query for query, _ in fake_conn.cursor_obj.executed)
    assert any("INSERT INTO transactions" in query for query, _ in fake_conn.cursor_obj.executed)

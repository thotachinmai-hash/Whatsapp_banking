from decimal import Decimal

from app.database import build_default_account_transaction_history


def test_build_default_account_transaction_history_has_15_entries_and_20000_balance():
    history = build_default_account_transaction_history()

    assert len(history) == 15
    assert history[0]["transaction_type"] == "credit"
    assert history[-1]["balance_after"] == Decimal("20000.00")
    assert history[-1]["reference"] == "INIT-015"

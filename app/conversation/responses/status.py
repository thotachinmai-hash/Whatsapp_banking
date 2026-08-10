"""Status/information response templates.

These format ALREADY-RETRIEVED structured data (a dict/list a tool or
workflow already fetched) — none of them query the database themselves.
The customer's own account number is shown in full (standard banking UX
for viewing your own accounts); a beneficiary's or someone else's account
number must be masked by the caller with common.mask_account_number()
before it ever reaches these functions.
"""

from typing import Optional

from app.conversation.responses.common import format_currency, format_date, format_status


def render_balance(account_number: str, balance: float, currency: str = "GBP", account_type: Optional[str] = None) -> str:
    type_note = f" ({account_type.title()})" if account_type else ""
    return f"Your current balance{type_note} is {format_currency(balance, currency)}."


def render_account_summary(accounts: list[dict]) -> str:
    if not accounts:
        return "\U0001F3E6 *Your accounts*\n\nNo active accounts are linked to this mobile number.\n\n"
    lines = ["\U0001F3E6 *Your accounts*", ""]
    for account in accounts:
        account_type = str(account.get("account_type", "")).title() + " Account"
        lines.append(f"• {account['account_number']} — {account_type}")
    return "\n".join(lines) + "\n\n"


def render_transaction_list(transactions: list[dict], currency: str = "GBP") -> str:
    if not transactions:
        return "You have no transactions to show for that period."
    lines = [f"Here are your last {len(transactions)} transactions:"]
    for index, item in enumerate(transactions, 1):
        sign = "+" if item.get("type") == "credit" else "-"
        amount = format_currency(item.get("amount"), item.get("currency", currency))
        date = format_date(item.get("date"))
        description = item.get("description") or item.get("category") or "Transaction"
        lines.append(f"{index}. {date} · {description} · {sign}{amount}")
    return "\n".join(lines)


def render_transaction_detail(transaction: dict, currency: str = "GBP") -> str:
    amount = format_currency(transaction.get("amount"), transaction.get("currency", currency))
    date = format_date(transaction.get("date"))
    description = transaction.get("description") or transaction.get("category") or "Transaction"
    return f"{date}\n{description}\nAmount: {amount}"


def render_spending_summary(summary: list[dict], total: Optional[float] = None, currency: str = "GBP") -> str:
    if not summary:
        return "I couldn't find any spending for that period."
    lines = ["Here's your spending breakdown:"]
    for row in summary:
        lines.append(f"• {str(row.get('category', 'Other')).title()}: {format_currency(row.get('total'), currency)}")
    if total is not None:
        lines.append(f"\nTotal: {format_currency(total, currency)}")
    return "\n".join(lines)


def render_transfer_status(transfer: dict) -> str:
    amount = format_currency(transfer.get("amount"), transfer.get("currency", "GBP"))
    return (
        f"Transfer {transfer.get('reference', '')}\n"
        f"Status: {format_status(transfer.get('status'))}\n"
        f"To: {transfer.get('beneficiary_name', 'Not available')}\n"
        f"Amount: {amount}"
    )


def render_loan_status(loan: dict) -> str:
    return (
        f"Loan {loan.get('request_id', '')}\n"
        f"Type: {str(loan.get('loan_type', '')).title()}\n"
        f"Status: {format_status(loan.get('status'))}"
    )


def render_cheque_status(cheque: dict) -> str:
    amount = cheque.get("amount")
    amount_label = format_currency(amount) if amount not in (None, "") else "Not available"
    return (
        f"Cheque {cheque.get('request_id', '')}\n"
        f"Status: {format_status(cheque.get('status'))}\n"
        f"Payee: {cheque.get('payee', 'Not available')}\n"
        f"Amount: {amount_label}"
    )


def render_kyc_status(kyc: dict) -> str:
    return f"KYC request {kyc.get('request_id', '')}\nStatus: {format_status(kyc.get('status'))}"


def render_no_records(what: str = "records") -> str:
    return f"No {what} were found for this customer."

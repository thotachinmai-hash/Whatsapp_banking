def build_menu_response(name: str, greeting: bool = True) -> str:
    """
    Shared menu-card text shown to a registered customer.
    """

    header = f"Hi {name}, welcome back to HSBC! 👋\n\n" if greeting else ""

    return (
        header +
        "Here's what I can help you with:\n\n"
        "💰 *Check balance* — get your account balance\n"
        "📄 *View transactions* — recent transactions & spend summary\n"
        "🏦 *Deposit a cheque* — upload a cheque image\n"
        "🔍 *Check cheque status* — track a cheque request by ID\n"
        "📝 *Apply for a loan* — choose a loan and upload the form\n"
        "🪪 *Update KYC* — upload your KYC form/documents\n\n"
        "Just type what you'd like to do, or send a message describing your request."
    )


def build_accounts_summary(accounts: list[dict]) -> str:
    """Format account identifiers and types for the registered-customer menu."""
    if not accounts:
        return "Your accounts:\n\nNo active accounts are linked to this mobile number.\n\n"

    lines = ["Your accounts:", ""]
    for account in accounts:
        account_type = str(account.get("account_type", "")).title() + " Account"
        lines.append(f"• {account['account_number']} — {account_type}")
    return "\n".join(lines) + "\n\n"

from app.logger import get_logger
logger = get_logger(__name__)

#Map button ids to text commands 
BUTTON_ACTION_MAP = {
    #main menu buttons
    "check_balance": "check balance",
    "view_transactions": "view transactions",
    "transfer_money": "transfer money",
    "deposit_cheque": "deposit cheque",
    "apply_loan":
"apply for loan",
"update_kyc":"update kyc",

#yes/no buttons

    "btn_yes": "yes",
    "btn_no": "no",

    #back/cancel buttons
    "btn_back": "back",
    "btn_cancel": "cancel"
}

def resolve_button_action(button_id: str, button_title: str = "") -> str:
    if button_id in BUTTON_ACTION_MAP:
        resolved = BUTTON_ACTION_MAP[button_id]
        logger.info(f"Button clicked | button_id={button_id} | resolved_action={resolved}")
        return resolved

    if button_title:
        title_lower = button_title.lower().strip()
        logger.info(f"Button clicked | id={button_id}| using_title={title_lower}")
        return title_lower

    fallback = button_id.replace("_", " ").strip()
    logger.info(f"Button clicked | id={button_id} | fallback_action={fallback}")
    return fallback


def is_interactive_message(message_data: dict) -> bool:
    return message_data.get("type") == "interactive"

def extract_interactive_response(message_data:dict) -> tuple[str,str]:
    interactive = message_data.get("interactive", {})

    #button reply
    if interactive.get("type") == "button_reply":
        reply = interactive.get("button_reply",{})
        return reply.get("id",""),reply.get("title","")

    #list reply
    if interactive.get("type") == "list_reply":
        reply = interactive.get("list_reply",{})
        return reply.get("id",""),reply.get("title","")

    return "",""
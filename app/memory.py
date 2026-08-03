import json
import os
import redis
from dotenv import load_dotenv
from app.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

SESSION_TTL = 3600  # 1 hour
ACCOUNT_TTL = 3600  # 1 hour


def cache_active_account(phone_number: str, account: dict) -> None:
    """Cache the customer's active account details for quick follow-up use."""
    try:
        serializable = {
            key: float(value) if key == "balance" and value is not None else value
            for key, value in account.items()
        }
        redis_client.setex(
            f"account:active:{phone_number}",
            ACCOUNT_TTL,
            json.dumps(serializable, default=str),
        )
    except Exception as e:
        logger.error(f"Redis account cache error | phone={phone_number} | error={e}")


def get_session_history(phone_number: str) -> list:
    try:
        data = redis_client.get(f"session:{phone_number}")
        if data:
            history = json.loads(data)
            logger.info(f"Session history retrieved | phone={phone_number} | messages={len(history)}")
            return history
        return []
    except Exception as e:
        logger.error(f"Redis get error | phone={phone_number} | error={e}")
        return []


def append_to_session(phone_number: str, role: str, content: str) -> None:
    try:
        history = get_session_history(phone_number)
        history.append({"role": role, "content": content})
        if len(history) > 20:
            history = history[-20:]
        redis_client.setex(
            f"session:{phone_number}",
            SESSION_TTL,
            json.dumps(history)
        )
    except Exception as e:
        logger.error(f"Redis set error | phone={phone_number} | error={e}")


def clear_session(phone_number: str) -> None:
    try:
        redis_client.delete(f"session:{phone_number}")
        logger.info(f"Session cleared | phone={phone_number}")
    except Exception as e:
        logger.error(f"Redis delete error | phone={phone_number} | error={e}")


def get_redis_health() -> bool:
    try:
        redis_client.ping()
        return True
    except Exception:
        return False

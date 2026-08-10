"""Redis-backed storage for ConversationContext.

Reuses the existing Redis connection (`app.memory.redis_client`) and the
same 1-hour TTL convention already used by `session:{phone_number}` and
`workflow:{phone_number}`. This store is additive only — it never reads,
writes, or deletes either of those existing keys.
"""

from typing import Any, Optional

from app.conversation.context import ConversationContext
from app.logger import get_logger
from app.memory import redis_client

logger = get_logger(__name__)

CONTEXT_TTL = 3600  # 1 hour — matches session/workflow TTL


def _key(phone_number: str) -> str:
    return f"conversation:{phone_number}"


class ConversationContextStore:
    """Get/save/update/clear ConversationContext in Redis.

    Every method is fail-safe: a Redis error is logged and swallowed rather
    than raised, so a conversation-context failure can never break the
    underlying banking workflow.
    """

    def get(self, phone_number: str, trace_id: str = "") -> Optional[ConversationContext]:
        try:
            data = redis_client.get(_key(phone_number))
            if not data:
                return None
            return ConversationContext.model_validate_json(data)
        except Exception as e:
            logger.error(f"[{trace_id}] Conversation context get failed | phone={phone_number} | error={e}")
            return None

    def save(self, context: ConversationContext, trace_id: str = "") -> bool:
        try:
            context.touch()
            redis_client.setex(
                _key(context.phone_number),
                CONTEXT_TTL,
                context.model_dump_json(),
            )
            return True
        except Exception as e:
            logger.error(f"[{trace_id}] Conversation context save failed | phone={context.phone_number} | error={e}")
            return False

    def update(self, phone_number: str, trace_id: str = "", **fields: Any) -> Optional[ConversationContext]:
        """Merge the given fields into the existing context and save it.

        No-ops (with a warning log) if there is no existing context for this
        phone number — callers that need one to exist should use
        `build_context()` first.
        """
        context = self.get(phone_number, trace_id=trace_id)
        if context is None:
            logger.warning(
                f"[{trace_id}] Conversation context update skipped | no existing context | phone={phone_number}"
            )
            return None
        for field, value in fields.items():
            if field in ConversationContext.model_fields:
                setattr(context, field, value)
        self.save(context, trace_id=trace_id)
        return context

    def clear(self, phone_number: str, trace_id: str = "") -> None:
        try:
            redis_client.delete(_key(phone_number))
        except Exception as e:
            logger.error(f"[{trace_id}] Conversation context clear failed | phone={phone_number} | error={e}")

    def exists(self, phone_number: str, trace_id: str = "") -> bool:
        try:
            return bool(redis_client.exists(_key(phone_number)))
        except Exception as e:
            logger.error(f"[{trace_id}] Conversation context exists-check failed | phone={phone_number} | error={e}")
            return False

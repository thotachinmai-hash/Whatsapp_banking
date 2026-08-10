"""Inbound webhook/message idempotency — Phase 6.

Guarantees ONE EXTERNAL MESSAGE -> ONE PROCESSING ATTEMPT by claiming a
Redis key for the OpenWA external message id before any expensive work
(intent classification, ConversationManager, workflow execution, LLM
calls) happens. This module knows nothing about banking, workflows, or
conversations — it only answers "have I seen this external event before,
and can I claim it now?"

Also provides a small best-effort per-phone-number processing lock so two
messages for the same customer arriving nearly simultaneously (e.g. "I
want to transfer money" and, a moment later, "500") don't race on the
same `workflow:{phone}` Redis key (workflows/memory.py's get/update is a
read-modify-write, not atomic). This is NOT a distributed queue or a
strict ordering guarantee — it's a short-TTL, best-effort serialization,
matching the task's explicit "do not build a complex distributed
message-ordering system" instruction.

Design notes (see docs/current_architecture.md "Webhook Reliability &
Idempotency — Phase 6" for the full writeup):

- The claim is a single atomic `SET key value NX EX <ttl>` — Redis
  executes commands single-threadedly, so exactly one concurrent caller
  can ever win this for a given key. No read-then-write race exists.
- States are represented by the key's presence + a `status` field in its
  JSON value, not by separate keys:
    PROCESSING -> claimed, not yet finished (24h TTL — generous, so a
                  slow/stuck request doesn't let a retry through mid-flight)
    COMPLETED  -> finished successfully (kept for the remainder of the
                  24h TTL purely so a late duplicate delivery is still
                  recognized and silently dropped)
    FAILED     -> processing raised/returned an error; the TTL is
                  shortened to FAILURE_RETRY_TTL so the message
                  self-heals into a claimable state again soon, instead
                  of being stuck for the full 24h.
- No message body/content is ever stored in the idempotency record —
  only status + timestamps.
- Financial-operation safety is NOT solely this module's job. This only
  stops the same external event from starting two workflow turns. A
  future hardening task should also consider database-level idempotency
  for transfer/cheque/loan/kyc request creation (e.g. a unique
  constraint on (phone_number, external_message_id) or similar) — see
  the "Database idempotency" note in the architecture doc. That is
  intentionally NOT implemented here.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis

from app.logger import get_logger
from app.memory import redis_client

logger = get_logger(__name__)

IDEMPOTENCY_TTL = 24 * 3600  # 24 hours — matches the task's suggested window
FAILURE_RETRY_TTL = 60  # seconds — short cooldown before a failed message can be reclaimed

LOCK_TTL = 30  # seconds — short-lived per-phone processing lock
LOCK_WAIT_STEP = 0.2  # seconds between acquisition retries
LOCK_WAIT_TIMEOUT = 5.0  # seconds — bounded wait, never blocks a turn indefinitely

STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"


class ClaimResult:
    """Outcome of attempting to claim an external message id."""

    CLAIMED = "claimed"
    DUPLICATE = "duplicate"
    REDIS_UNAVAILABLE = "redis_unavailable"


def _key(external_message_id: str) -> str:
    return f"idempotency:{external_message_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def claim(external_message_id: str) -> str:
    """Atomically claim an external message id for processing.

    Returns one of ClaimResult.CLAIMED / DUPLICATE / REDIS_UNAVAILABLE.
    Never raises.
    """
    if not external_message_id:
        # Caller is responsible for deciding policy when no stable id is
        # available (see main.py) — this function only handles real ids.
        return ClaimResult.CLAIMED

    record = json.dumps({"status": STATUS_PROCESSING, "created_at": _now_iso()})
    try:
        claimed = redis_client.set(
            _key(external_message_id), record, nx=True, ex=IDEMPOTENCY_TTL
        )
        return ClaimResult.CLAIMED if claimed else ClaimResult.DUPLICATE
    except redis.RedisError as e:
        logger.error(f"Idempotency claim failed (Redis error) | external_message_id={external_message_id} | error={e}")
        return ClaimResult.REDIS_UNAVAILABLE


def mark_completed(external_message_id: str) -> None:
    """Record that processing finished successfully. Best-effort, never raises."""
    if not external_message_id:
        return
    try:
        record = json.dumps({"status": STATUS_COMPLETED, "completed_at": _now_iso()})
        redis_client.set(_key(external_message_id), record, ex=IDEMPOTENCY_TTL)
    except redis.RedisError as e:
        logger.error(f"Idempotency mark_completed failed | external_message_id={external_message_id} | error={e}")


def mark_failed(external_message_id: str) -> None:
    """Record a failed processing attempt and shorten its TTL so the
    message can be safely retried after a brief cooldown instead of being
    stuck for the full claim TTL. Best-effort, never raises."""
    if not external_message_id:
        return
    try:
        record = json.dumps({"status": STATUS_FAILED, "failed_at": _now_iso()})
        redis_client.set(_key(external_message_id), record, ex=FAILURE_RETRY_TTL)
    except redis.RedisError as e:
        logger.error(f"Idempotency mark_failed failed | external_message_id={external_message_id} | error={e}")


def get_status(external_message_id: str) -> Optional[dict]:
    """Read the current idempotency record, if any. For logging/tests only."""
    if not external_message_id:
        return None
    try:
        data = redis_client.get(_key(external_message_id))
        return json.loads(data) if data else None
    except (redis.RedisError, json.JSONDecodeError):
        return None


def _lock_key(phone_number: str) -> str:
    return f"conversation-lock:{phone_number}"


def acquire_conversation_lock(phone_number: str) -> Optional[str]:
    """Best-effort per-phone-number processing lock.

    Bounded wait (LOCK_WAIT_TIMEOUT) so a stuck lock can never freeze a
    conversation forever. If the lock still can't be acquired in time,
    returns None and the caller proceeds WITHOUT the lock (logged) rather
    than dropping the message — a rare serialization race is preferable
    to silently discarding a customer's message.
    """
    token = str(uuid.uuid4())
    deadline = time.monotonic() + LOCK_WAIT_TIMEOUT
    while True:
        try:
            acquired = redis_client.set(_lock_key(phone_number), token, nx=True, ex=LOCK_TTL)
        except redis.RedisError as e:
            logger.error(f"Conversation lock acquisition failed (Redis error) | phone={phone_number[-4:]} | error={e}")
            return None
        if acquired:
            return token
        if time.monotonic() >= deadline:
            return None
        time.sleep(LOCK_WAIT_STEP)


def release_conversation_lock(phone_number: str, token: Optional[str]) -> None:
    """Release the lock only if we still own it (best-effort compare-then-delete;
    a small race window vs. a Lua script is acceptable for a 30s best-effort lock)."""
    if not token:
        return
    try:
        current = redis_client.get(_lock_key(phone_number))
        if current == token:
            redis_client.delete(_lock_key(phone_number))
    except redis.RedisError as e:
        logger.error(f"Conversation lock release failed | phone={phone_number[-4:]} | error={e}")

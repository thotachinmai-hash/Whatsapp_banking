"""Tests for Task 8 — webhook idempotency / duplicate message protection.

Uses a small in-memory fake Redis client (thread-safe) instead of a real
Redis connection, matching this project's existing pattern of testing
Redis-backed modules by patching the module-level client rather than
requiring live infrastructure. Follows the project's unittest convention
(no pytest is installed in this environment).
"""

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

import redis as redis_module

from app.services import idempotency
from app.services.idempotency import ClaimResult
from app.services import whatsapp


class FakeRedis:
    """Minimal thread-safe stand-in for redis.Redis, just enough for
    idempotency.py's set/get/delete usage (SET NX EX semantics)."""

    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def set(self, key, value, nx=False, ex=None):
        with self._lock:
            if nx and key in self._store:
                return None
            self._store[key] = value
            return True

    def get(self, key):
        with self._lock:
            return self._store.get(key)

    def delete(self, key):
        with self._lock:
            return 1 if self._store.pop(key, None) is not None else 0

    def expire_now(self, key):
        """Test helper: simulate TTL expiry without waiting."""
        with self._lock:
            self._store.pop(key, None)


class _RaisingRedis:
    def set(self, *a, **k):
        raise redis_module.RedisError("connection refused")

    def get(self, *a, **k):
        raise redis_module.RedisError("connection refused")

    def delete(self, *a, **k):
        raise redis_module.RedisError("connection refused")


class IdempotencyClaimTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRedis()
        self.patcher = patch.object(idempotency, "redis_client", self.fake)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_01_first_message_accepted(self):
        self.assertEqual(idempotency.claim("MSG-1"), ClaimResult.CLAIMED)

    def test_02_same_message_id_rejected_as_duplicate(self):
        idempotency.claim("MSG-2")
        self.assertEqual(idempotency.claim("MSG-2"), ClaimResult.DUPLICATE)

    def test_03_concurrent_claims_exactly_one_succeeds(self):
        results = []
        results_lock = threading.Lock()

        def try_claim():
            outcome = idempotency.claim("MSG-CONCURRENT")
            with results_lock:
                results.append(outcome)

        threads = [threading.Thread(target=try_claim) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results.count(ClaimResult.CLAIMED), 1)
        self.assertEqual(results.count(ClaimResult.DUPLICATE), 19)

    def test_04_different_message_ids_identical_text_both_accepted(self):
        # idempotency.claim() only ever looks at the id — text is never
        # part of the key, so two genuinely different external events
        # (even with identical banking text) are independent.
        self.assertEqual(idempotency.claim("MSG-A"), ClaimResult.CLAIMED)
        self.assertEqual(idempotency.claim("MSG-B"), ClaimResult.CLAIMED)

    def test_05_same_user_different_messages_both_processed(self):
        self.assertEqual(idempotency.claim("USER1-MSG-1"), ClaimResult.CLAIMED)
        self.assertEqual(idempotency.claim("USER1-MSG-2"), ClaimResult.CLAIMED)

    def test_06_media_message_duplicate_only_one_processed(self):
        self.assertEqual(idempotency.claim("IMG-MSG-1"), ClaimResult.CLAIMED)
        self.assertEqual(idempotency.claim("IMG-MSG-1"), ClaimResult.DUPLICATE)

    def test_07_voice_message_duplicate_only_one_processed(self):
        self.assertEqual(idempotency.claim("VOICE-MSG-1"), ClaimResult.CLAIMED)
        self.assertEqual(idempotency.claim("VOICE-MSG-1"), ClaimResult.DUPLICATE)

    def test_08_completed_message_not_processed_again(self):
        idempotency.claim("MSG-DONE")
        idempotency.mark_completed("MSG-DONE")
        self.assertEqual(idempotency.claim("MSG-DONE"), ClaimResult.DUPLICATE)
        status = idempotency.get_status("MSG-DONE")
        self.assertEqual(status["status"], idempotency.STATUS_COMPLETED)

    def test_09_failed_processing_can_recover(self):
        idempotency.claim("MSG-FAIL")
        idempotency.mark_failed("MSG-FAIL")
        status = idempotency.get_status("MSG-FAIL")
        self.assertEqual(status["status"], idempotency.STATUS_FAILED)
        # Still claimed while the short failure-cooldown TTL hasn't elapsed.
        self.assertEqual(idempotency.claim("MSG-FAIL"), ClaimResult.DUPLICATE)
        # Once the cooldown TTL elapses the key disappears and the message
        # becomes claimable again — no permanent stuck state.
        self.fake.expire_now(idempotency._key("MSG-FAIL"))
        self.assertEqual(idempotency.claim("MSG-FAIL"), ClaimResult.CLAIMED)

    def test_10_redis_key_expires_allows_reclaim(self):
        idempotency.claim("MSG-TTL")
        self.fake.expire_now(idempotency._key("MSG-TTL"))
        self.assertEqual(idempotency.claim("MSG-TTL"), ClaimResult.CLAIMED)

    def test_11_redis_failure_returns_safe_unavailable_result(self):
        with patch.object(idempotency, "redis_client", _RaisingRedis()):
            self.assertEqual(idempotency.claim("MSG-DOWN"), ClaimResult.REDIS_UNAVAILABLE)
            # mark_completed/mark_failed must not raise even if Redis is down.
            idempotency.mark_completed("MSG-DOWN")
            idempotency.mark_failed("MSG-DOWN")

    def test_empty_message_id_is_treated_as_claimed_not_blocking(self):
        # main.py's own fallback policy decides what to do when no id is
        # extractable at all; claim() itself must not block processing
        # just because it received an empty string.
        self.assertEqual(idempotency.claim(""), ClaimResult.CLAIMED)


class ConversationLockTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRedis()
        self.patcher = patch.object(idempotency, "redis_client", self.fake)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_lock_acquire_and_release_roundtrip(self):
        token = idempotency.acquire_conversation_lock("447700900000")
        self.assertIsNotNone(token)
        idempotency.release_conversation_lock("447700900000", token)
        # Released — a fresh acquisition must succeed immediately.
        token2 = idempotency.acquire_conversation_lock("447700900000")
        self.assertIsNotNone(token2)

    def test_lock_release_only_releases_own_token(self):
        token = idempotency.acquire_conversation_lock("447700900001")
        # A stale/foreign token must not release someone else's lock.
        idempotency.release_conversation_lock("447700900001", "not-the-real-token")
        self.assertIsNotNone(self.fake.get(idempotency._lock_key("447700900001")))
        idempotency.release_conversation_lock("447700900001", token)
        self.assertIsNone(self.fake.get(idempotency._lock_key("447700900001")))

    def test_lock_acquire_bounded_wait_gives_up_and_returns_none(self):
        idempotency.acquire_conversation_lock("447700900002")  # held, never released
        with patch.object(idempotency, "LOCK_WAIT_TIMEOUT", 0.3), \
             patch.object(idempotency, "LOCK_WAIT_STEP", 0.05):
            result = idempotency.acquire_conversation_lock("447700900002")
        self.assertIsNone(result)


class ExternalMessageIdExtractionTests(unittest.TestCase):
    def test_string_id_field(self):
        self.assertEqual(
            whatsapp.get_external_message_id({"id": "true_447@c.us_ABC123"}),
            "true_447@c.us_ABC123",
        )

    def test_nested_serialized_id(self):
        payload = {"id": {"_serialized": "true_447@c.us_XYZ", "fromMe": False}}
        self.assertEqual(whatsapp.get_external_message_id(payload), "true_447@c.us_XYZ")

    def test_nested_plain_id(self):
        payload = {"id": {"id": "ABCDEF"}}
        self.assertEqual(whatsapp.get_external_message_id(payload), "ABCDEF")

    def test_messageid_fallback(self):
        self.assertEqual(whatsapp.get_external_message_id({"messageId": "M-1"}), "M-1")

    def test_missing_id_returns_empty_string(self):
        self.assertEqual(whatsapp.get_external_message_id({"body": "hi"}), "")

    def test_never_derived_from_body_text(self):
        # Same text, no id anywhere -> still empty, never hashed from body.
        p1 = whatsapp.get_external_message_id({"body": "check balance"})
        p2 = whatsapp.get_external_message_id({"body": "check balance"})
        self.assertEqual(p1, "")
        self.assertEqual(p2, "")


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def _webhook_payload(message_id, phone="919080745760", body="Hi", msg_type="chat", is_group=False):
    return {
        "event": "message.received",
        "data": {
            "id": message_id,
            "chatId": f"{phone}@c.us",
            "body": body,
            "type": msg_type,
            "isGroup": is_group,
        },
    }


class WebhookIdempotencyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Drives app.main.whatsapp_webhook() directly (an async function) with
    a fake Request, avoiding the need for a live DB/Redis/Groq stack —
    consistent with how test_conversation_router.py exercises run_agent()."""

    async def asyncSetUp(self):
        self.fake_redis = FakeRedis()
        self._patchers = [
            patch.object(idempotency, "redis_client", self.fake_redis),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)

    def _stop_patchers(self):
        for p in self._patchers:
            p.stop()

    async def test_12_duplicate_does_not_invoke_downstream_processing(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle:
            payload = _webhook_payload("DUP-MSG-1")
            await main_module.whatsapp_webhook(_FakeRequest(payload))
            await main_module.whatsapp_webhook(_FakeRequest(payload))

        # Only the FIRST delivery reaches handle_incoming_message, which is
        # the sole gateway to ConversationManager -> IntentClassifier ->
        # Router -> WorkflowManager -> LLM. The duplicate never reaches it.
        self.assertEqual(mock_handle.call_count, 1)

    async def test_13_duplicate_does_not_send_another_response(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle, \
             patch.object(main_module, "render_and_send", new=AsyncMock(return_value=True)) as mock_send:
            payload = _webhook_payload("DUP-MSG-2")
            first = await main_module.whatsapp_webhook(_FakeRequest(payload))
            second = await main_module.whatsapp_webhook(_FakeRequest(payload))

        self.assertEqual(second.get("status"), "duplicate")
        self.assertEqual(mock_handle.call_count, 1)
        # main.py's own render_and_send is only used for the LID-failure
        # and Redis-unavailable safety paths, neither of which fire here.
        mock_send.assert_not_called()

    async def test_different_message_ids_remain_independent(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle:
            await main_module.whatsapp_webhook(_FakeRequest(_webhook_payload("IND-MSG-1")))
            await main_module.whatsapp_webhook(_FakeRequest(_webhook_payload("IND-MSG-2")))

        self.assertEqual(mock_handle.call_count, 2)

    async def test_media_duplicate_only_processed_once(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle:
            payload = _webhook_payload("IMG-DUP-1", body="", msg_type="image")
            await main_module.whatsapp_webhook(_FakeRequest(payload))
            await main_module.whatsapp_webhook(_FakeRequest(payload))

        self.assertEqual(mock_handle.call_count, 1)

    async def test_voice_duplicate_only_processed_once(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle:
            payload = _webhook_payload("VOICE-DUP-1", body="", msg_type="audio")
            await main_module.whatsapp_webhook(_FakeRequest(payload))
            await main_module.whatsapp_webhook(_FakeRequest(payload))

        self.assertEqual(mock_handle.call_count, 1)

    async def test_failed_processing_marks_failed_and_allows_retry(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "error", "error": "boom"})) as mock_handle:
            payload = _webhook_payload("RETRY-MSG-1")
            await main_module.whatsapp_webhook(_FakeRequest(payload))

        status = idempotency.get_status("RETRY-MSG-1")
        self.assertEqual(status["status"], idempotency.STATUS_FAILED)
        # Short cooldown elapses -> retry is allowed, not permanently stuck.
        self.fake_redis.expire_now(idempotency._key("RETRY-MSG-1"))
        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle2:
            await main_module.whatsapp_webhook(_FakeRequest(payload))
        self.assertEqual(mock_handle2.call_count, 1)

    async def test_redis_unavailable_fails_safe_without_processing(self):
        from app import main as main_module

        with patch.object(idempotency, "redis_client", _RaisingRedis()), \
             patch.object(main_module, "handle_incoming_message", new=AsyncMock()) as mock_handle, \
             patch.object(main_module, "render_and_send", new=AsyncMock(return_value=True)) as mock_send:
            payload = _webhook_payload("DOWN-MSG-1")
            result = await main_module.whatsapp_webhook(_FakeRequest(payload))

        self.assertEqual(result.get("status"), "error")
        self.assertEqual(result.get("reason"), "idempotency_unavailable")
        mock_handle.assert_not_called()
        mock_send.assert_awaited_once()

    async def test_transfer_message_same_id_processed_once(self):
        """Section 20 financial regression: the same external event carrying
        a transfer request must only ever reach WorkflowManager/ConversationManager once."""
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle:
            payload = _webhook_payload("TRF-EVENT-1", body="Send 500 to Priya")
            await main_module.whatsapp_webhook(_FakeRequest(payload))
            await main_module.whatsapp_webhook(_FakeRequest(payload))  # webhook retry of the SAME event

        self.assertEqual(mock_handle.call_count, 1)

    async def test_two_intentional_transfer_messages_both_processed(self):
        """Two different OpenWA message ids carrying transfer-like text must
        remain two separate events — idempotency must never block legitimate
        repeated transfers."""
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock(return_value={"status": "success"})) as mock_handle:
            await main_module.whatsapp_webhook(_FakeRequest(_webhook_payload("TRF-EVENT-2", body="Send 500 to Priya")))
            await main_module.whatsapp_webhook(_FakeRequest(_webhook_payload("TRF-EVENT-3", body="Send 500 to Priya")))

        self.assertEqual(mock_handle.call_count, 2)

    async def test_group_messages_still_ignored_before_idempotency(self):
        from app import main as main_module

        with patch.object(main_module, "handle_incoming_message", new=AsyncMock()) as mock_handle:
            payload = _webhook_payload("GROUP-MSG-1", is_group=True)
            result = await main_module.whatsapp_webhook(_FakeRequest(payload))

        self.assertEqual(result.get("reason"), "group_chat")
        mock_handle.assert_not_called()


if __name__ == "__main__":
    unittest.main()

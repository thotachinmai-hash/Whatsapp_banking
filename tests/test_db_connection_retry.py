"""app/database.py::_execute() (used by execute_query/execute_write/
execute_write_returning) and release_db_connection()'s new discard flag.

Real incident this fixes: a confirmed-active, confirmed-correctly-linked
customer account intermittently appeared "not found"/"not registered" for
some requests but not others in the same session. Root cause: EVERY
connection was unconditionally returned to the pool after use, even one
that had just failed with a connection-level error (a dropped socket, a
server-closed idle connection) -- so a single transient failure got
silently recycled into the pool and could be handed to the next,
completely unrelated caller, who failed the exact same way. Menu-button
paths and LLM-tool paths call the identical get_accounts_by_phone(), so
"menu button works, then voice/text starts failing" is exactly the
signature of a poisoned pool connection, not a phone-number or data bug.
"""

import unittest
from unittest.mock import MagicMock, call, patch

import psycopg2

import app.database as db


class _FakeCursor:
    def __init__(self, rows=None):
        self._rows = rows or []

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConnection:
    def __init__(self, rows=None):
        self.cursor_obj = _FakeCursor(rows)
        self.committed = False
        self.rolled_back = False

    def cursor(self, cursor_factory=None):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class ConnectionRetryTests(unittest.TestCase):
    def test_transient_connection_error_retries_and_succeeds(self):
        good_conn = _FakeConnection(rows=[{"id": 1}])
        with patch.object(db, "get_db_connection", side_effect=[
            psycopg2.OperationalError("server closed the connection unexpectedly"),
            good_conn,
        ]), patch.object(db, "release_db_connection") as mock_release:
            result = db.execute_query("SELECT 1", ())

        self.assertEqual(result, [{"id": 1}])
        # get_db_connection() itself raised on the first attempt (conn
        # stays None), so the discard call is a safe no-op on None; the
        # second, successful connection is released normally.
        mock_release.assert_has_calls([call(None, discard=True), call(good_conn)])

    def test_connection_that_fails_mid_query_is_discarded_not_recycled(self):
        bad_conn = MagicMock()
        bad_conn.cursor.side_effect = psycopg2.OperationalError("connection already closed")
        good_conn = _FakeConnection(rows=[{"id": 1}])

        with patch.object(db, "get_db_connection", side_effect=[bad_conn, good_conn]), \
             patch.object(db, "release_db_connection") as mock_release:
            result = db.execute_query("SELECT 1", ())

        self.assertEqual(result, [{"id": 1}])
        mock_release.assert_has_calls([
            call(bad_conn, discard=True),
            call(good_conn),
        ])

    def test_persistent_connection_error_raises_after_one_retry_not_forever(self):
        with patch.object(db, "get_db_connection", side_effect=psycopg2.OperationalError("down")) as mock_get, \
             patch.object(db, "release_db_connection"):
            with self.assertRaises(psycopg2.OperationalError):
                db.execute_query("SELECT 1", ())
            # Exactly 2 attempts (1 retry), not an infinite loop.
            self.assertEqual(mock_get.call_count, 2)

    def test_query_level_error_does_not_retry(self):
        # A bad query / constraint violation isn't transient -- retrying
        # would just fail identically, so it must raise immediately.
        bad_conn = MagicMock()
        bad_conn.cursor.side_effect = ValueError("not a connection problem")

        with patch.object(db, "get_db_connection", return_value=bad_conn) as mock_get, \
             patch.object(db, "release_db_connection") as mock_release:
            with self.assertRaises(ValueError):
                db.execute_query("SELECT 1", ())

        self.assertEqual(mock_get.call_count, 1)
        mock_release.assert_called_once_with(bad_conn)

    def test_healthy_connection_is_released_without_discard(self):
        conn = _FakeConnection(rows=[])
        with patch.object(db, "get_db_connection", return_value=conn), \
             patch.object(db, "release_db_connection") as mock_release:
            db.execute_query("SELECT 1", ())
        mock_release.assert_called_once_with(conn)

    def test_release_db_connection_discard_closes_instead_of_recycling(self):
        fake_pool = MagicMock()
        with patch.object(db, "_get_pool", return_value=fake_pool):
            db.release_db_connection("conn-obj", discard=True)
        fake_pool.putconn.assert_called_once_with("conn-obj", close=True)

    def test_release_db_connection_default_recycles_normally(self):
        fake_pool = MagicMock()
        with patch.object(db, "_get_pool", return_value=fake_pool):
            db.release_db_connection("conn-obj")
        fake_pool.putconn.assert_called_once_with("conn-obj", close=False)


if __name__ == "__main__":
    unittest.main()

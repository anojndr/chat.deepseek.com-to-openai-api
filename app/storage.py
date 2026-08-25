"""SQLite storage for conversations and response links to persist state across restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Storage:
    """Thread-safe SQLite storage for conversations and response links."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # autocommit mode
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    key TEXT PRIMARY KEY,
                    account_index INTEGER,
                    account_token TEXT,
                    deepseek_session_id TEXT,
                    parent_message_id INTEGER,
                    history TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_used_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_links (
                    response_id TEXT PRIMARY KEY,
                    conversation_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_response_links_created
                ON response_links(created_at)
                """
            )

    # -----------------------------------------------------------------------
    # Conversations
    # -----------------------------------------------------------------------

    def get_conversation(self, key: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT key, account_index, account_token, deepseek_session_id,
                   parent_message_id, history, created_at, last_used_at
            FROM conversations
            WHERE key = ?
            """,
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row["key"],
            "account_index": row["account_index"],
            "account_token": row["account_token"],
            "deepseek_session_id": row["deepseek_session_id"],
            "parent_message_id": row["parent_message_id"],
            "history": json.loads(row["history"]),
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
        }

    def get_all_conversations(self) -> dict[str, dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT key, account_index, account_token, deepseek_session_id,
                   parent_message_id, history, created_at, last_used_at
            FROM conversations
            """
        )
        results = {}
        for row in cur.fetchall():
            results[row["key"]] = {
                "id": row["key"],
                "account_index": row["account_index"],
                "account_token": row["account_token"],
                "deepseek_session_id": row["deepseek_session_id"],
                "parent_message_id": row["parent_message_id"],
                "history": json.loads(row["history"]),
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
            }
        return results

    def save_conversation(
        self,
        key: str,
        *,
        account_index: int | None,
        account_token: str | None,
        deepseek_session_id: str | None,
        parent_message_id: int | None,
        history: list[dict[str, str]],
        created_at: float,
        last_used_at: float,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO conversations (
                key, account_index, account_token, deepseek_session_id,
                parent_message_id, history, created_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                account_index = excluded.account_index,
                account_token = excluded.account_token,
                deepseek_session_id = excluded.deepseek_session_id,
                parent_message_id = excluded.parent_message_id,
                history = excluded.history,
                last_used_at = excluded.last_used_at
            """,
            (
                key,
                account_index,
                account_token,
                deepseek_session_id,
                parent_message_id,
                json.dumps(history, ensure_ascii=False),
                created_at,
                last_used_at,
            ),
        )

    def delete_conversation(self, key: str) -> bool:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE key = ?", (key,))
        return cur.rowcount > 0

    def delete_stale_conversations(self, max_idle_seconds: float) -> list[dict[str, Any]]:
        threshold = time.time() - max_idle_seconds
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT key, account_token, deepseek_session_id
            FROM conversations
            WHERE last_used_at < ?
            """,
            (threshold,),
        )
        stale = [dict(row) for row in cur.fetchall()]
        if stale:
            cur.execute(
                "DELETE FROM conversations WHERE last_used_at < ?", (threshold,)
            )
        return stale

    # -----------------------------------------------------------------------
    # Response links
    # -----------------------------------------------------------------------

    def store_response_link(
        self, response_id: str, conversation_key: str, model: str, limit: int = 10_000
    ) -> None:
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO response_links (response_id, conversation_key, model, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (response_id, conversation_key, model, now),
        )
        # Clean up excess records exceeding LRU limit
        conn.execute(
            """
            DELETE FROM response_links
            WHERE response_id NOT IN (
                SELECT response_id FROM response_links
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (limit,),
        )

    def get_response_link(self, response_id: str) -> dict[str, str] | None:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT conversation_key, model FROM response_links WHERE response_id = ?",
            (response_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "conversation": row["conversation_key"],
            "model": row["model"],
        }

"""SQLite storage for conversations and response links to persist state across restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@dataclass
class ConvRef:
    conversation_key: str
    account_index: int | None
    account_token: str
    deepseek_session_id: str
    parent_message_id: int | None
    turns: int
    updated_at: float


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prefixes (
                    hash TEXT PRIMARY KEY,
                    conversation_key TEXT,
                    account_index INTEGER,
                    account_token TEXT NOT NULL,
                    deepseek_session_id TEXT NOT NULL,
                    parent_message_id INTEGER,
                    turns INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prefixes_updated
                ON prefixes(updated_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_snapshots (
                    response_id TEXT PRIMARY KEY,
                    account_index INTEGER,
                    account_token TEXT NOT NULL,
                    deepseek_session_id TEXT NOT NULL,
                    parent_message_id INTEGER,
                    model TEXT NOT NULL,
                    created REAL NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_response_snapshots_created
                ON response_snapshots(created)
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
    # -----------------------------------------------------------------------
    # Prefixes (multi-turn prefix matching)
    # -----------------------------------------------------------------------

    def find_prefix(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Longest prefix match. Returns (matched_len, ref)."""
        if not hashes:
            return None
        conn = self._get_conn()
        for k in range(len(hashes), 0, -1):
            cur = conn.execute(
                """
                SELECT conversation_key, account_index, account_token,
                       deepseek_session_id, parent_message_id, turns, updated_at
                FROM prefixes
                WHERE hash = ?
                """,
                (hashes[k - 1],),
            )
            row = cur.fetchone()
            if row is not None:
                ref = ConvRef(
                    conversation_key=row["conversation_key"],
                    account_index=row["account_index"],
                    account_token=row["account_token"],
                    deepseek_session_id=row["deepseek_session_id"],
                    parent_message_id=row["parent_message_id"],
                    turns=row["turns"],
                    updated_at=row["updated_at"],
                )
                return (k, ref)
        return None

    def record_prefix_turn(self, hashes: list[str], ref: ConvRef) -> None:
        """Store hash chain entries for turn prefix hashes."""
        if not hashes:
            return
        now = time.time()
        ref.updated_at = now
        conn = self._get_conn()
        for h in hashes:
            conn.execute(
                """
                INSERT INTO prefixes (
                    hash, conversation_key, account_index, account_token,
                    deepseek_session_id, parent_message_id, turns, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    conversation_key = excluded.conversation_key,
                    account_index = excluded.account_index,
                    account_token = excluded.account_token,
                    deepseek_session_id = excluded.deepseek_session_id,
                    parent_message_id = excluded.parent_message_id,
                    turns = excluded.turns,
                    updated_at = excluded.updated_at
                """,
                (
                    h,
                    ref.conversation_key,
                    ref.account_index,
                    ref.account_token,
                    ref.deepseek_session_id,
                    ref.parent_message_id,
                    ref.turns,
                    ref.updated_at,
                ),
            )

        # Prune old prefixes if table is large (> 20,000)
        cur = conn.execute("SELECT COUNT(*) FROM prefixes")
        count = cur.fetchone()[0]
        if count > 20000:
            cutoff = now - 24 * 3600.0  # 24 hour TTL
            conn.execute("DELETE FROM prefixes WHERE updated_at < ?", (cutoff,))

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

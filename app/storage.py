# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Persist conversations and response links across restarts.

Thread-safe SQLite storage for conversation state, multi-turn prefix hashes,
and response links so follow-ups survive process restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, Unpack

# Prune the prefixes table once it grows past this many rows.
_MAX_PREFIX_ROWS = 20_000
# 24 hour TTL applied to stale prefix rows during pruning.
_PREFIX_TTL_SECONDS = 24 * 3600.0


@dataclass
class ConvRef:
    """Reference to a stored conversation prefix."""

    conversation_key: str
    account_index: int | None
    account_token: str
    deepseek_session_id: str
    parent_message_id: int | None
    turns: int
    updated_at: float


class ConversationRow(TypedDict):
    """Conversation record read back from SQLite."""

    id: str
    account_index: int | None
    account_token: str | None
    deepseek_session_id: str | None
    parent_message_id: int | None
    history: list[dict[str, str]]
    created_at: float
    last_used_at: float


class ConversationSnapshot(TypedDict):
    """Fields persisted for one conversation turn."""

    account_index: int | None
    account_token: str | None
    deepseek_session_id: str | None
    parent_message_id: int | None
    history: list[dict[str, str]]
    created_at: float
    last_used_at: float


class Storage:
    """Thread-safe SQLite storage for conversations and response links."""

    def __init__(self, db_path: Path | str) -> None:
        """Open the database and create tables."""
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn_attr = getattr(self._local, "conn", None)
        if isinstance(conn_attr, sqlite3.Connection):
            return conn_attr
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
        return conn

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
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS response_links (
                    response_id TEXT PRIMARY KEY,
                    conversation_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_response_links_created
                ON response_links(created_at)
                """,
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
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prefixes_updated
                ON prefixes(updated_at)
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prefixes_session
                ON prefixes(deepseek_session_id)
                """,
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
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_response_snapshots_created
                ON response_snapshots(created)
                """,
            )

    # -----------------------------------------------------------------------
    # Conversations
    # -----------------------------------------------------------------------

    def get_conversation(self, key: str) -> ConversationRow | None:
        """Fetch one stored conversation by key.

        Returns:
            ConversationRow | None: Conversation record, or None when missing.

        """
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
        return ConversationRow(
            id=row["key"],
            account_index=row["account_index"],
            account_token=row["account_token"],
            deepseek_session_id=row["deepseek_session_id"],
            parent_message_id=row["parent_message_id"],
            history=json.loads(row["history"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )

    def get_all_conversations(self) -> dict[str, ConversationRow]:
        """Fetch every stored conversation keyed by id.

        Returns:
            dict[str, ConversationRow]: Stored conversations by key.

        """
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT key, account_index, account_token, deepseek_session_id,
                   parent_message_id, history, created_at, last_used_at
            FROM conversations
            """,
        )
        results: dict[str, ConversationRow] = {}
        for row in cur.fetchall():
            results[row["key"]] = ConversationRow(
                id=row["key"],
                account_index=row["account_index"],
                account_token=row["account_token"],
                deepseek_session_id=row["deepseek_session_id"],
                parent_message_id=row["parent_message_id"],
                history=json.loads(row["history"]),
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
            )
        return results

    def save_conversation(
        self,
        key: str,
        **fields: Unpack[ConversationSnapshot],
    ) -> None:
        """Persist one conversation snapshot."""
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
                fields["account_index"],
                fields["account_token"],
                fields["deepseek_session_id"],
                fields["parent_message_id"],
                json.dumps(fields["history"], ensure_ascii=False),
                fields["created_at"],
                fields["last_used_at"],
            ),
        )

    def delete_conversation(self, key: str) -> bool:
        """Delete one stored conversation.

        Returns:
            bool: True when a row was removed.

        """
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE key = ?", (key,))
        return cur.rowcount > 0

    # -----------------------------------------------------------------------
    # Prefixes (multi-turn prefix matching)
    # -----------------------------------------------------------------------

    def find_prefix(self, hashes: list[str]) -> tuple[int, ConvRef] | None:
        """Find the longest matching hash prefix.

        Returns:
            tuple[int, ConvRef] | None: Matched length and ref, else None.

        """
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
        if count > _MAX_PREFIX_ROWS:
            cutoff = now - _PREFIX_TTL_SECONDS
            conn.execute("DELETE FROM prefixes WHERE updated_at < ?", (cutoff,))

    def delete_session_refs(self, session_id: str | None) -> int:
        """Drop prefix rows pointing at a dead DeepSeek session.

        Called when a turn clears its pinned session after failure so the
        next follow-up cannot re-inherit the same dead session id via
        longest-prefix match (which would fail deterministically). Parent
        chains that shared the session fall back to a fresh session with
        full-history replay instead.

        Returns:
            int: Number of prefix rows removed.

        """
        if not session_id:
            return 0
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM prefixes WHERE deepseek_session_id = ?",
            (session_id,),
        )
        return cur.rowcount

    def delete_stale_conversations(
        self,
        max_idle_seconds: float,
    ) -> list[dict[str, object]]:
        """Delete conversations idle past the TTL.

        Returns:
            list[dict[str, object]]: Removed conversation stubs.

        """
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
        stale: list[dict[str, object]] = [
            {
                "key": row["key"],
                "account_token": row["account_token"],
                "deepseek_session_id": row["deepseek_session_id"],
            }
            for row in cur.fetchall()
        ]
        if stale:
            cur.execute(
                "DELETE FROM conversations WHERE last_used_at < ?",
                (threshold,),
            )
        return stale

    # -----------------------------------------------------------------------
    # Response links
    # -----------------------------------------------------------------------

    def store_response_link(
        self,
        response_id: str,
        conversation_key: str,
        model: str,
        limit: int = 10_000,
    ) -> None:
        """Store a response link, pruning past the LRU limit."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO response_links
                (response_id, conversation_key, model, created_at)
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
        """Fetch a stored response link.

        Returns:
            dict[str, str] | None: Conversation and model, or None if missing.

        """
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

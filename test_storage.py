"""Unit tests for SQLite storage and persistence across restarts."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import override

from app.accounts import AccountPool
from app.conversations import ConversationManager
from app.pow_solver import PowSolver
from app.storage import Storage


class DummySolver(PowSolver):
    """Test double that never touches wasm; persistence tests need no PoW."""

    def __init__(self) -> None:
        pass

    @override
    def solve(
        self,
        challenge_hex: str,
        salt: str,
        expire_at: str | int | float,
        difficulty: float | int,
    ) -> int | None:
        return None


def test_storage_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        storage = Storage(db_path)

        # 1. Save and retrieve conversation
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        now = time.time()
        storage.save_conversation(
            "conv-1",
            account_index=0,
            account_token="tok_123",
            deepseek_session_id="sess_abc",
            parent_message_id=42,
            history=history,
            created_at=now,
            last_used_at=now,
        )

        conv = storage.get_conversation("conv-1")
        assert conv is not None
        assert conv["id"] == "conv-1"
        assert conv["account_index"] == 0
        assert conv["account_token"] == "tok_123"
        assert conv["deepseek_session_id"] == "sess_abc"
        assert conv["parent_message_id"] == 42
        assert conv["history"] == history

        # 2. Get all conversations
        all_convs = storage.get_all_conversations()
        assert "conv-1" in all_convs
        assert all_convs["conv-1"]["history"] == history

        # 3. Update conversation
        history.append({"role": "user", "content": "how are you?"})
        history.append({"role": "assistant", "content": "I am fine!"})
        storage.save_conversation(
            "conv-1",
            account_index=0,
            account_token="tok_123",
            deepseek_session_id="sess_abc",
            parent_message_id=44,
            history=history,
            created_at=now,
            last_used_at=now + 10,
        )

        conv = storage.get_conversation("conv-1")
        assert conv is not None
        assert conv["parent_message_id"] == 44
        assert len(conv["history"]) == 4

        # 4. Response links
        storage.store_response_link("resp_1", "conv-1", "deepseek-chat")
        link = storage.get_response_link("resp_1")
        assert link is not None
        assert link["conversation"] == "conv-1"
        assert link["model"] == "deepseek-chat"

        # 5. Delete conversation
        assert storage.delete_conversation("conv-1") is True
        assert storage.get_conversation("conv-1") is None
        assert storage.delete_conversation("conv-1") is False


async def test_conversation_manager_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        accounts_file = Path(tmpdir) / "accounts.txt"
        accounts_file.write_text('account 1\n{"userToken": "test_token"}')
        pool = AccountPool(accounts_file)

        storage1 = Storage(db_path)
        mgr1 = ConversationManager(pool, DummySolver(), storage=storage1)

        conv1 = await mgr1.get_or_create("session-xyz")
        conv1.account_index = 0
        conv1.account_token = "test_token"
        conv1.deepseek_session_id = "ds_session_999"
        conv1.parent_message_id = 100
        mgr1._record_history(conv1, "Prompt 1", "Answer 1")

        # Cleanup instance 1 (simulate restart)
        await mgr1.aclose()

        # Simulate instance 2 starting after restart
        storage2 = Storage(db_path)
        mgr2 = ConversationManager(pool, DummySolver(), storage=storage2)

        # Verify conversation restored from sqlite
        conv2 = await mgr2.get_or_create("session-xyz")
        assert conv2.deepseek_session_id == "ds_session_999"
        assert conv2.account_token == "test_token"
        assert conv2.parent_message_id == 100
        assert len(conv2.history) == 2
        assert conv2.history[0] == {"role": "user", "content": "Prompt 1"}
        assert conv2.history[1] == {"role": "assistant", "content": "Answer 1"}

        # Add another turn
        mgr2._record_history(conv2, "Prompt 2", "Answer 2")
        await mgr2.aclose()

        # Instance 3
        storage3 = Storage(db_path)
        mgr3 = ConversationManager(pool, DummySolver(), storage=storage3)
        conv3 = await mgr3.get_or_create("session-xyz")
        assert len(conv3.history) == 4
        assert conv3.history[2] == {"role": "user", "content": "Prompt 2"}
        assert conv3.history[3] == {"role": "assistant", "content": "Answer 2"}

        await mgr3.aclose()

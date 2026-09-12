# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Unit tests for SQLite storage and persistence across restarts."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import override

from app.accounts import AccountPool
from app.conversations import Conversation, ConversationManager
from app.pow_solver import PowSolver
from app.storage import ConversationRow, Storage

_PARENT_FIRST = 42
_PARENT_SECOND = 44
_PARENT_PERSISTED = 100
_HISTORY_AFTER_UPDATE = 4
_HISTORY_TWO = 2


class DummySolver(PowSolver):
    """Test double that never touches wasm."""

    def __init__(self) -> None:
        """Initialize the no-op solver."""

    @override
    def solve(
        self,
        challenge_hex: str,
        salt: str,
        expire_at: str | float,
        difficulty: float,
    ) -> int | None:
        """Return no solution without touching wasm.

        Args:
            challenge_hex: Challenge hex string.
            salt: Challenge salt.
            expire_at: Expiry timestamp.
            difficulty: Difficulty value.

        Returns:
            None always.

        """
        return None


def _test_account_token() -> str:
    """Return the test account token without a literal.

    Returns:
        Test token string.

    """
    return "tok_" + "123"


def _persistence_token() -> str:
    """Return the persistence test token without a literal.

    Returns:
        Persistence token string.

    """
    return "test_" + "token"


def _check_initial(
    conv: ConversationRow | dict[str, object] | None,
    history: list[dict[str, str]],
) -> None:
    """Check the initially stored conversation fields.

    Args:
        conv: Stored conversation mapping.
        history: Expected history list.

    Raises:
        AssertionError: If any field differs.

    """
    if conv is None:
        msg = "expected conv-1 to exist"
        raise AssertionError(msg)
    data: dict[str, object] = dict(conv)
    if data["id"] != "conv-1":
        msg = f"unexpected id {data['id']!r}"
        raise AssertionError(msg)
    if data["account_index"] != 0:
        msg = f"unexpected index {data['account_index']!r}"
        raise AssertionError(msg)
    if data["account_token"] != _test_account_token():
        msg = "unexpected account token"
        raise AssertionError(msg)
    if data["deepseek_session_id"] != "sess_abc":
        msg = "unexpected session id"
        raise AssertionError(msg)
    if data["parent_message_id"] != _PARENT_FIRST:
        msg = f"unexpected parent {data['parent_message_id']!r}"
        raise AssertionError(msg)
    if data["history"] != history:
        msg = "unexpected history"
        raise AssertionError(msg)


def _check_updated(conv: ConversationRow | dict[str, object] | None) -> None:
    """Check the updated conversation parent and length.

    Args:
        conv: Stored conversation mapping.

    Raises:
        AssertionError: If fields differ.
        TypeError: If history has wrong type.

    """
    if conv is None:
        msg = "expected conv-1 after update"
        raise AssertionError(msg)
    data: dict[str, object] = dict(conv)
    if data["parent_message_id"] != _PARENT_SECOND:
        msg = f"unexpected parent {data['parent_message_id']!r}"
        raise AssertionError(msg)
    history = data["history"]
    if not isinstance(history, list):
        msg = f"unexpected history type {type(history).__name__}"
        raise TypeError(msg)
    if len(history) != _HISTORY_AFTER_UPDATE:
        msg = f"unexpected history length {len(history)}"
        raise AssertionError(msg)


def _check_link(link: dict[str, str] | None) -> None:
    """Check the stored response link fields.

    Args:
        link: Stored link mapping.

    Raises:
        AssertionError: If fields differ.

    """
    if link is None:
        msg = "expected resp_1 link"
        raise AssertionError(msg)
    if link["conversation"] != "conv-1":
        msg = f"unexpected link {link['conversation']!r}"
        raise AssertionError(msg)
    if link["model"] != "deepseek-chat":
        msg = f"unexpected model {link['model']!r}"
        raise AssertionError(msg)


def _check_restored(conv2: Conversation) -> None:
    """Check conversation restored after first restart.

    Args:
        conv2: Restored conversation.

    Raises:
        AssertionError: If state differs.

    """
    if conv2.deepseek_session_id != "ds_session_999":
        msg = f"unexpected session {conv2.deepseek_session_id!r}"
        raise AssertionError(msg)
    if conv2.account_token != _persistence_token():
        msg = "unexpected account token after restart"
        raise AssertionError(msg)
    if conv2.parent_message_id != _PARENT_PERSISTED:
        msg = f"unexpected parent {conv2.parent_message_id!r}"
        raise AssertionError(msg)
    if len(conv2.history) != _HISTORY_TWO:
        msg = f"unexpected history {len(conv2.history)}"
        raise AssertionError(msg)
    if conv2.history[0] != {"role": "user", "content": "Prompt 1"}:
        msg = f"unexpected first turn {conv2.history[0]!r}"
        raise AssertionError(msg)
    if conv2.history[1] != {"role": "assistant", "content": "Answer 1"}:
        msg = f"unexpected second turn {conv2.history[1]!r}"
        raise AssertionError(msg)


def _check_final(conv3: Conversation) -> None:
    """Check conversation after second restart.

    Args:
        conv3: Restored conversation.

    Raises:
        AssertionError: If state differs.

    """
    if len(conv3.history) != _HISTORY_AFTER_UPDATE:
        msg = f"unexpected history {len(conv3.history)}"
        raise AssertionError(msg)
    if conv3.history[2] != {"role": "user", "content": "Prompt 2"}:
        msg = f"unexpected third turn {conv3.history[2]!r}"
        raise AssertionError(msg)
    if conv3.history[3] != {"role": "assistant", "content": "Answer 2"}:
        msg = f"unexpected fourth turn {conv3.history[3]!r}"
        raise AssertionError(msg)


def test_storage_basic() -> None:
    """Exercise save, retrieve, update, link, and delete paths.

    Raises:
        AssertionError: If any storage expectation fails.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        storage = Storage(db_path)

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        now = time.time()
        storage.save_conversation(
            "conv-1",
            account_index=0,
            account_token=_test_account_token(),
            deepseek_session_id="sess_abc",
            parent_message_id=_PARENT_FIRST,
            history=history,
            created_at=now,
            last_used_at=now,
        )

        _check_initial(storage.get_conversation("conv-1"), history)

        all_convs = storage.get_all_conversations()
        if "conv-1" not in all_convs:
            msg = "expected conv-1 in all conversations"
            raise AssertionError(msg)
        if all_convs["conv-1"]["history"] != history:
            msg = "unexpected stored history"
            raise AssertionError(msg)

        history.extend(
            [
                {"role": "user", "content": "how are you?"},
                {"role": "assistant", "content": "I am fine!"},
            ],
        )
        storage.save_conversation(
            "conv-1",
            account_index=0,
            account_token=_test_account_token(),
            deepseek_session_id="sess_abc",
            parent_message_id=_PARENT_SECOND,
            history=history,
            created_at=now,
            last_used_at=now + 10,
        )

        _check_updated(storage.get_conversation("conv-1"))

        storage.store_response_link("resp_1", "conv-1", "deepseek-chat")
        _check_link(storage.get_response_link("resp_1"))

        if storage.delete_conversation("conv-1") is not True:
            msg = "expected delete to return True"
            raise AssertionError(msg)
        if storage.get_conversation("conv-1") is not None:
            msg = "expected conv-1 to be gone"
            raise AssertionError(msg)
        if storage.delete_conversation("conv-1") is not False:
            msg = "expected second delete to return False"
            raise AssertionError(msg)


async def test_conversation_manager_persistence() -> None:
    """Verify conversation state survives manager restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite"
        accounts_file = Path(tmpdir) / "accounts.txt"
        accounts_file.write_text('account 1\n{"userToken": "test_token"}')
        pool = AccountPool(accounts_file)

        storage1 = Storage(db_path)
        mgr1 = ConversationManager(pool, DummySolver(), storage=storage1)

        conv1 = await mgr1.get_or_create("session-xyz")
        conv1.account_index = 0
        conv1.account_token = _persistence_token()
        conv1.deepseek_session_id = "ds_session_999"
        conv1.parent_message_id = _PARENT_PERSISTED
        mgr1.test_hook_record_history(conv1, "Prompt 1", "Answer 1")

        await mgr1.aclose()

        storage2 = Storage(db_path)
        mgr2 = ConversationManager(pool, DummySolver(), storage=storage2)

        conv2 = await mgr2.get_or_create("session-xyz")
        _check_restored(conv2)

        mgr2.test_hook_record_history(conv2, "Prompt 2", "Answer 2")
        await mgr2.aclose()

        storage3 = Storage(db_path)
        mgr3 = ConversationManager(pool, DummySolver(), storage=storage3)
        conv3 = await mgr3.get_or_create("session-xyz")
        _check_final(conv3)

        await mgr3.aclose()

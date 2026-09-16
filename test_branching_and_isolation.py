# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Test conversation branching and isolation for new chats."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, override

from fastapi.testclient import TestClient

import app.main as main_mod
from app.accounts import AccountPool
from app.conversations import ConversationManager
from app.deepseek import CompletionOptions, DeepSeekClient
from app.pow_solver import PowSolver
from app.storage import Storage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_HTTP_OK = 200
_ONE_SESSION = 1
_TWO_SESSIONS = 2
_PARENT_TURN_TWO = 101
_PARENT_TURN_THREE = 102


class RecordedCall(TypedDict):
    """Single recorded fake completion call."""

    prompt: str
    chat_session_id: str
    parent_message_id: int | None


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


class FakeDeepSeekClient(DeepSeekClient):
    """In-memory stand-in that records calls and replays canned fragments."""

    def __init__(
        self,
        token: str,
        pow_solver: PowSolver | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize the fake client and its record buffers.

        Args:
            token: Account token this client serves.
            pow_solver: Ignored solver kept for signature compatibility.
            timeout: Ignored timeout kept for signature compatibility.

        """
        self.token = token
        self._pow_solver = pow_solver
        self._timeout = timeout
        self.created_sessions: list[str] = []
        self.recorded_calls: list[RecordedCall] = []

    @override
    async def create_session(self) -> str:
        """Create and record a fake session id.

        Returns:
            New fake session identifier.

        """
        sid = f"sess_{len(self.created_sessions) + 1}"
        self.created_sessions.append(sid)
        return sid

    @override
    async def upload_file(
        self,
        filename: str,
        content: bytes,
        mime: str | None = None,
        *,
        vision: bool = False,
    ) -> str:
        """Return a canned file id without uploading.

        Args:
            filename: File name.
            content: File bytes.
            mime: Optional mime type.
            vision: Whether vision processing was requested.

        Returns:
            Canned file identifier.

        """
        _ = (filename, content, mime, vision)
        return "file_123"

    @override
    async def stream_completion(
        self,
        *,
        prompt: str,
        chat_session_id: str,
        **options: Unpack[CompletionOptions],
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay canned fragments while recording the call.

        Yields:
            Canned completion events.

        """
        parent_message_id = options.get("parent_message_id")
        self.recorded_calls.append(
            {
                "prompt": prompt,
                "chat_session_id": chat_session_id,
                "parent_message_id": parent_message_id,
            },
        )
        rid = (parent_message_id or 100) + 1
        ans = f"Reply to {prompt[:20]}"
        yield {"event": "ready", "data": {"response_message_id": rid}}
        yield {
            "event": None,
            "data": {
                "p": "response/fragments",
                "o": "APPEND",
                "v": {"type": "RESPONSE", "content": ""},
            },
        }
        yield {"event": None, "data": {"v": ans}}

    @override
    async def aclose(self) -> None:
        """Close the fake client without action."""

    @override
    async def delete_session(self, session_id: str) -> None:
        """Delete a fake session without action.

        Args:
            session_id: Session id to delete.

        """
        _ = session_id


class TestBranchingAndIsolation(unittest.TestCase):
    """Verify new chats isolate sessions and branches parent correctly."""

    @override
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_data.sqlite"
        self.accounts_path = Path(self.tmpdir.name) / "accounts.txt"
        self.accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')

        self.storage = Storage(self.db_path)
        self.pool = AccountPool(self.accounts_path)

        self.manager = ConversationManager(
            self.pool,
            DummySolver(),
            storage=self.storage,
        )
        token = "test_" + "tok_1"
        self.fake_client = FakeDeepSeekClient(token)
        self.manager.test_hook_inject_client(token, self.fake_client)

        main_mod.hook_set_state(
            storage=self.storage,
            pool=self.pool,
            solver=DummySolver(),
            manager=self.manager,
        )
        self.client = TestClient(main_mod.app)

    @override
    def tearDown(self) -> None:
        """Clean up temporary state."""
        self.tmpdir.cleanup()

    @staticmethod
    def _check_status(code: int) -> None:
        """Check a chat response status is OK.

        Args:
            code: Observed status code.

        Raises:
            AssertionError: If the code differs.

        """
        if code != _HTTP_OK:
            msg = f"unexpected status {code}"
            raise AssertionError(msg)

    def _check_call(
        self,
        expected_session: str,
        expected_parent: int | None,
    ) -> None:
        """Check the last recorded fake call fields.

        Args:
            expected_session: Expected session id.
            expected_parent: Expected parent id.

        Raises:
            AssertionError: If fields differ.

        """
        last = self.fake_client.recorded_calls[-1]
        if last["chat_session_id"] != expected_session:
            msg = f"unexpected session {last['chat_session_id']!r}"
            raise AssertionError(msg)
        if last["parent_message_id"] != expected_parent:
            msg = f"unexpected parent {last['parent_message_id']!r}"
            raise AssertionError(msg)

    def _check_session_count(self, expected: int) -> None:
        """Check how many fake sessions were created.

        Args:
            expected: Expected session count.

        Raises:
            AssertionError: If count differs.

        """
        actual = len(self.fake_client.created_sessions)
        if actual != expected:
            msg = f"unexpected session count {actual}"
            raise AssertionError(msg)

    def test_new_chat_isolation(self) -> None:
        """Verify separate new chats use isolated sessions."""
        r1 = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "My name is Petrig."}],
            },
        )
        self._check_status(r1.status_code)
        self._check_session_count(_ONE_SESSION)
        sess1 = self.fake_client.created_sessions[0]
        self._check_call(sess1, None)

        r2 = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "what is my name again?"}],
            },
        )
        self._check_status(r2.status_code)
        self._check_session_count(_TWO_SESSIONS)
        sess2 = self.fake_client.created_sessions[1]
        self._check_call(sess2, None)

    def test_branching_conversation(self) -> None:
        """Verify branching parents to the selected turn."""
        messages = [{"role": "user", "content": "My name is Petrig."}]
        r1 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": messages},
        )
        self._check_status(r1.status_code)
        ans1 = r1.json()["choices"][0]["message"]["content"]
        sess1 = self.fake_client.created_sessions[0]
        self._check_call(sess1, None)

        messages.extend(
            [
                {"role": "assistant", "content": ans1},
                {"role": "user", "content": "what is my name again?"},
            ],
        )
        r2 = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": messages},
        )
        self._check_status(r2.status_code)
        ans2 = r2.json()["choices"][0]["message"]["content"]
        self._check_call(sess1, _PARENT_TURN_TWO)

        branch_a_msgs = list(messages)
        branch_a_msgs.extend(
            [
                {"role": "assistant", "content": ans2},
                {"role": "user", "content": "remember the string ABC123XYZ"},
            ],
        )
        r3_a = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": branch_a_msgs},
        )
        self._check_status(r3_a.status_code)
        self._check_call(sess1, _PARENT_TURN_THREE)

        branch_b_msgs = list(messages)
        branch_b_msgs.extend(
            [
                {"role": "assistant", "content": ans2},
                {"role": "user", "content": "what string did i ask you to remember?"},
            ],
        )
        r3_b = self.client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": branch_b_msgs},
        )
        self._check_status(r3_b.status_code)
        self._check_call(sess1, _PARENT_TURN_THREE)

    def test_previous_response_id_survives_restart(self) -> None:
        """Verify a chained follow-up resolves after an in-memory wipe."""
        r1 = self.client.post(
            "/v1/responses",
            json={"model": "deepseek-chat", "input": "explain thematic rrl"},
        )
        self._check_status(r1.status_code)
        first_id = r1.json()["id"]
        sess1 = self.fake_client.created_sessions[0]

        # Simulate a process restart: drop in-memory conversations and
        # response links, keeping only SQLite rows.
        self.manager.test_hook_drop_memory()
        main_mod.test_hook_clear_response_links()
        r2 = self.client.post(
            "/v1/responses",
            json={
                "model": "deepseek-chat",
                "input": "what sections to avoid",
                "previous_response_id": first_id,
            },
        )
        self._check_status(r2.status_code)
        # Same pinned DeepSeek session continues the native chain.
        self._check_session_count(_ONE_SESSION)
        self._check_call(sess1, _PARENT_TURN_TWO)


if __name__ == "__main__":
    unittest.main()

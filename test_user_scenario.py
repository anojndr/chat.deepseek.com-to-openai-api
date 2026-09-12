# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Check branching and isolation for user scenario."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, override
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_mod
from app.accounts import AccountPool
from app.citations import CitationRewriter, rewrite_citations
from app.conversations import ConversationManager
from app.deepseek import CompletionOptions, DeepSeekClient
from app.pow_solver import PowSolver
from app.storage import Storage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_FAKE_ACCOUNT_ID = "test_tok_1"
_HTTP_OK = 200
_PARENT_FIRST = 101
_PARENT_SECOND = 102
_REMEMBER_PARENT = 103
_BASE_PARENT = 100
_TEST_FILE_ID = "file_123"
_MODEL_CHAT = "deepseek-chat"
_ENDPOINT = "/v1/chat/completions"
_NAME_AGAIN_PROMPT = "what is my name again? Answer in one sentence only."
_REMEMBER_PROMPT = (
    'remember the string `*h#n3XBe8Y$SjJ92y4FX`. reply with "understood" only.'
)
_STRING_AGAIN_PROMPT = "what string did i ask you to remember again?"


class DummySolver(PowSolver):
    """Refuse to solve PoW in tests."""

    def __init__(self) -> None:
        """Initialize without wasm."""

    @override
    def solve(
        self,
        challenge_hex: str,
        salt: str,
        expire_at: str | float,
        difficulty: float,
    ) -> int | None:
        """Return no solution.

        Returns:
            None: Never solves.

        """
        return None


class FakeDeepSeekClient(DeepSeekClient):
    """Replay canned answers keyed by prompt."""

    def __init__(
        self,
        token: str,
        _pow_solver: PowSolver | None = None,
        _timeout: float = 120.0,
    ) -> None:
        """Store token and call log."""
        self.token = token
        self.created_sessions: list[str] = []
        self.recorded_calls: list[dict[str, Any]] = []

    @override
    async def create_session(self) -> str:
        """Return canned session id.

        Returns:
            str: Session id.

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
        """Return canned file id.

        Returns:
            str: File id.

        """
        return _TEST_FILE_ID

    @override
    async def stream_completion(
        self,
        *,
        prompt: str,
        chat_session_id: str,
        **options: Unpack[CompletionOptions],
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay canned answer.

        Yields:
            dict[str, Any]: Stream event.

        """
        parent_message_id = options.get("parent_message_id")
        self.recorded_calls.append(
            {
                "prompt": prompt,
                "chat_session_id": chat_session_id,
                "parent_message_id": parent_message_id,
            },
        )
        rid = (parent_message_id or _BASE_PARENT) + 1
        ans = _answer_for(prompt, parent_message_id)
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
        """Close without action."""

    @override
    async def delete_session(self, session_id: str) -> None:
        """Delete without action."""


def _answer_for(prompt: str, parent_message_id: int | None) -> str:
    """Select canned answer for prompt.

    Returns:
        str: Canned answer.

    """
    if "My name is Petrig" in prompt:
        return "Your name is Petrig."
    if "remember the string" in prompt:
        return "understood"
    return _answer_followup(prompt, parent_message_id)


def _answer_followup(prompt: str, parent_message_id: int | None) -> str:
    """Select followup answer.

    Returns:
        str: Canned answer.

    """
    if "what is my name again?" in prompt:
        if parent_message_id:
            return "Your name is Petrig."
        return "I don't know your name."
    if "what string did i ask you to remember again?" in prompt:
        if parent_message_id == _REMEMBER_PARENT:
            return "You asked me to remember: `*h#n3XBe8Y$SjJ92y4FX`"
        return "You haven't asked me to remember any string."
    return f"Reply to {prompt[:20]}"


def _check_equal(actual: object, expected: object, label: str) -> None:
    """Require values to match.

    Raises:
        AssertionError: If values differ.

    """
    if actual != expected:
        msg = f"{label}: {actual!r} != {expected!r}"
        raise AssertionError(msg)


def _check_is_none(value: object, label: str) -> None:
    """Require value to be None.

    Raises:
        AssertionError: If value is not None.

    """
    if value is not None:
        msg = f"{label}: {value!r} is not None"
        raise AssertionError(msg)


def _check_contains(haystack: str, needle: str, label: str) -> None:
    """Require substring present.

    Raises:
        AssertionError: If needle missing.

    """
    if needle not in haystack:
        msg = f"{label}: {needle!r} missing"
        raise AssertionError(msg)


def _check_absent(haystack: str, needle: str, label: str) -> None:
    """Require substring absent.

    Raises:
        AssertionError: If needle present.

    """
    if needle in haystack:
        msg = f"{label}: {needle!r} should be absent"
        raise AssertionError(msg)


class TestUserScenario(unittest.TestCase):
    """Verify branching and new chat isolation."""

    @override
    def setUp(self) -> None:
        """Create manager and test client."""
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
        self.fake_client = FakeDeepSeekClient(_FAKE_ACCOUNT_ID)
        patch.object(
            self.manager,
            "client_for",
            return_value=self.fake_client,
        ).start()
        self.addCleanup(patch.stopall)
        patch.object(main_mod, "_storage", self.storage).start()
        patch.object(main_mod, "_pool", self.pool).start()
        patch.object(main_mod, "_solver", DummySolver()).start()
        patch.object(main_mod, "_manager", self.manager).start()
        self.client = TestClient(main_mod.app)

    @override
    def tearDown(self) -> None:
        """Cleanup temp dir."""
        self.tmpdir.cleanup()

    @staticmethod
    def test_space_before_punctuation_and_citations() -> None:
        """Check citation rewrite spacing."""
        raw_text = (
            "This girl is **Hina Yumihara** from the 2014 mecha "
            "anime **Buddy Complex** .\n\n"
            "### Series Details\n\n"
            "*   **Character Role**: A transfer student who is secretly "
            "a time traveler and pilot from the future .\n"
            "*   **Studio**: Sunrise .\n"
            "*   **Plot Summary**: High school student Aoba Watase is "
            "thrust into a future war after being saved by Hina, "
            "and the series follows his journey piloting a giant robot ."
        )
        cleaned = rewrite_citations(raw_text, [])
        _check_absent(cleaned, " .", "cleaned")
        _check_contains(cleaned, "**Buddy Complex**.", "complex")
        _check_contains(cleaned, "Sunrise.", "studio")
        _check_contains(cleaned, "giant robot.", "robot")
        raw_cite = "From the anime **Buddy Complex** [!citation:1][!citation:2]."
        rewritten = rewrite_citations(
            raw_cite,
            ["https://example.com/1", "https://example.com/2"],
        )
        _check_equal(
            rewritten,
            "From the anime **Buddy Complex** "
            "[citation:1](https://example.com/1) "
            "[citation:2](https://example.com/2).",
            "citations",
        )
        rw = CitationRewriter([])
        first_chunk = (
            "This girl is **Hina Yumihara** from the 2014 mecha "
            "anime **Buddy Complex** "
        )
        chunks = [
            first_chunk,
            ".\n\n*   **Studio**: Sunrise ",
            " .\n*   **Plot Summary**: High school student",
            " .",
        ]
        streamed = "".join(rw.feed(c) for c in chunks) + rw.finish()
        _check_absent(streamed, " .", "streamed")
        _check_contains(streamed, "**Buddy Complex**.", "complex stream")
        _check_contains(streamed, "Sunrise.", "studio stream")
        _check_contains(streamed, "High school student.", "student stream")

    def test_user_branching_and_new_chat_scenario(self) -> None:
        """Check branching and new chat isolation."""
        msgs: list[dict[str, str]] = [
            {
                "role": "user",
                "content": "My name is Petrig. Answer in one sentence only.",
            },
        ]
        ans1 = self._post_and_answer(msgs)
        _check_equal(ans1, "Your name is Petrig.", "ans1")
        self._check_parent_none()
        msgs.extend(
            [
                {"role": "assistant", "content": ans1},
                {
                    "role": "user",
                    "content": _NAME_AGAIN_PROMPT,
                },
            ],
        )
        ans2 = self._post_and_answer(msgs)
        _check_equal(ans2, "Your name is Petrig.", "ans2")
        self._check_parent(_PARENT_FIRST, "turn2")
        branch_a = [
            *msgs,
            {"role": "assistant", "content": ans2},
            {
                "role": "user",
                "content": _REMEMBER_PROMPT,
            },
        ]
        ans3_a = self._post_and_answer(branch_a)
        _check_equal(ans3_a, "understood", "branch a")
        self._check_parent(_PARENT_SECOND, "branch a")
        branch_b = [
            *msgs,
            {"role": "assistant", "content": ans2},
            {
                "role": "user",
                "content": _STRING_AGAIN_PROMPT,
            },
        ]
        ans3_b = self._post_and_answer(branch_b)
        self._check_parent(_PARENT_SECOND, "branch b")
        _check_equal(
            ans3_b,
            "You haven't asked me to remember any string.",
            "branch b",
        )
        ans_new_1 = self._post_and_answer(
            [
                {
                    "role": "user",
                    "content": _NAME_AGAIN_PROMPT,
                },
            ],
        )
        self._check_parent_none()
        _check_equal(ans_new_1, "I don't know your name.", "new1")
        ans_new_2 = self._post_and_answer(
            [
                {
                    "role": "user",
                    "content": _STRING_AGAIN_PROMPT,
                },
            ],
        )
        self._check_parent_none()
        _check_equal(
            ans_new_2,
            "You haven't asked me to remember any string.",
            "new2",
        )

    def _post_and_answer(self, messages: list[dict[str, str]]) -> str:
        """Post chat and return content.

        Returns:
            str: Assistant content.

        Raises:
            AssertionError: If status is not ok.
            TypeError: If response shape is invalid.

        """
        resp = self.client.post(
            _ENDPOINT,
            json={"model": _MODEL_CHAT, "messages": messages},
        )
        _check_equal(resp.status_code, _HTTP_OK, "status")
        data = resp.json()
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            msg = "missing choices"
            raise AssertionError(msg)
        first = choices[0]
        if not isinstance(first, dict):
            msg = "bad choice shape"
            raise TypeError(msg)
        message = first.get("message")
        if not isinstance(message, dict):
            msg = "bad message shape"
            raise TypeError(msg)
        content = message.get("content")
        if not isinstance(content, str):
            msg = "bad content shape"
            raise TypeError(msg)
        return content

    def _check_parent(self, expected: int, label: str) -> None:
        """Check last parent id matches."""
        actual = self.fake_client.recorded_calls[-1]["parent_message_id"]
        _check_equal(actual, expected, label)

    def _check_parent_none(self) -> None:
        """Check last parent is None."""
        actual = self.fake_client.recorded_calls[-1]["parent_message_id"]
        _check_is_none(actual, "parent")


if __name__ == "__main__":
    unittest.main()

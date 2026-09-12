# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Check unified model routes file turns to default."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, override
from unittest.mock import patch

from app.accounts import AccountPool
from app.conversations import ConversationManager
from app.deepseek import CompletionOptions, DeepSeekClient
from app.models import parse_model
from app.pow_solver import PowSolver
from app.storage import Storage
from app.turn import PreparedTurn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_FAKE_ACCOUNT_ID = "test_tok_1"
_TEST_IMAGE_PROMPT = "what is in this image?"
_TEST_IMAGE_NAME = "tiny.png"
_TEST_IMAGE_MIME = "image/png"
_TEST_IMAGE_BYTES = b"raw"
_TEST_FILE_ID = "file_123"
_TEST_SESSION = "sess_1"
_TEST_RESPONSE_ID = 101
_TEST_CONTENT = "ok"


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
    """Record calls and replay canned fragments."""

    def __init__(self, token: str) -> None:
        """Store token and call log."""
        self.token = token
        self.recorded_calls: list[dict[str, Any]] = []

    @override
    async def create_session(self) -> str:
        """Return canned session id.

        Returns:
            str: Session id.

        """
        return _TEST_SESSION

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
        """Replay canned fragments.

        Yields:
            dict[str, Any]: Stream event.

        """
        model_type = options.get("model_type")
        ref_file_ids = options.get("ref_file_ids")
        self.recorded_calls.append(
            {"model_type": model_type, "ref_file_ids": list(ref_file_ids or [])},
        )
        yield {"event": "ready", "data": {"response_message_id": _TEST_RESPONSE_ID}}
        yield {
            "event": None,
            "data": {
                "p": "response/fragments",
                "o": "APPEND",
                "v": {"type": "RESPONSE", "content": ""},
            },
        }
        yield {"event": None, "data": {"v": _TEST_CONTENT}}

    @override
    async def aclose(self) -> None:
        """Close without action."""

    @override
    async def delete_session(self, session_id: str) -> None:
        """Delete without action."""


def _image_turn() -> PreparedTurn:
    """Build image turn fixture.

    Returns:
        PreparedTurn: Image prompt.

    """
    return PreparedTurn(
        prompt=_TEST_IMAGE_PROMPT,
        files=[(_TEST_IMAGE_NAME, _TEST_IMAGE_BYTES, _TEST_IMAGE_MIME)],
    )


def _check_is_none(value: object, label: str) -> None:
    """Require value to be None.

    Raises:
        AssertionError: If value is not None.

    """
    if value is not None:
        msg = f"{label}: {value!r} is not None"
        raise AssertionError(msg)


def _check_equal(actual: object, expected: object, label: str) -> None:
    """Require values to match.

    Raises:
        AssertionError: If values differ.

    """
    if actual != expected:
        msg = f"{label}: {actual!r} != {expected!r}"
        raise AssertionError(msg)


class TestUnifiedModel(unittest.TestCase):
    """Verify model alias routing."""

    @override
    def setUp(self) -> None:
        """Create manager with fake client."""
        self.tmpdir = tempfile.TemporaryDirectory()
        accounts_path = Path(self.tmpdir.name) / "accounts.txt"
        accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')
        self.pool = AccountPool(accounts_path)
        self.manager = ConversationManager(
            self.pool,
            DummySolver(),
            storage=Storage(Path(self.tmpdir.name) / "t.sqlite"),
        )
        self.fake_client = FakeDeepSeekClient(_FAKE_ACCOUNT_ID)
        patch.object(
            self.manager,
            "client_for",
            return_value=self.fake_client,
        ).start()
        self.addCleanup(patch.stopall)

    @override
    def tearDown(self) -> None:
        """Cleanup temp dir."""
        self.tmpdir.cleanup()

    def test_instant_aliases_route_to_default(self) -> None:
        """Check instant aliases map to default.

        Raises:
            AssertionError: If alias routes incorrectly.

        """
        for model_id in ("instant", "deepseek-instant", "deepseek-chat", "default"):
            with self.subTest(model=model_id):
                spec = parse_model(model_id)
                _check_is_none(spec.model_type, model_id)
                if spec.deepthink:
                    msg = f"{model_id}: deepthink should be off"
                    raise AssertionError(msg)

    @staticmethod
    def test_expert_vision_aliases_route_explicitly() -> None:
        """Check expert and vision aliases route explicitly."""
        _check_equal(parse_model("expert").model_type, "expert", "expert")
        _check_equal(
            parse_model("deepseek-expert").model_type,
            "expert",
            "deepseek-expert",
        )
        _check_equal(parse_model("vision").model_type, "vision", "vision")
        _check_equal(
            parse_model("deepseek-vision").model_type,
            "vision",
            "deepseek-vision",
        )

    @staticmethod
    def test_instant_deepthink_suffix_enables_thinking() -> None:
        """Check deepthink suffix enables thinking.

        Raises:
            AssertionError: If routing mismatches.

        """
        spec = parse_model("deepseek-instant-deepthink")
        _check_is_none(spec.model_type, "deepthink alias")
        if not spec.deepthink:
            msg = "deepthink should be on"
            raise AssertionError(msg)


class TestUnifiedFileTurns(unittest.IsolatedAsyncioTestCase):
    """Verify file turns keep default model."""

    @override
    def setUp(self) -> None:
        """Create manager with fake client."""
        self.tmpdir = tempfile.TemporaryDirectory()
        accounts_path = Path(self.tmpdir.name) / "accounts.txt"
        accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')
        self.pool = AccountPool(accounts_path)
        self.manager = ConversationManager(
            self.pool,
            DummySolver(),
            storage=Storage(Path(self.tmpdir.name) / "t.sqlite"),
        )
        self.fake_client = FakeDeepSeekClient(_FAKE_ACCOUNT_ID)
        patch.object(
            self.manager,
            "client_for",
            return_value=self.fake_client,
        ).start()
        self.addCleanup(patch.stopall)

    @override
    def tearDown(self) -> None:
        """Cleanup temp dir."""
        self.tmpdir.cleanup()

    async def test_file_turn_keeps_default_model(self) -> None:
        """Check file turn keeps default model."""
        result = await self.manager.run_turn(
            "k-default",
            _image_turn(),
            deepthink=False,
            model_type=None,
        )
        _check_equal(result.content, _TEST_CONTENT, "content")
        call = self.fake_client.recorded_calls[-1]
        _check_is_none(call["model_type"], "model_type")
        _check_equal(call["ref_file_ids"], [_TEST_FILE_ID], "files")

    async def test_explicit_vision_still_passes_through(self) -> None:
        """Check explicit vision passes through."""
        result = await self.manager.run_turn(
            "k-vision",
            _image_turn(),
            deepthink=False,
            model_type="vision",
        )
        _check_equal(result.content, _TEST_CONTENT, "content")
        _check_equal(
            self.fake_client.recorded_calls[-1]["model_type"],
            "vision",
            "model_type",
        )

    async def test_stream_file_turn_keeps_default_model(self) -> None:
        """Check streamed file turn keeps default model.

        Raises:
            AssertionError: If stream or model mismatches.

        """
        seen = [
            str(ev.value)
            async for ev in self.manager.stream_turn(
                "k-stream",
                _image_turn(),
                deepthink=False,
                model_type=None,
            )
            if ev.kind == "content"
        ]
        if not seen:
            msg = "expected content events"
            raise AssertionError(msg)
        call = self.fake_client.recorded_calls[-1]
        _check_is_none(call["model_type"], "model_type")
        _check_equal(call["ref_file_ids"], [_TEST_FILE_ID], "files")


if __name__ == "__main__":
    unittest.main()

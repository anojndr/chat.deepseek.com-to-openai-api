"""Unified Instant/Expert/Vision model: default entry handles images natively.

The backend now parses image uploads as model_kind VISION and answers them
from every model_type, so file turns must NOT auto-switch to
model_type="vision" (sessions also pin model_type on first use, making
per-turn switching silently stick to the first value). Explicit `vision`
still passes through for callers that ask for it.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, override

from app.accounts import AccountPool
from app.conversations import ConversationManager
from app.deepseek import DeepSeekClient
from app.models import parse_model
from app.pow_solver import PowSolver
from app.storage import Storage
from app.turn import PreparedTurn


class DummySolver(PowSolver):
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


class FakeDeepSeekClient(DeepSeekClient):
    def __init__(self, token: str) -> None:
        self.token = token
        self.recorded_calls: list[dict[str, Any]] = []

    @override
    async def create_session(self) -> str:
        return "sess_1"

    @override
    async def upload_file(
        self,
        filename: str,
        content: bytes,
        mime: str | None = None,
        *,
        vision: bool = False,
    ) -> str:
        return "file_123"

    @override
    async def stream_completion(
        self,
        *,
        prompt: str,
        chat_session_id: str,
        parent_message_id: int | None = None,
        ref_file_ids: list[str] | None = None,
        thinking_enabled: bool = False,
        search_enabled: bool = True,
        model_type: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.recorded_calls.append(
            {"model_type": model_type, "ref_file_ids": list(ref_file_ids or [])}
        )
        yield {"event": "ready", "data": {"response_message_id": 101}}
        yield {
            "event": None,
            "data": {
                "p": "response/fragments",
                "o": "APPEND",
                "v": {"type": "RESPONSE", "content": ""},
            },
        }
        yield {"event": None, "data": {"v": "ok"}}

    @override
    async def aclose(self) -> None:
        pass

    @override
    async def delete_session(self, session_id: str) -> None:
        pass


def _image_turn() -> PreparedTurn:
    return PreparedTurn(
        prompt="what is in this image?", files=[("tiny.png", b"raw", "image/png")]
    )


class TestUnifiedModel(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        accounts_path = Path(self.tmpdir.name) / "accounts.txt"
        accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')
        self.pool = AccountPool(accounts_path)
        self.manager = ConversationManager(
            self.pool,
            DummySolver(),
            storage=Storage(Path(self.tmpdir.name) / "t.sqlite"),
        )
        self.fake_client = FakeDeepSeekClient("test_tok_1")
        self.manager._clients["test_tok_1"] = self.fake_client

    @override
    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_instant_aliases_route_to_default(self) -> None:
        for model_id in ("instant", "deepseek-instant", "deepseek-chat", "default"):
            with self.subTest(model=model_id):
                spec = parse_model(model_id)
                self.assertIsNone(spec.model_type)
                self.assertFalse(spec.deepthink)

    def test_expert_vision_aliases_route_explicitly(self) -> None:
        self.assertEqual(parse_model("expert").model_type, "expert")
        self.assertEqual(parse_model("deepseek-expert").model_type, "expert")
        self.assertEqual(parse_model("vision").model_type, "vision")
        self.assertEqual(parse_model("deepseek-vision").model_type, "vision")

    def test_instant_deepthink_suffix_enables_thinking(self) -> None:
        spec = parse_model("deepseek-instant-deepthink")
        self.assertIsNone(spec.model_type)
        self.assertTrue(spec.deepthink)


class TestUnifiedFileTurns(unittest.IsolatedAsyncioTestCase):
    @override
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        accounts_path = Path(self.tmpdir.name) / "accounts.txt"
        accounts_path.write_text('account 1\n{"userToken": "test_tok_1"}')
        self.pool = AccountPool(accounts_path)
        self.manager = ConversationManager(
            self.pool,
            DummySolver(),
            storage=Storage(Path(self.tmpdir.name) / "t.sqlite"),
        )
        self.fake_client = FakeDeepSeekClient("test_tok_1")
        self.manager._clients["test_tok_1"] = self.fake_client

    @override
    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    async def test_file_turn_keeps_default_model(self) -> None:
        result = await self.manager.run_turn(
            "k-default", _image_turn(), deepthink=False, model_type=None
        )
        self.assertEqual(result.content, "ok")
        call = self.fake_client.recorded_calls[-1]
        self.assertIsNone(call["model_type"])
        self.assertEqual(call["ref_file_ids"], ["file_123"])

    async def test_explicit_vision_still_passes_through(self) -> None:
        result = await self.manager.run_turn(
            "k-vision", _image_turn(), deepthink=False, model_type="vision"
        )
        self.assertEqual(result.content, "ok")
        self.assertEqual(self.fake_client.recorded_calls[-1]["model_type"], "vision")

    async def test_stream_file_turn_keeps_default_model(self) -> None:
        seen: list[str] = []
        async for ev in self.manager.stream_turn(
            "k-stream", _image_turn(), deepthink=False, model_type=None
        ):
            if ev.kind == "content":
                seen.append(str(ev.value))
        self.assertTrue(seen)
        call = self.fake_client.recorded_calls[-1]
        self.assertIsNone(call["model_type"])
        self.assertEqual(call["ref_file_ids"], ["file_123"])


if __name__ == "__main__":
    unittest.main()

"""Regression tests: stale parent_message_id / session recovery after a dead stream.

Root cause being guarded: _stream_events/_collect used to record the
response_message_id only at natural stream end (streaming variant), so a
stream that died or a cancelled request left parent_message_id pointing at an
ancestor message. The NEXT turn then appended at the wrong node of the
DeepSeek session tree and upstream answered the previous topic instead of the
new prompt.
"""

import asyncio
import tempfile
from pathlib import Path

from app.storage import Storage
from app.accounts import AccountPool
from app.conversations import ConversationManager, DeepSeekError
from app.turn import prepare_turn


class FakeDeepSeekClient:
    """Scriptable stand-in for DeepSeekClient."""

    def __init__(self, token: str, script: list | None = None) -> None:
        self.token = token
        self.script: list[list] = [script] if script is not None else []
        self.calls: list[dict] = []
        self.sessions: list[str] = []

    async def create_session(self) -> str:
        sid = f"s{len(self.sessions) + 1}"
        self.sessions.append(sid)
        return sid

    async def upload_file(self, *a, **k):  # pragma: no cover - unused here
        raise AssertionError("no files expected")

    async def stream_completion(self, **kwargs):
        self.calls.append(kwargs)
        for item in self.script.pop(0):
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, dict) and "__hang__" in item:
                await item["__hang__"].wait()
                continue
            yield item


def _turn(rid: int, text: str):
    """ready + RESPONSE fragment + one implicit content delta."""
    return [
        {"event": "ready", "data": {"response_message_id": rid}},
        {
            "event": None,
            "data": {"p": "response/fragments", "o": "APPEND", "v": {"type": "RESPONSE", "content": ""}},
        },
        {"event": None, "data": {"v": text}},
    ]


def _ready_ok():
    return _turn(101, "a1")


def _ready2_then_die():
    return [*_turn(202, "par"), RuntimeError("upstream connection reset")]


def _ready3_ok():
    return _turn(303, "a3")


def _make_manager(tmpdir: str):
    db_path = Path(tmpdir) / "t.sqlite"
    accounts_path = Path(tmpdir) / "accounts.txt"
    accounts_path.write_text('account 1\n{"userToken": "tok"}')

    class DummySolver:
        pass

    mgr = ConversationManager(AccountPool(accounts_path), DummySolver(), storage=Storage(db_path))
    client = FakeDeepSeekClient("tok")
    mgr.client_for = lambda token: client  # type: ignore[method-assign]
    return mgr, client


def _prepared(text: str):
    return prepare_turn([{"role": "user", "content": text}], is_first_turn=False)


async def test_midstream_failure_recovers_with_full_replay():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client = _make_manager(tmpdir)

        # Turn 1: healthy - establishes session s1 + history.
        client.script.append(_ready_ok())
        result = await mgr.run_turn("k", _prepared("q1"), deepthink=False, model_type=None)
        assert result.content == "a1"
        conv = await mgr.get_or_create("k")
        assert conv.deepseek_session_id == "s1"
        assert conv.parent_message_id == 101
        assert client.calls[-1]["chat_session_id"] == "s1"

        # Turn 2: upstream commits message 202, then the stream dies mid-answer.
        client.script.append(_ready2_then_die())
        try:
            await mgr.run_turn("k", _prepared("q2"), deepthink=False, model_type=None)
            raise AssertionError("expected DeepSeekError")
        except DeepSeekError:
            pass

        # The committed-but-unrecorded turn must NOT stay pinned: dropping the
        # session (and persisting that!) forces a clean replay next turn,
        # otherwise the next turn branches at ancestor 101/202 and upstream
        # answers the wrong thread entirely.
        stored = storage_check(mgr, "k")
        assert stored["deepseek_session_id"] is None, stored
        assert stored["parent_message_id"] is None, stored

        # Turn 3: healthy again - must replay FULL proxy history into the new
        # session (q1/a1 present, broken q2 absent) instead of a bare prompt.
        client.script.append(_ready3_ok())
        await mgr.run_turn("k", _prepared("q3"), deepthink=False, model_type=None)
        replay_call = client.calls[-1]
        assert replay_call["chat_session_id"] == "s2"
        prompt = replay_call["prompt"]
        assert "[user] q1" in prompt and "[assistant] a1" in prompt, prompt[:500]
        assert "q2" not in prompt, "failed turn leaked into history"
        assert prompt.rstrip().endswith("q3")
        assert replay_call["parent_message_id"] is None

        conv = await mgr.get_or_create("k")
        assert conv.parent_message_id == 303
        await mgr.aclose()


def storage_check(mgr: ConversationManager, key: str) -> dict:
    data = mgr._storage.get_conversation(key)
    assert data is not None
    return data


async def test_cancelled_stream_drops_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client = _make_manager(tmpdir)

        client.script.append(_ready_ok())
        await mgr.run_turn("k2", _prepared("q1"), deepthink=False, model_type=None)
        assert (await mgr.get_or_create("k2")).deepseek_session_id == "s1"

        # Turn 2 hangs forever after the ready event (client disconnect shape).
        client.script.append([
            {"event": "ready", "data": {"response_message_id": 202}},
            {"__hang__": asyncio.Event()},
        ])

        async def consume():
            async for _ev in mgr.stream_turn(
                "k2", _prepared("q2"), deepthink=False, model_type=None
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        stored = storage_check(mgr, "k2")
        assert stored["deepseek_session_id"] is None, stored
        assert stored["parent_message_id"] is None, stored
        await mgr.aclose()


async def test_ready_persisted_before_stream_finishes():
    """The rid must land in storage the moment upstream commits the slot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, _client = _make_manager(tmpdir)

        committed = asyncio.Event()

        class SlowClient(FakeDeepSeekClient):
            async def stream_completion(self, **kwargs):
                self.calls.append(kwargs)
                yield {"event": "ready", "data": {"response_message_id": 555}}
                committed.set()
                await asyncio.sleep(0.4)
                raise RuntimeError("boom")

        slow = SlowClient("tok")
        mgr.client_for = lambda token: slow  # type: ignore[method-assign]

        async def burn():
            try:
                await mgr.run_turn("k3", _prepared("q"), deepthink=False, model_type=None)
            except Exception:
                pass

        task = asyncio.create_task(burn())
        try:
            await asyncio.wait_for(committed.wait(), timeout=2)
            await asyncio.sleep(0.05)
            # rid durable BEFORE completion finished:
            assert storage_check(mgr, "k3")["parent_message_id"] == 555
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await mgr.aclose()

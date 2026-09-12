# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Check stale session recovery after dead streams."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, override

from app.accounts import AccountPool
from app.conversations import ConversationManager, DeepSeekError
from app.deepseek import CompletionOptions, DeepSeekClient
from app.pow_solver import PowSolver
from app.storage import ConversationRow, ConvRef, Storage
from app.turn import prepare_turn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from app.turn import PreparedTurn

_RID_FIRST = 101
_RID_PARTIAL = 202
_RID_RECOVERED = 303
_RID_READY = 555
_TEXT_FIRST = "a1"
_TEXT_PARTIAL = "par"
_TEXT_RECOVERED = "a3"
_TEXT_THIRD = "a3"
_EXPECTED_CALLS = 3
_CANCEL_DELAY_S = 0.15
_COMMIT_DELAY_S = 0.05
_SLOW_DELAY_S = 0.4
_COMMIT_TIMEOUT_S = 2.0
_FAKE_ID = "tok"
_HANG_KEY = "__hang__"


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
    """Replay scripted stream fragments."""

    def __init__(
        self,
        token: str,
        script: list[dict[str, Any] | BaseException] | None = None,
        _pow_solver: PowSolver | None = None,
        _timeout: float = 120.0,
    ) -> None:
        """Store token and script."""
        self.token = token
        self.script: list[list[dict[str, Any] | BaseException]] = (
            [script] if script is not None else []
        )
        self.calls: list[dict[str, Any]] = []
        self.sessions: list[str] = []

    @override
    async def create_session(self) -> str:
        """Return canned session id.

        Returns:
            str: Session id.

        """
        sid = f"s{len(self.sessions) + 1}"
        self.sessions.append(sid)
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
        """Reject unexpected uploads.

        Raises:
            AssertionError: Always raised.

        """
        msg = "no files expected"
        raise AssertionError(msg)

    @override
    async def stream_completion(
        self,
        *,
        prompt: str,
        chat_session_id: str,
        **options: Unpack[CompletionOptions],
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay scripted fragments.

        Yields:
            dict[str, Any]: Stream event.

        Raises:
            AssertionError: If hang marker has wrong type.

        """
        parent_message_id = options.get("parent_message_id")
        ref_file_ids = options.get("ref_file_ids")
        thinking_enabled = options.get("thinking_enabled", False)
        search_enabled = options.get("search_enabled", True)
        model_type = options.get("model_type")
        self.calls.append(
            {
                "prompt": prompt,
                "chat_session_id": chat_session_id,
                "parent_message_id": parent_message_id,
                "ref_file_ids": ref_file_ids,
                "thinking_enabled": thinking_enabled,
                "search_enabled": search_enabled,
                "model_type": model_type,
            },
        )
        for item in self.script.pop(0):
            if isinstance(item, BaseException):
                raise item
            if _HANG_KEY in item:
                hang = item[_HANG_KEY]
                if not isinstance(hang, asyncio.Event):
                    msg = "hang marker must be Event"
                    raise AssertionError(msg)
                await hang.wait()
                continue
            yield item

    @override
    async def aclose(self) -> None:
        """Close without action."""

    @override
    async def delete_session(self, session_id: str) -> None:
        """Delete without action."""


def _turn(rid: int, text: str) -> list[dict[str, Any] | BaseException]:
    """Build ready plus fragment events.

    Returns:
        list[dict[str, Any] | BaseException]: Stream script.

    """
    return [
        {"event": "ready", "data": {"response_message_id": rid}},
        {
            "event": None,
            "data": {
                "p": "response/fragments",
                "o": "APPEND",
                "v": {"type": "RESPONSE", "content": ""},
            },
        },
        {"event": None, "data": {"v": text}},
    ]


def _ready_ok() -> list[dict[str, Any] | BaseException]:
    """Build healthy first turn script.

    Returns:
        list[dict[str, Any] | BaseException]: Stream script.

    """
    return _turn(_RID_FIRST, _TEXT_FIRST)


def _ready2_then_die() -> list[dict[str, Any] | BaseException]:
    """Build partial turn that dies midstream.

    Returns:
        list[dict[str, Any] | BaseException]: Stream script.

    """
    return [*_turn(_RID_PARTIAL, _TEXT_PARTIAL), RuntimeError("upstream reset")]


def _ready3_ok() -> list[dict[str, Any] | BaseException]:
    """Build healthy recovery script.

    Returns:
        list[dict[str, Any] | BaseException]: Stream script.

    """
    return _turn(_RID_RECOVERED, _TEXT_RECOVERED)


def _make_manager(
    tmpdir: str,
) -> tuple[ConversationManager, FakeDeepSeekClient, Storage, AccountPool]:
    """Build manager with fake client.

    Returns:
        tuple[ConversationManager, FakeDeepSeekClient, Storage, AccountPool]:
            Manager, client, storage, and pool.

    """
    db_path = Path(tmpdir) / "t.sqlite"
    accounts_path = Path(tmpdir) / "accounts.txt"
    accounts_path.write_text('account 1\n{"userToken": "tok"}')
    pool = AccountPool(accounts_path)
    storage = Storage(db_path)
    mgr = ConversationManager(pool, DummySolver(), storage=storage)
    client = FakeDeepSeekClient(_FAKE_ID)

    mgr.__dict__["_clients"] = {_FAKE_ID: client}
    return mgr, client, storage, pool


def _prepared(text: str) -> PreparedTurn:
    """Build followup turn fixture.

    Returns:
        PreparedTurn: Prepared turn.

    """
    return prepare_turn([{"role": "user", "content": text}], is_first_turn=False)


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


def storage_check(storage: Storage, key: str) -> ConversationRow:
    """Fetch stored conversation row.

    Returns:
        ConversationRow: Stored row.

    Raises:
        AssertionError: If row missing.

    """
    data = storage.get_conversation(key)
    if data is None:
        msg = f"missing row for {key}"
        raise AssertionError(msg)
    return data


async def test_midstream_failure_recovers_with_full_replay() -> None:
    """Check midstream failure replays full history.

    Raises:
        AssertionError: If recovery mismatches.
        TypeError: If prompt shape is invalid.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client, storage, _pool = _make_manager(tmpdir)
        client.script.append(_ready_ok())
        result = await mgr.run_turn(
            "k",
            _prepared("q1"),
            deepthink=False,
            model_type=None,
        )
        _check_equal(result.content, _TEXT_FIRST, "content")
        conv = await mgr.get_or_create("k")
        _check_equal(conv.deepseek_session_id, "s1", "session")
        _check_equal(conv.parent_message_id, _RID_FIRST, "parent")
        _check_equal(client.calls[-1]["chat_session_id"], "s1", "call session")
        client.script.append(_ready2_then_die())
        client.script.append(_ready2_then_die())
        with contextlib.suppress(DeepSeekError):
            await mgr.run_turn(
                "k",
                _prepared("q2"),
                deepthink=False,
                model_type=None,
            )
            msg = "expected DeepSeekError"
            raise AssertionError(msg)
        stored = storage_check(storage, "k")
        _check_is_none(stored["deepseek_session_id"], "session")
        _check_is_none(stored["parent_message_id"], "parent")
        client.script.append(_ready3_ok())
        await mgr.run_turn("k", _prepared("q3"), deepthink=False, model_type=None)
        replay_call = client.calls[-1]
        _check_equal(replay_call["chat_session_id"], "s3", "replay session")
        prompt = replay_call["prompt"]
        if not isinstance(prompt, str):
            msg = "prompt must be str"
            raise TypeError(msg)
        _check_contains(prompt, "[user] q1", "replay q1")
        _check_contains(prompt, "[assistant] a1", "replay a1")
        _check_absent(prompt, "q2", "replay q2")
        if not prompt.rstrip().endswith("q3"):
            msg = "replay should end with q3"
            raise AssertionError(msg)
        _check_is_none(replay_call["parent_message_id"], "replay parent")
        conv = await mgr.get_or_create("k")
        _check_equal(conv.parent_message_id, _RID_RECOVERED, "recovered parent")
        await mgr.aclose()


async def test_cancelled_stream_drops_session() -> None:
    """Check cancelled stream drops session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client, storage, _pool = _make_manager(tmpdir)
        client.script.append(_ready_ok())
        await mgr.run_turn("k2", _prepared("q1"), deepthink=False, model_type=None)
        _check_equal(
            (await mgr.get_or_create("k2")).deepseek_session_id,
            "s1",
            "session",
        )
        client.script.append(
            [
                {"event": "ready", "data": {"response_message_id": _RID_PARTIAL}},
                {_HANG_KEY: asyncio.Event()},
            ],
        )

        async def consume() -> None:
            """Consume stream without action."""
            async for _ev in mgr.stream_turn(
                "k2",
                _prepared("q2"),
                deepthink=False,
                model_type=None,
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(_CANCEL_DELAY_S)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        stored = storage_check(storage, "k2")
        _check_is_none(stored["deepseek_session_id"], "session")
        _check_is_none(stored["parent_message_id"], "parent")
        await mgr.aclose()


async def test_ready_persisted_before_stream_finishes() -> None:
    """Check ready id persists before stream ends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, _client, storage, _pool = _make_manager(tmpdir)
        committed = asyncio.Event()

        class SlowClient(FakeDeepSeekClient):
            """Hang after ready event."""

            @override
            async def stream_completion(
                self,
                *,
                prompt: str,
                chat_session_id: str,
                **options: Unpack[CompletionOptions],
            ) -> AsyncIterator[dict[str, Any]]:
                """Yield ready then hang.

                Yields:
                    dict[str, Any]: Stream event.

                Raises:
                    RuntimeError: After ready event.

                """
                parent_message_id = options.get("parent_message_id")
                self.calls.append(
                    {
                        "prompt": prompt,
                        "chat_session_id": chat_session_id,
                        "parent_message_id": parent_message_id,
                    },
                )
                yield {"event": "ready", "data": {"response_message_id": _RID_READY}}
                committed.set()
                await asyncio.sleep(_SLOW_DELAY_S)
                boom = "boom"
                raise RuntimeError(boom)

        slow = SlowClient(_FAKE_ID)

        mgr.__dict__["_clients"] = {_FAKE_ID: slow}

        async def burn() -> None:
            """Run turn that fails."""
            with contextlib.suppress(Exception):
                await mgr.run_turn(
                    "k3",
                    _prepared("q"),
                    deepthink=False,
                    model_type=None,
                )

        task = asyncio.create_task(burn())
        try:
            await asyncio.wait_for(committed.wait(), timeout=_COMMIT_TIMEOUT_S)
            await asyncio.sleep(_COMMIT_DELAY_S)
            _check_equal(
                storage_check(storage, "k3")["parent_message_id"],
                _RID_READY,
                "ready",
            )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await mgr.aclose()


async def test_single_account_retries_empty_then_replays() -> None:
    """Check empty first attempt retries with replay.

    Raises:
        AssertionError: If retry mismatches.
        TypeError: If prompt shape is invalid.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client, storage, pool = _make_manager(tmpdir)
        _check_equal(pool.size, 1, "pool size")
        client.script.append(_ready_ok())
        result = await mgr.run_turn(
            "kr",
            _prepared("q1"),
            deepthink=False,
            model_type=None,
        )
        _check_equal(result.content, _TEXT_FIRST, "content")
        client.script.append([])
        client.script.append(_turn(_RID_RECOVERED, _TEXT_THIRD))
        result2 = await mgr.run_turn(
            "kr",
            _prepared("Cost?"),
            deepthink=False,
            model_type=None,
        )
        _check_equal(result2.content, _TEXT_THIRD, "retry content")
        _check_equal(len(client.calls), _EXPECTED_CALLS, "calls")
        replay_prompt = client.calls[-1]["prompt"]
        if not isinstance(replay_prompt, str):
            msg = "prompt must be str"
            raise TypeError(msg)
        _check_contains(replay_prompt, "[user] q1", "replay q1")
        _check_contains(replay_prompt, "[assistant] a1", "replay a1")
        if not replay_prompt.rstrip().endswith("Cost?"):
            msg = "replay should end with Cost?"
            raise AssertionError(msg)
        stored = storage_check(storage, "kr")
        _check_equal(stored["deepseek_session_id"], "s2", "session")
        _check_equal(stored["parent_message_id"], _RID_RECOVERED, "parent")
        await mgr.aclose()


async def test_stream_single_account_retries_empty() -> None:
    """Check streaming empty attempt retries.

    Raises:
        AssertionError: If retry mismatches.
        TypeError: If prompt shape is invalid.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client, storage, pool = _make_manager(tmpdir)
        _check_equal(pool.size, 1, "pool size")
        client.script.append(_ready_ok())
        await mgr.run_turn("ks", _prepared("q1"), deepthink=False, model_type=None)
        client.script.append([])
        client.script.append(_turn(_RID_RECOVERED, _TEXT_THIRD))
        seen = [
            str(ev.value)
            async for ev in mgr.stream_turn(
                "ks",
                _prepared("Cost?"),
                deepthink=False,
                model_type=None,
            )
            if ev.kind == "content"
        ]
        _check_equal("".join(seen), _TEXT_THIRD, "streamed")
        _check_equal(len(client.calls), _EXPECTED_CALLS, "calls")
        replay_prompt = client.calls[-1]["prompt"]
        if not isinstance(replay_prompt, str):
            msg = "prompt must be str"
            raise TypeError(msg)
        _check_contains(replay_prompt, "[user] q1", "replay q1")
        _check_contains(replay_prompt, "[assistant] a1", "replay a1")
        if not replay_prompt.rstrip().endswith("Cost?"):
            msg = "replay should end with Cost?"
            raise AssertionError(msg)
        stored = storage_check(storage, "ks")
        _check_equal(stored["deepseek_session_id"], "s2", "session")
        _check_equal(stored["parent_message_id"], _RID_RECOVERED, "parent")
        await mgr.aclose()


async def test_failed_empty_conversation_leaves_no_row() -> None:
    """Check failed empty conversation leaves no row.

    Raises:
        AssertionError: If junk row remains.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client, storage, _pool = _make_manager(tmpdir)
        client.script.append([DeepSeekError("boom", status=502)])
        client.script.append([DeepSeekError("boom again", status=502)])
        with contextlib.suppress(DeepSeekError):
            await mgr.run_turn(
                "kjunk",
                _prepared("Cost?"),
                deepthink=False,
                model_type=None,
            )
            msg = "expected DeepSeekError"
            raise AssertionError(msg)
        if storage.get_conversation("kjunk") is not None:
            msg = "junk row should be absent"
            raise AssertionError(msg)
        _check_equal(mgr.transcript("kjunk"), [], "transcript")
        await mgr.aclose()


async def test_failure_invalidates_stale_prefix_refs() -> None:
    """Check dead session prefixes are invalidated.

    Raises:
        AssertionError: If prefixes survive.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr, client, storage, _pool = _make_manager(tmpdir)
        if storage.get_conversation("missing") is not None:
            msg = "missing key should be absent"
            raise AssertionError(msg)
        client.script.append(_ready_ok())
        await mgr.run_turn("kp", _prepared("q1"), deepthink=False, model_type=None)
        conv = await mgr.get_or_create("kp")
        dead = conv.deepseek_session_id
        if dead is None:
            msg = "session should exist"
            raise AssertionError(msg)
        storage.record_prefix_turn(
            ["hash-dead-follow"],
            ConvRef(
                conversation_key="kp",
                account_index=conv.account_index,
                account_token=conv.account_token or "",
                deepseek_session_id=dead,
                parent_message_id=conv.parent_message_id,
                turns=1,
                updated_at=time.time(),
            ),
        )
        if storage.find_prefix(["hash-dead-follow"]) is None:
            msg = "prefix should exist"
            raise AssertionError(msg)
        client.script.append([DeepSeekError("session gone", status=502)])
        client.script.append(_turn(_RID_RECOVERED, "recovered"))
        result = await mgr.run_turn(
            "kp",
            _prepared("q2"),
            deepthink=False,
            model_type=None,
        )
        _check_equal(result.content, "recovered", "content")
        if storage.find_prefix(["hash-dead-follow"]) is not None:
            msg = "dead prefix should be gone"
            raise AssertionError(msg)
        await mgr.aclose()


if __name__ == "__main__":
    unittest.main()

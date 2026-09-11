"""Upstream stall watchdog: a wedged SSE stream must fail fast, not hang.

Regression for a live incident where one turn parked 13+ minutes inside
``stream_completion`` with zero deltas and no terminal event: httpx's read
timeout only trips on total silence, and SSE keepalives reset it.
"""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from app import deepseek as deepseek_mod
from app.deepseek import DeepSeekClient, DeepSeekError


class _FakeResponse:
    """Minimal stub of the httpx streaming response surface we use."""

    def __init__(self, lines: AsyncIterator[str]) -> None:
        self.status_code = 200
        self._lines = lines
        self.closed = False

    def aiter_lines(self) -> AsyncIterator[str]:
        return self._lines

    async def aread(self) -> bytes:
        return b""

    async def aclose(self) -> None:
        self.closed = True
        aclose = getattr(self._lines, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except StopAsyncIteration:
                pass


class _FakeHttp:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def build_request(self, *_args: object, **_kwargs: object) -> object:
        return object()

    async def send(self, _request: object, **_kwargs: object) -> _FakeResponse:
        return self._response


async def _silent() -> AsyncIterator[str]:
    await asyncio.sleep(10)
    yield "data: {}\n"
    return


async def _keepalive_dribble() -> AsyncIterator[str]:
    while True:
        yield ": ping"
        yield ""
        await asyncio.sleep(0.05)


async def _blank_dribble() -> AsyncIterator[str]:
    while True:
        yield ""
        await asyncio.sleep(0.05)


async def _healthy() -> AsyncIterator[str]:
    yield 'data: {"v": {"response": {"fragments": [{"type": "RESPONSE", "content": "hi"}]}}}'
    yield ""
    return


def _client_for(response: _FakeResponse) -> DeepSeekClient:
    client = DeepSeekClient(token="test-token", pow_solver=MagicMock())
    patch.object(client, "_http", _FakeHttp(response)).start()
    patch.object(client, "_get_pow", new=AsyncMock(return_value=({}, {}))).start()
    return client


class TestStreamStall(unittest.IsolatedAsyncioTestCase):
    async def test_silent_stream_raises_stalled(self) -> None:
        resp = _FakeResponse(_silent())
        client = _client_for(resp)
        try:
            with patch.object(deepseek_mod, "STALL_TIMEOUT", 0.2):
                with self.assertRaises(DeepSeekError) as ctx:
                    async for _ in client.stream_completion(
                        prompt="hi", chat_session_id="sess"
                    ):
                        pass
        finally:
            patch.stopall()
        self.assertIn("stalled", str(ctx.exception))
        self.assertTrue(resp.closed)

    async def test_keepalive_dribble_without_events_raises_stalled(self) -> None:
        resp = _FakeResponse(_keepalive_dribble())
        client = _client_for(resp)
        try:
            with patch.object(deepseek_mod, "STALL_TIMEOUT", 0.3):
                with self.assertRaises(DeepSeekError) as ctx:
                    async for _ in client.stream_completion(
                        prompt="hi", chat_session_id="sess"
                    ):
                        pass
        finally:
            patch.stopall()
        self.assertIn("stalled", str(ctx.exception))
        self.assertTrue(resp.closed)

    async def test_blank_dribble_without_events_raises_stalled(self) -> None:
        resp = _FakeResponse(_blank_dribble())
        client = _client_for(resp)
        try:
            with patch.object(deepseek_mod, "STALL_TIMEOUT", 0.3):
                with self.assertRaises(DeepSeekError) as ctx:
                    async for _ in client.stream_completion(
                        prompt="hi", chat_session_id="sess"
                    ):
                        pass
        finally:
            patch.stopall()
        self.assertIn("stalled", str(ctx.exception))
        self.assertTrue(resp.closed)

    async def test_healthy_stream_unaffected(self) -> None:
        resp = _FakeResponse(_healthy())
        client = _client_for(resp)
        try:
            with patch.object(deepseek_mod, "STALL_TIMEOUT", 5.0):
                events = [
                    ev
                    async for ev in client.stream_completion(
                        prompt="hi", chat_session_id="sess"
                    )
                ]
        finally:
            patch.stopall()
        self.assertTrue(events)
        self.assertTrue(resp.closed)


if __name__ == "__main__":
    unittest.main()

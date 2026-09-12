# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Check stalled streams fail fast instead of hanging."""

from __future__ import annotations

import asyncio
import contextlib
import unittest
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from app import deepseek as deepseek_mod
from app.deepseek import DeepSeekClient, DeepSeekError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Watchdog timeouts used by the stall tests.
_STALL_SHORT_S = 0.2
_STALL_MEDIUM_S = 0.3
_STALL_LONG_S = 5.0
# Delays used by fake SSE generators.
_SILENT_DELAY_S = 10.0
_DRIBBLE_DELAY_S = 0.05
# Substring expected in stall errors.
_STALLED_MARKER = "stalled"
# Healthy payload content marker.
_HEALTHY_CONTENT = "hi"


class _FakeResponse:
    """Mimic httpx streaming response surface."""

    def __init__(self, lines: AsyncIterator[str]) -> None:
        """Store lines iterator."""
        self.status_code = 200
        self._lines = lines
        self.closed = False

    def aiter_lines(self) -> AsyncIterator[str]:
        """Return lines iterator.

        Returns:
            AsyncIterator[str]: Line stream.

        """
        return self._lines

    @staticmethod
    async def aread() -> bytes:
        """Return empty body.

        Returns:
            bytes: Empty payload.

        """
        return b""

    async def aclose(self) -> None:
        """Mark response closed."""
        self.closed = True
        aclose = getattr(self._lines, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(StopAsyncIteration):
                await aclose()


class _FakeHttp:
    """Mimic httpx client send path."""

    def __init__(self, response: _FakeResponse) -> None:
        """Store fake response."""
        self._response = response

    @staticmethod
    def build_request(*_args: object, **_kwargs: object) -> object:
        """Build dummy request.

        Returns:
            object: Dummy request object.

        """
        return object()

    async def send(self, _request: object, **_kwargs: object) -> _FakeResponse:
        """Return fake response.

        Returns:
            _FakeResponse: Stored response.

        """
        return self._response


async def _silent() -> AsyncIterator[str]:
    """Yield one payload after a long silence.

    Yields:
        str: SSE data line.

    """
    await asyncio.sleep(_SILENT_DELAY_S)
    yield "data: {}\n"


async def _keepalive_dribble() -> AsyncIterator[str]:
    """Yield keepalive lines forever.

    Yields:
        str: Keepalive or blank line.

    """
    while True:
        yield ": ping"
        yield ""
        await asyncio.sleep(_DRIBBLE_DELAY_S)


async def _blank_dribble() -> AsyncIterator[str]:
    """Yield blank lines forever.

    Yields:
        str: Blank line.

    """
    while True:
        yield ""
        await asyncio.sleep(_DRIBBLE_DELAY_S)


async def _healthy() -> AsyncIterator[str]:
    """Yield one healthy RESPONSE fragment.

    Yields:
        str: SSE data line.

    """
    await asyncio.sleep(0)
    yield (
        'data: {"v": {"response": {"fragments": '
        '[{"type": "RESPONSE", "content": "hi"}]}}}'
    )
    yield ""


def _client_for(response: _FakeResponse) -> DeepSeekClient:
    """Build client stubbed onto fake transport.

    Returns:
        DeepSeekClient: Stubbed client.

    """
    client = DeepSeekClient("test-token", MagicMock())
    patch.object(client, "_http", _FakeHttp(response)).start()
    patch.object(client, "_get_pow", new=AsyncMock(return_value=({}, {}))).start()
    return client


async def _expect_stalled(
    client: DeepSeekClient,
    resp: _FakeResponse,
    stall_s: float,
) -> None:
    """Run stream and require a stall failure.

    Raises:
        AssertionError: If stream does not stall or response stays open.

    """
    try:
        with patch.object(deepseek_mod, "STALL_TIMEOUT", stall_s):
            async for _ in client.stream_completion(
                prompt="hi",
                chat_session_id="sess",
            ):
                pass
    except DeepSeekError as exc:
        text = str(exc)
        if _STALLED_MARKER not in text:
            msg = f"missing marker in {text!r}"
            raise AssertionError(msg) from exc
        if not resp.closed:
            msg = "response not closed after stall"
            raise AssertionError(msg) from None
        return
    msg = "expected DeepSeekError for stalled stream"
    raise AssertionError(msg)


class TestStreamStall(unittest.IsolatedAsyncioTestCase):
    """Verify stall watchdog trips on wedged streams."""

    @staticmethod
    async def test_silent_stream_raises_stalled() -> None:
        """Check silent stream raises stalled."""
        resp = _FakeResponse(_silent())
        client = _client_for(resp)
        try:
            await _expect_stalled(client, resp, _STALL_SHORT_S)
        finally:
            patch.stopall()

    @staticmethod
    async def test_keepalive_dribble_without_events_raises_stalled() -> None:
        """Check keepalive dribble raises stalled."""
        resp = _FakeResponse(_keepalive_dribble())
        client = _client_for(resp)
        try:
            await _expect_stalled(client, resp, _STALL_MEDIUM_S)
        finally:
            patch.stopall()

    @staticmethod
    async def test_blank_dribble_without_events_raises_stalled() -> None:
        """Check blank dribble raises stalled."""
        resp = _FakeResponse(_blank_dribble())
        client = _client_for(resp)
        try:
            await _expect_stalled(client, resp, _STALL_MEDIUM_S)
        finally:
            patch.stopall()

    @staticmethod
    async def test_healthy_stream_unaffected() -> None:
        """Check healthy stream passes through.

        Raises:
            AssertionError: If stream yields nothing or stays open.

        """
        resp = _FakeResponse(_healthy())
        client = _client_for(resp)
        try:
            with patch.object(deepseek_mod, "STALL_TIMEOUT", _STALL_LONG_S):
                events = [
                    ev
                    async for ev in client.stream_completion(
                        prompt="hi",
                        chat_session_id="sess",
                    )
                ]
        finally:
            patch.stopall()
        if not events:
            msg = "expected events from healthy stream"
            raise AssertionError(msg)
        if not resp.closed:
            msg = "response not closed after healthy stream"
            raise AssertionError(msg)


if __name__ == "__main__":
    unittest.main()

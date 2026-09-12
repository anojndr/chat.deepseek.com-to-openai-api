# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Tests for the optional Show Sources bridge (include_sources)."""

from __future__ import annotations

import asyncio
import json
import re
import unittest
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import main
from app.citations import source_appendix
from app.conversations import Conversation, ConversationManager, StreamEvent, TurnResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator

    from starlette.types import Message, Receive, Scope

SOURCES: list[dict[str, object]] = [
    {"url": "https://example.com/news", "title": "Example News"},
    {"url": "https://x.com/agency/status/1", "title": "Agency post"},
]

_CAP_ENTRIES = 50
_OVER_CAP = 60
_TEST_PORT = 50000


def _as_dict(value: object) -> dict[str, object]:
    """Narrow an unknown JSON value to a string-keyed mapping.

    Args:
        value: Value to narrow.

    Returns:
        Mapping with string keys.

    Raises:
        TypeError: If value is not a mapping with string keys.

    """
    if not isinstance(value, dict):
        msg = f"unexpected type {type(value).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            msg = f"unexpected key type {type(key).__name__}"
            raise TypeError(msg)
        result[key] = item
    return result


def _as_str(value: object) -> str:
    """Narrow an unknown value to text.

    Args:
        value: Value to narrow.

    Returns:
        String value.

    Raises:
        TypeError: If value is not a string.

    """
    if not isinstance(value, str):
        msg = f"unexpected type {type(value).__name__}"
        raise TypeError(msg)
    return value


def _make_request(body: dict[str, object]) -> Request:
    """Build a real Starlette Request carrying the given JSON body.

    Args:
        body: JSON body to encode.

    Returns:
        Starlette request with the body.

    """
    body_bytes = json.dumps(body).encode()

    async def receive() -> Message:
        """Return the encoded body as a single chunk.

        Returns:
            Single HTTP request message.

        """
        await asyncio.sleep(0)
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "client": ("testclient", _TEST_PORT),
    }
    receive_chan: Receive = receive
    return Request(scope, receive_chan)


APPENDIX = (
    "\n\nSources\n"
    "1. [Example News](https://example.com/news) (example.com) "
    "via `latest news philippines`\n"
    "2. [Agency post](https://x.com/agency/status/1) (x.com) "
    "via `latest news philippines`\n"
    "\nSearch Queries\n"
    "1. `latest news philippines`"
)


def _check_equal(actual: object, expected: object, label: str) -> None:
    """Check two values are equal.

    Args:
        actual: Observed value.
        expected: Expected value.
        label: Label for messages.

    Raises:
        AssertionError: If values differ.

    """
    if actual != expected:
        msg = f"{label}: {actual!r} != {expected!r}"
        raise AssertionError(msg)


def _check_in(needle: str, haystack: str, label: str) -> None:
    """Check a substring is present.

    Args:
        needle: Substring to find.
        haystack: Text to search.
        label: Label for messages.

    Raises:
        AssertionError: If missing.

    """
    if needle not in haystack:
        msg = f"{label}: {needle!r} not in output"
        raise AssertionError(msg)


def _check_not_in(needle: str, haystack: str, label: str) -> None:
    """Check a substring is absent.

    Args:
        needle: Substring to avoid.
        haystack: Text to search.
        label: Label for messages.

    Raises:
        AssertionError: If present.

    """
    if needle in haystack:
        msg = f"{label}: {needle!r} should be absent"
        raise AssertionError(msg)


def _decode_chunk_text(chunk: str | bytes | memoryview[int]) -> str:
    """Decode a stream chunk to text.

    Args:
        chunk: Raw chunk value.

    Returns:
        Decoded text.

    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, memoryview):
        return chunk.tobytes().decode()
    return chunk.decode()


def _collect_chat_contents(
    chunks: list[str | bytes | memoryview[int]],
) -> list[str]:
    """Collect chat delta contents from SSE chunks.

    Args:
        chunks: Raw SSE chunks.

    Returns:
        List of content deltas.

    """
    contents: list[str] = []
    for chunk in chunks:
        line = _decode_chunk_text(chunk).strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        raw: object = json.loads(line[len("data: ") :])
        payload = _as_dict(raw)
        if payload.get("object") != "chat.completion.chunk":
            continue
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_val: object = choices[0]
            if not isinstance(first_val, dict):
                continue
            first = _as_dict(first_val)
            delta_val = first.get("delta")
            if not isinstance(delta_val, dict):
                continue
            delta = _as_dict(delta_val)
            content = delta.get("content")
            if isinstance(content, str):
                contents.append(content)
    return contents


def _collect_sse_events(
    chunks: list[str | bytes | memoryview[int]],
) -> dict[str, list[dict[str, object]]]:
    """Collect SSE payloads grouped by type.

    Args:
        chunks: Raw SSE chunks.

    Returns:
        Mapping of event type to payloads.

    """
    events: dict[str, list[dict[str, object]]] = {}
    for chunk in chunks:
        for raw_block in _decode_chunk_text(chunk).strip().split("\n\n"):
            block = raw_block.strip()
            if not block:
                continue
            lines = block.splitlines()
            data_line = next(
                (line for line in lines if line.startswith("data: ")),
                None,
            )
            if data_line:
                raw: object = json.loads(data_line[len("data: ") :])
                payload = _as_dict(raw)
                kind = payload.get("type")
                if isinstance(kind, str):
                    events.setdefault(kind, []).append(payload)
    return events


def _completed_text(completed: dict[str, object]) -> object:
    """Extract answer text from a completed response payload.

    Args:
        completed: Completed event payload.

    Returns:
        Answer text value.

    Raises:
        TypeError: If payload shape differs.

    """
    response_val: object = completed["response"]
    if not isinstance(response_val, dict):
        msg = f"unexpected response type {type(response_val).__name__}"
        raise TypeError(msg)
    response = _as_dict(response_val)
    output: object = response["output"]
    if not isinstance(output, list):
        msg = f"unexpected output type {type(output).__name__}"
        raise TypeError(msg)
    first_val: object = output[0]
    if not isinstance(first_val, dict):
        msg = f"unexpected item type {type(first_val).__name__}"
        raise TypeError(msg)
    first = _as_dict(first_val)
    content: object = first["content"]
    if not isinstance(content, list):
        msg = f"unexpected content type {type(content).__name__}"
        raise TypeError(msg)
    entry_val: object = content[0]
    if not isinstance(entry_val, dict):
        msg = f"unexpected entry type {type(entry_val).__name__}"
        raise TypeError(msg)
    entry = _as_dict(entry_val)
    return entry["text"]


class SourceAppendixFormattingTest(unittest.TestCase):
    """Verify the sources appendix formatting contract."""

    @staticmethod
    def test_full_form() -> None:
        """Check the full appendix matches the pinned contract."""
        _check_equal(
            source_appendix(SOURCES, "latest news philippines"),
            APPENDIX,
            "appendix",
        )

    @staticmethod
    def test_empty_sources_yields_empty() -> None:
        """Check empty sources yield an empty string."""
        _check_equal(source_appendix([], "q"), "", "empty")

    @staticmethod
    def test_no_query_omits_search_queries_and_via() -> None:
        """Check empty query omits via markers and query list."""
        out = source_appendix(SOURCES, "")
        _check_not_in("via `", out, "via")
        _check_not_in("Search Queries", out, "queries")

    @staticmethod
    def test_title_falls_back_to_url_without_host_suffix() -> None:
        """Check missing titles fall back to the URL."""
        out = source_appendix([{"url": "https://x.io/a", "title": ""}], "q")
        _check_in("1. [https://x.io/a](https://x.io/a)", out, "fallback")
        _check_not_in(") (", out, "host suffix")

    @staticmethod
    def test_url_parens_and_spaces_escaped() -> None:
        """Check URLs escape spaces and parentheses."""
        out = source_appendix([{"url": "https://x.io/a b)c", "title": "T"}], "q")
        _check_in("https://x.io/a%20b%29c", out, "escaped url")

    @staticmethod
    def test_query_backticks_sanitized() -> None:
        """Check backticks in queries are sanitized."""
        out = source_appendix(SOURCES[:1], "what's `up`")
        _check_in("via `what's 'up'`", out, "sanitized")

    @staticmethod
    def test_queries_as_list_with_non_string_or_empty() -> None:
        """Check list queries handle non-string entries."""
        out = source_appendix(SOURCES[:1], [{"nested": "query"}])
        _check_in("via `{'nested': 'query'}`", out, "nested")

        out_none = source_appendix(SOURCES[:1], [None])
        _check_not_in("Search Queries", out_none, "none query")

    @staticmethod
    def test_multi_line_query_collapses_without_breaking_spans() -> None:
        """Check multiline queries collapse to a single line."""
        out = source_appendix(SOURCES[:1], "latest news\nphilippines\t(2026)")
        expected = (
            "\n\nSources\n"
            "1. [Example News](https://example.com/news) (example.com) "
            "via `latest news philippines (2026)`\n"
            "\nSearch Queries\n"
            "1. `latest news philippines (2026)`"
        )
        _check_equal(out, expected, "collapsed")

    @staticmethod
    def test_title_newlines_collapsed() -> None:
        """Check newlines in titles collapse to spaces."""
        out = source_appendix(
            [{"url": "https://x.io/a", "title": "line1\nline2"}],
            "q",
        )
        _check_in("[line1 line2](https://x.io/a)", out, "title")

    @staticmethod
    def test_caps_at_50_entries() -> None:
        """Check the appendix caps at fifty entries."""
        many = [
            {"url": f"https://x.io/{i}", "title": f"t{i}"} for i in range(_OVER_CAP)
        ]
        out = source_appendix(many, "q")
        entries = [line for line in out.splitlines() if re.match(r"^\d+\. \[", line)]
        _check_equal(len(entries), _CAP_ENTRIES, "entry count")


class IncludeSourcesFlagTest(unittest.TestCase):
    """Verify include_sources flag parsing uses config defaults."""

    @staticmethod
    def test_flag_none_uses_config_default() -> None:
        """Check None falls back to the configured default.

        Raises:
            AssertionError: If flag parsing fails.

        """
        with patch("app.main.INCLUDE_SOURCES", new=True):
            if not main.should_include_sources(flag=None):
                msg = "expected True when config is True"
                raise AssertionError(msg)
        with patch("app.main.INCLUDE_SOURCES", new=False):
            if main.should_include_sources(flag=None):
                msg = "expected False when config is False"
                raise AssertionError(msg)

    @staticmethod
    def test_flag_overrides_config_default() -> None:
        """Check explicit flags override the configured default.

        Raises:
            AssertionError: If override fails.

        """
        with patch("app.main.INCLUDE_SOURCES", new=True):
            if main.should_include_sources(flag=False):
                msg = "expected False to override True config"
                raise AssertionError(msg)
        with patch("app.main.INCLUDE_SOURCES", new=False):
            if not main.should_include_sources(flag=True):
                msg = "expected True to override False config"
                raise AssertionError(msg)

    @staticmethod
    def test_string_flags_parse_like_env_values() -> None:
        """Check string flags parse like environment values.

        Raises:
            AssertionError: If parsing fails.

        """
        for truthy in ("1", "true", "TRUE", "yes", "on", " on "):
            if not main.should_include_sources(flag=truthy):
                msg = f"expected truthy for {truthy!r}"
                raise AssertionError(msg)
        for falsy in ("0", "false", "no", "off", "", "garbage"):
            if main.should_include_sources(flag=falsy):
                msg = f"expected falsy for {falsy!r}"
                raise AssertionError(msg)


class ChatCompletionsAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    """Verify chat completions append sources when requested."""

    @staticmethod
    async def test_chat_non_stream_appends_sources_when_requested() -> None:
        """Check non-streaming chat appends sources when requested.

        Raises:
            TypeError: If response shape differs.

        """
        body: dict[str, object] = {
            "model": "deepseek-chat",
            "include_sources": True,
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }
        fake_result = TurnResult(
            content="Here is the news.",
            reasoning=None,
            title=None,
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_manager = AsyncMock()
        fake_manager.run_turn.return_value = fake_result

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.chat_completions(_make_request(body))
        if not isinstance(resp, JSONResponse):
            msg = f"unexpected type {type(resp).__name__}"
            raise TypeError(msg)
        data = json.loads(bytes(resp.body).decode())
        _check_equal(
            data["choices"][0]["message"]["content"],
            "Here is the news." + APPENDIX,
            "content",
        )

    @staticmethod
    async def test_chat_non_stream_omits_sources_by_default() -> None:
        """Check non-streaming chat omits sources by default.

        Raises:
            TypeError: If response shape differs.

        """
        body: dict[str, object] = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }
        fake_result = TurnResult(
            content="Here is the news.",
            reasoning=None,
            title=None,
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_manager = AsyncMock()
        fake_manager.run_turn.return_value = fake_result

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.chat_completions(_make_request(body))
        if not isinstance(resp, JSONResponse):
            msg = f"unexpected type {type(resp).__name__}"
            raise TypeError(msg)
        data = json.loads(bytes(resp.body).decode())
        _check_equal(
            data["choices"][0]["message"]["content"],
            "Here is the news.",
            "content",
        )

    @staticmethod
    async def test_chat_stream_appends_sources_chunk() -> None:
        """Check streaming chat appends sources in chunks.

        Raises:
            TypeError: If stream shape differs.

        """
        body: dict[str, object] = {
            "model": "deepseek-chat",
            "stream": True,
            "include_sources": True,
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }

        async def fake_stream_turn(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[StreamEvent]:
            """Replay canned stream events.

            Yields:
                Canned stream events.

            """
            await asyncio.sleep(0)
            yield StreamEvent("content", "Here is the news.")
            search_value: list[object] = ["latest news philippines"]
            yield StreamEvent("search", search_value)
            sources_value: list[object] = []
            sources_value.extend(SOURCES)
            yield StreamEvent("sources", sources_value)

        fake_manager = AsyncMock()
        fake_manager.stream_turn = fake_stream_turn

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.chat_completions(_make_request(body))
            if not isinstance(resp, StreamingResponse):
                msg = f"unexpected type {type(resp).__name__}"
                raise TypeError(msg)
            chunks = [c async for c in resp.body_iterator]

        contents = _collect_chat_contents(chunks)
        _check_equal("".join(contents), "Here is the news." + APPENDIX, "stream")


class ResponsesApiAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    """Verify the Responses API appends sources when requested."""

    @staticmethod
    async def test_responses_non_stream_appends_sources_when_requested() -> None:
        """Check non-streaming responses append sources when requested.

        Raises:
            TypeError: If response shape differs.

        """
        body: dict[str, object] = {
            "model": "deepseek-chat",
            "include_sources": True,
            "input": "latest news philippines",
        }
        fake_result = TurnResult(
            content="Here is the news.",
            reasoning=None,
            title=None,
            sources=SOURCES,
            search_queries=["latest news philippines"],
        )
        fake_manager = AsyncMock()
        fake_manager.run_turn.return_value = fake_result

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.responses_api(_make_request(body))
        if not isinstance(resp, JSONResponse):
            msg = f"unexpected type {type(resp).__name__}"
            raise TypeError(msg)
        data = json.loads(bytes(resp.body).decode())
        _check_equal(
            data["output"][0]["content"][0]["text"],
            "Here is the news." + APPENDIX,
            "text",
        )

    @staticmethod
    async def test_responses_stream_appends_sources_in_events() -> None:
        """Check streaming responses append sources in events.

        Raises:
            TypeError: If stream shape differs.

        """
        body: dict[str, object] = {
            "model": "deepseek-chat",
            "stream": True,
            "include_sources": True,
            "input": "latest news philippines",
        }

        async def fake_stream_turn(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[StreamEvent]:
            """Replay canned stream events.

            Yields:
                Canned stream events.

            """
            await asyncio.sleep(0)
            yield StreamEvent("content", "Here is the news.")
            search_value: list[object] = ["latest news philippines"]
            yield StreamEvent("search", search_value)
            sources_value: list[object] = []
            sources_value.extend(SOURCES)
            yield StreamEvent("sources", sources_value)

        fake_manager = AsyncMock()
        fake_manager.stream_turn = fake_stream_turn

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.responses_api(_make_request(body))
            if not isinstance(resp, StreamingResponse):
                msg = f"unexpected type {type(resp).__name__}"
                raise TypeError(msg)
            chunks = [c async for c in resp.body_iterator]

        events = _collect_sse_events(chunks)
        completed = events["response.completed"][0]
        _check_equal(
            _completed_text(completed),
            "Here is the news." + APPENDIX,
            "text",
        )


class StreamEventsSourceEmissionTest(unittest.IsolatedAsyncioTestCase):
    """Verify sources emit only when new results arrive."""

    @staticmethod
    async def test_sources_only_emitted_when_new_results_arrive() -> None:
        """Check sources emit once per new result batch."""
        cm = ConversationManager(pool=AsyncMock(), pow_solver=AsyncMock())
        conv = Conversation(id="test-id", history=[], parent_message_id=None)
        client = AsyncMock()

        async def fake_stream_completion(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterable[dict[str, object]]:
            """Replay canned completion events.

            Yields:
                Canned completion payloads.

            """
            await asyncio.sleep(0)
            yield {
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [
                        {
                            "type": "SEARCH",
                            "results": [
                                {"url": "https://example.com/1", "title": "1"},
                            ],
                        },
                    ],
                },
            }
            yield {
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "RESPONSE", "content": "Hello"}],
                },
            }
            yield {
                "event": "message",
                "data": {"v": " world"},
            }
            yield {
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [
                        {
                            "type": "SEARCH",
                            "results": [
                                {"url": "https://example.com/2", "title": "2"},
                            ],
                        },
                    ],
                },
            }

        client.stream_completion = fake_stream_completion
        events = [
            ev
            async for ev in cm.test_hook_stream_events(
                client,
                prompt="prompt",
                conv=conv,
                ref_file_ids=[],
                thinking_enabled=False,
                model_type=None,
            )
        ]
        source_events = [ev for ev in events if ev.kind == "sources"]
        _check_equal(len(source_events), 2, "source count")


if __name__ == "__main__":
    unittest.main()

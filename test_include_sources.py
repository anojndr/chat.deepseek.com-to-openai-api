"""Tests for the optional "Show Sources" bridge (include_sources).

Pins the llmcord-go-compatible appendix contract: enabled only via
`include_sources: true` (or DEEPSEEK_INCLUDE_SOURCES=1 / INCLUDE_SOURCES=1),
appended after the answer content, with the stored session history kept clean.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.citations import source_appendix
from app.conversations import Conversation, ConversationManager, StreamEvent, TurnResult
import app.main as main

SOURCES: list[dict[str, str]] = [
    {"url": "https://example.com/news", "title": "Example News"},
    {"url": "https://x.com/agency/status/1", "title": "Agency post"},
]


def _make_request(body: dict[str, Any]) -> Request:
    """Build a real Starlette Request carrying the given JSON body."""
    body_bytes = json.dumps(body).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "client": ("testclient", 50000),
    }
    return Request(scope, receive)


APPENDIX = (
    "\n\nSources\n"
    "1. [Example News](https://example.com/news) (example.com) "
    "via `latest news philippines`\n"
    "2. [Agency post](https://x.com/agency/status/1) (x.com) "
    "via `latest news philippines`\n"
    "\nSearch Queries\n"
    "1. `latest news philippines`"
)


class SourceAppendixFormattingTest(unittest.TestCase):
    def test_full_form(self):
        self.assertEqual(source_appendix(SOURCES, "latest news philippines"), APPENDIX)

    def test_empty_sources_yields_empty(self):
        self.assertEqual(source_appendix([], "q"), "")

    def test_no_query_omits_search_queries_and_via(self):
        out = source_appendix(SOURCES, "")
        self.assertNotIn("via `", out)
        self.assertNotIn("Search Queries", out)

    def test_title_falls_back_to_url_without_host_suffix(self):
        out = source_appendix([{"url": "https://x.io/a", "title": ""}], "q")
        self.assertIn("1. [https://x.io/a](https://x.io/a)", out)
        self.assertNotIn(") (", out)

    def test_url_parens_and_spaces_escaped(self):
        out = source_appendix([{"url": "https://x.io/a b)c", "title": "T"}], "q")
        self.assertIn("https://x.io/a%20b%29c", out)

    def test_query_backticks_sanitized(self):
        out = source_appendix(SOURCES[:1], "what's `up`")
        self.assertIn("via `what's 'up'`", out)

    def test_queries_as_list_with_non_string_or_empty(self):
        out = source_appendix(SOURCES[:1], [{"nested": "query"}])
        self.assertIn("via `{'nested': 'query'}`", out)

        out_none = source_appendix(SOURCES[:1], [None])
        self.assertNotIn("Search Queries", out_none)

    def test_multi_line_query_collapses_without_breaking_spans(self):
        out = source_appendix(SOURCES[:1], "latest news\nphilippines\t(2026)")
        expected = (
            "\n\nSources\n"
            "1. [Example News](https://example.com/news) (example.com) "
            "via `latest news philippines (2026)`\n"
            "\nSearch Queries\n"
            "1. `latest news philippines (2026)`"
        )
        self.assertEqual(out, expected)

    def test_title_newlines_collapsed(self):
        out = source_appendix([{"url": "https://x.io/a", "title": "line1\nline2"}], "q")
        self.assertIn("[line1 line2](https://x.io/a)", out)

    def test_caps_at_50_entries(self):
        many = [{"url": f"https://x.io/{i}", "title": f"t{i}"} for i in range(60)]
        out = source_appendix(many, "q")
        entries = [l for l in out.splitlines() if re.match(r"^\d+\. \[", l)]
        self.assertEqual(len(entries), 50)


class IncludeSourcesFlagTest(unittest.TestCase):
    def test_flag_none_uses_config_default(self):
        with patch("app.main.INCLUDE_SOURCES", True):
            self.assertTrue(main._include_sources(None))
        with patch("app.main.INCLUDE_SOURCES", False):
            self.assertFalse(main._include_sources(None))

    def test_flag_overrides_config_default(self):
        with patch("app.main.INCLUDE_SOURCES", True):
            self.assertFalse(main._include_sources(False))
        with patch("app.main.INCLUDE_SOURCES", False):
            self.assertTrue(main._include_sources(True))

    def test_string_flags_parse_like_env_values(self):
        for truthy in ("1", "true", "TRUE", "yes", "on", " on "):
            self.assertTrue(main._include_sources(truthy))
        for falsy in ("0", "false", "no", "off", "", "garbage"):
            self.assertFalse(main._include_sources(falsy))


class ChatCompletionsAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_non_stream_appends_sources_when_requested(self):
        body = {
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
        assert isinstance(resp, JSONResponse)
        data = json.loads(bytes(resp.body).decode())
        self.assertEqual(
            data["choices"][0]["message"]["content"], "Here is the news." + APPENDIX
        )

    async def test_chat_non_stream_omits_sources_by_default(self):
        body = {
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
        assert isinstance(resp, JSONResponse)
        data = json.loads(bytes(resp.body).decode())
        self.assertEqual(data["choices"][0]["message"]["content"], "Here is the news.")

    async def test_chat_stream_appends_sources_chunk(self):
        body = {
            "model": "deepseek-chat",
            "stream": True,
            "include_sources": True,
            "messages": [{"role": "user", "content": "latest news philippines"}],
        }

        async def fake_stream_turn(*args, **kwargs):
            yield StreamEvent("content", "Here is the news.")
            yield StreamEvent("search", ["latest news philippines"])
            yield StreamEvent("sources", SOURCES)

        fake_manager = AsyncMock()
        fake_manager.stream_turn = fake_stream_turn

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.chat_completions(_make_request(body))
            assert isinstance(resp, StreamingResponse)
            chunks = [c async for c in resp.body_iterator]

        def _chunk_text(chunk: str | bytes | memoryview[int]) -> str:
            if isinstance(chunk, str):
                return chunk
            if isinstance(chunk, memoryview):
                return chunk.tobytes().decode()
            return chunk.decode()

        contents: list[str] = []
        for chunk in chunks:
            line = _chunk_text(chunk).strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[len("data: ") :])
            if payload.get("object") != "chat.completion.chunk":
                continue
            choices = payload.get("choices")
            if choices and "delta" in choices[0]:
                delta = choices[0]["delta"]
                if "content" in delta:
                    contents.append(delta["content"])

        self.assertEqual("".join(contents), "Here is the news." + APPENDIX)


class ResponsesApiAppendixEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_responses_non_stream_appends_sources_when_requested(self):
        body = {
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
        assert isinstance(resp, JSONResponse)
        data = json.loads(bytes(resp.body).decode())
        self.assertEqual(
            data["output"][0]["content"][0]["text"], "Here is the news." + APPENDIX
        )

    async def test_responses_stream_appends_sources_in_events(self):
        body = {
            "model": "deepseek-chat",
            "stream": True,
            "include_sources": True,
            "input": "latest news philippines",
        }

        async def fake_stream_turn(*args, **kwargs):
            yield StreamEvent("content", "Here is the news.")
            yield StreamEvent("search", ["latest news philippines"])
            yield StreamEvent("sources", SOURCES)

        fake_manager = AsyncMock()
        fake_manager.stream_turn = fake_stream_turn

        with patch("app.main.manager", return_value=fake_manager):
            resp = await main.responses_api(_make_request(body))
            assert isinstance(resp, StreamingResponse)
            chunks = [c async for c in resp.body_iterator]

        def _event_text(chunk: str | bytes | memoryview[int]) -> str:
            if isinstance(chunk, str):
                return chunk
            if isinstance(chunk, memoryview):
                return chunk.tobytes().decode()
            return chunk.decode()

        events: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            for block in _event_text(chunk).strip().split("\n\n"):
                block = block.strip()
                if not block:
                    continue
                lines = block.splitlines()
                data_line = next(
                    (line for line in lines if line.startswith("data: ")), None
                )
                if data_line:
                    payload = json.loads(data_line[len("data: ") :])
                    events.setdefault(payload["type"], []).append(payload)

        completed = events["response.completed"][0]
        self.assertEqual(
            completed["response"]["output"][0]["content"][0]["text"],
            "Here is the news." + APPENDIX,
        )


class StreamEventsSourceEmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_sources_only_emitted_when_new_results_arrive(self):
        cm = ConversationManager(pool=AsyncMock(), pow_solver=AsyncMock())
        conv = Conversation(id="test-id", history=[], parent_message_id=None)
        client = AsyncMock()

        async def fake_stream_completion(*args, **kwargs):
            # Event with search results
            yield {
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [
                        {
                            "type": "SEARCH",
                            "results": [{"url": "https://example.com/1", "title": "1"}],
                        }
                    ],
                },
            }
            # Token delta 1
            yield {
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "RESPONSE", "content": "Hello"}],
                },
            }
            # Token delta 2
            yield {
                "event": "message",
                "data": {"v": " world"},
            }
            # New search result arrives
            yield {
                "event": "message",
                "data": {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [
                        {
                            "type": "SEARCH",
                            "results": [{"url": "https://example.com/2", "title": "2"}],
                        }
                    ],
                },
            }

        client.stream_completion = fake_stream_completion
        events = [
            ev
            async for ev in cm._stream_events(
                client,
                prompt="prompt",
                conv=conv,
                ref_file_ids=[],
                thinking_enabled=False,
                model_type=None,
            )
        ]
        source_events = [ev for ev in events if ev.kind == "sources"]
        # Should emit exactly twice (once for first result, once for second result), not on every token
        self.assertEqual(len(source_events), 2)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""SSE fragment aggregation: turn DeepSeek's patch-stream into clean deltas.

DeepSeek streams a JSON patch protocol:
  data {"v": {...}}                       initial snapshot (response object)
  data {"p": path, "o": "APPEND"/"SET", "v": ...}   explicit patches
  data {"v": "<text>"}                    implicit append to current fragment content
  data {"p": "...", "o": "BATCH", "v": [...]}       nested batch
Named events: ready / update_session / title / close.

The aggregator tracks the current fragment (THINK vs RESPONSE) and emits:
  ("reasoning", text)   for THINK fragments
  ("content", text)     for RESPONSE/SEARCHABLE fragments
  ("search", queries)   when a TOOL_SEARCH fragment appears
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
# Tuples (not sets): membership tests stay hash-free so schemaless
# JSON values (e.g. a non-string fragment "type") behave as before.
_IGNORED_EVENTS = ("ready", "update_session", "update_file")
_SEARCH_TYPES = ("TOOL_SEARCH", "SEARCH")
_CONTENT_KINDS = ("content", "reasoning")
_CONTENT_TYPES = ("", "RESPONSE")
_FRAGMENT_PATHS = ("response/fragments", "fragments")
_APPEND_OPS = ("APPEND", "SET")
_STATUS_PATHS = ("response/status", "response/quasi_status")
_SUB_PATH_RE = re.compile(r"^(?:response/)?fragments/(.+)$")


class FragmentAggregator:
    """Aggregate DeepSeek SSE patch streams into content deltas.

    Tracks the current THINK/RESPONSE fragment across snapshot and patch
    events, buffering implicit text until its fragment arrives.
    """

    def __init__(self) -> None:
        """Initialize empty fragment state."""
        self.fragments: list[dict[str, Any]] = []
        # index into self.fragments of the fragment receiving implicit appends
        self.current = -1
        self._pending: list[str] = []
        # search-result URLs in arrival order; citation N = reference_urls[N-1]
        self.reference_urls: list[str | None] = []
        # structured search results in arrival order
        self.search_results: list[dict[str, Any]] = []

    # -- public ------------------------------------------------------------

    def apply(
        self,
        event: str | None,
        data: object,
    ) -> Iterator[tuple[str, Any]]:
        """Feed one SSE event to the aggregator.

        Yields:
            tuple[str, Any]: Kinded deltas and meta/search events.

        """
        # update_file arrives on file-ref turns (live 2026-09-11: ready →
        # update_file → update_session → snapshot → patches); file state only.
        if event in _IGNORED_EVENTS:
            return
        if event == "title":
            yield from self._apply_title(data)
            return
        if event == "close":
            yield "meta", {"close": True}
            return
        if not isinstance(data, dict):
            return
        payload = cast("dict[str, Any]", data)
        if "p" not in payload and isinstance(payload.get("v"), dict):
            # initial snapshot — emit every fragment's content in stream order
            yield from self._apply_snapshot(payload)
            return
        if "p" in payload:
            yield from self._apply_patch(payload)
            return
        yield from self._apply_implicit(payload.get("v"))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _apply_title(data: object) -> Iterator[tuple[str, Any]]:
        """Yield a title meta event for title SSE payloads.

        Yields:
            tuple[str, Any]: Title meta event, if the payload carries one.

        """
        if isinstance(data, dict):
            yield "meta", {"title": data.get("content")}

    def _apply_snapshot(self, data: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Ingest an initial response snapshot in stream order.

        Yields:
            tuple[str, Any]: Content, reasoning, and search events.

        """
        response = data["v"].get("response") or {}
        self.fragments = list(response.get("fragments") or [])
        for idx, frag in enumerate(self.fragments):
            yield from self._ingest_snapshot_fragment(idx, frag)

    def _ingest_snapshot_fragment(
        self,
        idx: int,
        frag: dict[str, Any],
    ) -> Iterator[tuple[str, Any]]:
        """Record one snapshot fragment and emit its content.

        Yields:
            tuple[str, Any]: Content, reasoning, and search events.

        """
        ftype = frag.get("type")
        content = frag.get("content")
        if content:
            yield self._kind(frag), str(content)
        if ftype in _SEARCH_TYPES:
            self._capture_results(frag.get("results"))
        yield from self._emit_queries(frag)
        if self._is_content_frag(frag) or ftype == "THINK":
            self.current = idx
        elif ftype in _SEARCH_TYPES:
            # search fragment: implicit appends must buffer, not attach
            self.current = -1

    @staticmethod
    def _emit_queries(frag: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Yield a search event for a fragment carrying queries.

        Yields:
            tuple[str, Any]: Search event, if the fragment has queries.

        """
        queries = frag.get("queries")
        if isinstance(queries, list) and queries:
            yield (
                "search",
                [
                    q.get("query") if isinstance(q, dict) else str(q)
                    for q in queries
                    if (q.get("query") if isinstance(q, dict) else q)
                ],
            )

    def _apply_implicit(self, value: object) -> Iterator[tuple[str, Any]]:
        """Attach implicit string deltas to the current fragment.

        Yields:
            tuple[str, Any]: Content or reasoning deltas.

        """
        if not isinstance(value, str):
            return
        if self.current < 0:
            self._pending.append(value)
            return
        frag = self.fragments[self.current]
        kind = self._kind(frag)
        if kind in _CONTENT_KINDS:
            frag["content"] = (frag.get("content") or "") + value
            yield kind, value
        else:
            self._pending.append(value)

    @staticmethod
    def _kind(frag: dict[str, Any]) -> str:
        """Classify a fragment as reasoning, search, or content.

        Returns:
            str: One of "reasoning", "search", or "content".

        """
        ftype = frag.get("type", "")
        if ftype == "THINK":
            return "reasoning"
        if ftype in _SEARCH_TYPES:
            return "search"
        return "content"

    def _capture_results(self, value: object) -> None:
        """Record search-result URLs (positional; None when one lacks a url)."""
        if not isinstance(value, list):
            return
        for item in value:
            if not isinstance(item, dict):
                self.reference_urls.append(None)
                continue
            record = cast("dict[str, Any]", item)
            url = record.get("url")
            self.reference_urls.append(str(url) if url else None)
            self.search_results.append(record)

    @staticmethod
    def _is_content_frag(frag: dict[str, Any]) -> bool:
        """Check whether a fragment holds answer text.

        Returns:
            bool: True for empty or RESPONSE typed fragments.

        """
        return frag.get("type", "") in _CONTENT_TYPES

    def _frag_at(self, path: str) -> dict[str, Any] | None:
        """Return the fragment addressed by a patch path.

        Returns:
            dict[str, Any] | None: Last fragment, or None when unresolvable.

        """
        parts = path.split("/")
        try:
            idx = parts.index("fragments")
            rest = parts[idx + 1 :]
            if rest and rest[0] == "-1":
                return self.fragments[-1] if self.fragments else None
        except (ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _fragment_tail(path: str) -> str | None:
        """Return the trailing field of a fragment-subpath patch.

        Returns:
            str | None: Field name, or None for non-field paths.

        """
        match = _SUB_PATH_RE.match(path)
        if match is None or path.endswith("/fragments"):
            return None
        return str(match.group(1).split("/")[-1])

    def _apply_patch(self, data: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Apply one explicit patch event.

        Yields:
            tuple[str, Any]: Content, reasoning, and search events.

        """
        op = data.get("o", "SET")
        path = data["p"]
        value = data.get("v")

        if op == "BATCH":
            yield from self._apply_batch(value)
            return
        if path in _FRAGMENT_PATHS:
            yield from self._apply_fragments_patch(op, value)
            return
        tail = self._fragment_tail(path) if isinstance(path, str) else None
        if tail is not None:
            yield from self._apply_fragment_field(path, tail, op, value)
            return
        if path in _STATUS_PATHS:
            return

    def _apply_batch(self, value: object) -> Iterator[tuple[str, Any]]:
        """Apply the sub-patches of a BATCH patch in order.

        Yields:
            tuple[str, Any]: Content, reasoning, and search events.

        """
        if not isinstance(value, list):
            return
        for raw in value:
            if not isinstance(raw, dict):
                continue
            raw_dict = cast("dict[str, Any]", raw)
            sub = {"o": raw_dict.get("o", "SET"), **raw_dict}
            yield from self._apply_patch(sub)

    def _apply_fragments_patch(
        self,
        op: object,
        value: object,
    ) -> Iterator[tuple[str, Any]]:
        """Apply a patch addressed at the fragments list itself.

        Yields:
            tuple[str, Any]: Content, reasoning, and search events.

        """
        if op in _APPEND_OPS:
            new_frags = value if isinstance(value, list) else [value]
            for frag in new_frags:
                if isinstance(frag, dict):
                    yield from self._append_fragment(
                        cast("dict[str, Any]", frag),
                    )
            return
        if op == "DELETE":
            self.fragments = []
            self.current = -1
            self._pending.clear()

    def _append_fragment(self, frag: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Append one fragment and emit its prelude and content.

        Yields:
            tuple[str, Any]: Content, reasoning, and search events.

        """
        self.fragments.append(frag)
        if frag.get("type") in _SEARCH_TYPES:
            # search fragments never receive implicit appends;
            # park current so implicit text buffers until the
            # RESPONSE/THINK fragment shows up
            self.current = -1
            self._capture_results(frag.get("results"))
            yield from self._emit_queries(frag)
            return
        if self._is_content_frag(frag) or frag.get("type") == "THINK":
            self.current = len(self.fragments) - 1
            # implicit deltas buffered while this fragment was
            # pending arrived BEFORE the append, so stream them
            # first; the fragment's own initial content follows.
            yield from self._flush_pending(frag)

    def _flush_pending(self, frag: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        """Replay buffered implicit deltas into a new fragment.

        Yields:
            tuple[str, Any]: Content or reasoning deltas.

        """
        prelude, self._pending = self._pending, []
        for piece in prelude:
            yield self._kind(frag), piece
        initial = str(frag.get("content") or "")
        full = "".join(prelude) + initial
        if full:
            frag["content"] = full
            if initial:
                yield self._kind(frag), initial

    def _apply_fragment_field(
        self,
        path: str,
        tail: str,
        op: object,
        value: object,
    ) -> Iterator[tuple[str, Any]]:
        """Apply a patch addressed at one fragment field.

        Yields:
            tuple[str, Any]: Content or reasoning deltas.

        """
        frag = self._frag_at(path)
        if frag is None or tail == "-1":
            return
        if tail == "results" and op in _APPEND_OPS:
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, dict):
                    self._capture_results([item])
            frag[tail] = value
            return
        if op == "SET":
            frag[tail] = value
            if tail == "content" and isinstance(value, str) and value:
                yield self._kind(frag), value
        elif op == "APPEND":
            if tail != "content":
                # references etc.: update state only — stringified
                # metadata must never leak out as answer text
                frag[tail] = value
                return
            appended = str(value)
            frag[tail] = (frag.get(tail) or "") + appended
            yield self._kind(frag), appended

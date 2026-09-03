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

from collections.abc import Iterator
import re
from typing import Any


class FragmentAggregator:
    def __init__(self) -> None:
        self.fragments: list[dict[str, Any]] = []
        # index into self.fragments of the fragment receiving implicit appends
        self.current = -1
        self._pending: list[str] = []
        # search-result URLs in arrival order; citation N = reference_urls[N-1]
        self.reference_urls: list[str | None] = []
        # structured search results in arrival order
        self.search_results: list[dict[str, Any]] = []

    # -- public ------------------------------------------------------------

    def apply(self, event: str | None, data: Any) -> Iterator[tuple[str, Any]]:
        """Feed one SSE event; yield ('reasoning'|'content'|'search'|'meta', payload)."""
        if event in ("ready", "update_session"):
            return
        if event == "title":
            if isinstance(data, dict):
                yield "meta", {"title": data.get("content")}
            return
        if event == "close":
            yield "meta", {"close": True}
            return
        if not isinstance(data, dict):
            return

        if "p" not in data and isinstance(data.get("v"), dict):
            # initial snapshot — emit every fragment's content in stream order
            response = data["v"].get("response") or {}
            self.fragments = list(response.get("fragments") or [])
            for idx, frag in enumerate(self.fragments):
                ftype = frag.get("type")
                content = frag.get("content")
                if content:
                    yield self._kind(frag), str(content)
                if ftype in ("TOOL_SEARCH", "SEARCH"):
                    self._capture_results(frag.get("results"))
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
                if self._is_content_frag(frag) or ftype == "THINK":
                    self.current = idx
                elif ftype in ("TOOL_SEARCH", "SEARCH"):
                    # search fragment: implicit appends must buffer, not attach
                    self.current = -1
            return

        if "p" in data:
            yield from self._apply_patch(data)
            return

        value = data.get("v")
        if isinstance(value, str) and self.current >= 0:
            frag = self.fragments[self.current]
            kind = self._kind(frag)
            if kind in ("content", "reasoning"):
                frag["content"] = (frag.get("content") or "") + value
                yield kind, value
            else:
                self._pending.append(value)
        elif isinstance(value, str):
            self._pending.append(value)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _kind(frag: dict[str, Any]) -> str:
        ftype = frag.get("type", "")
        if ftype == "THINK":
            return "reasoning"
        if ftype in ("TOOL_SEARCH", "SEARCH"):
            return "search"
        return "content"

    def _capture_results(self, value: Any) -> None:
        """Record search-result URLs (positional; None when a result lacks one)."""
        if not isinstance(value, list):
            return
        for item in value:
            url = item.get("url") if isinstance(item, dict) else None
            self.reference_urls.append(str(url) if url else None)
            if isinstance(item, dict):
                self.search_results.append(item)

    def _is_content_frag(self, frag: dict[str, Any]) -> bool:
        return frag.get("type", "") in ("", "RESPONSE")

    def _frag_at(self, path: str) -> dict[str, Any] | None:
        """path like 'response/fragments/-1/results' -> last fragment dict."""
        parts = path.split("/")
        try:
            idx = parts.index("fragments")
            rest = parts[idx + 1 :]
            if rest and rest[0] == "-1":
                return self.fragments[-1] if self.fragments else None
        except (ValueError, IndexError):
            pass
        return None

    def _apply_patch(self, data: dict[str, Any]) -> Iterator[tuple[str, Any]]:
        op = data.get("o", "SET")
        path = data["p"]
        value = data.get("v")

        if op == "BATCH":
            for sub in value or []:
                if isinstance(sub, dict):
                    sub = {"o": sub.get("o", "SET"), **sub}
                    yield from self._apply_patch(sub)
            return

        if path in ("response/fragments", "fragments"):
            if op in ("APPEND", "SET"):
                new_frags = value if isinstance(value, list) else [value]
                for frag in new_frags:
                    if not isinstance(frag, dict):
                        continue
                    self.fragments.append(frag)
                    if frag.get("type") in ("TOOL_SEARCH", "SEARCH"):
                        # search fragments never receive implicit appends;
                        # park current so implicit text buffers until the
                        # RESPONSE/THINK fragment shows up
                        self.current = -1
                        self._capture_results(frag.get("results"))
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
                        continue
                    if self._is_content_frag(frag) or frag.get("type") == "THINK":
                        self.current = len(self.fragments) - 1
                        # implicit deltas buffered while this fragment was
                        # pending arrived BEFORE the append, so stream them
                        # first; the fragment's own initial content follows.
                        prelude, self._pending = self._pending, []
                        for piece in prelude:
                            yield self._kind(frag), piece
                        initial = str(frag.get("content") or "")
                        full = "".join(prelude) + initial
                        if full:
                            frag["content"] = full
                            if initial:
                                yield self._kind(frag), initial

                return
            if op == "DELETE":
                self.fragments = []
                self.current = -1
                self._pending.clear()
                return

        sub = re.match(r"^(?:response/)?fragments/(.+)$", path)
        if sub and not path.endswith("/fragments"):
            frag = self._frag_at(path)
            parts_after = sub.group(1).split("/")
            tail = parts_after[-1]
            if frag is None or tail == "-1":
                return
            if tail == "results" and op in ("SET", "APPEND"):
                items = value if isinstance(value, list) else [value]
                for it in items:
                    if isinstance(it, dict):
                        self._capture_results([it])
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
            return

        if path == "response/status" or path == "response/quasi_status":
            return

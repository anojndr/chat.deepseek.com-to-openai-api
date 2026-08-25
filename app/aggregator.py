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
from typing import Any


class FragmentAggregator:
    def __init__(self) -> None:
        self.fragments: list[dict[str, Any]] = []
        # index into self.fragments of the fragment receiving implicit appends
        self.current = -1
        self._pending: list[str] = []

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
            # initial snapshot
            response = data["v"].get("response") or {}
            self.fragments = list(response.get("fragments") or [])
            if self.fragments:
                self.current = len(self.fragments) - 1
                frag = self.fragments[self.current]
                kind = self._kind(frag)
                content = frag.get("content")
                if content:
                    yield kind, content
                if frag.get("type") == "TOOL_SEARCH" and frag.get("queries"):
                    yield "search", frag.get("queries")
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
                    yield from (
                        item
                        for item in self._apply_patch(sub)
                        if item[0] != "search"
                    )
            return

        if path in ("response/fragments", "fragments"):
            if op in ("APPEND", "SET"):
                new_frags = value if isinstance(value, list) else [value]
                for frag in new_frags:
                    if not isinstance(frag, dict):
                        continue
                    self.fragments.append(frag)
                    if frag.get("type") == "TOOL_SEARCH":
                        if frag.get("queries"):
                            yield "search", [
                                q.get("query") for q in frag["queries"] if q.get("query")
                            ]
                        continue  # search fragment: never the append target
                    if self._is_content_frag(frag) or frag.get("type") == "THINK":
                        self.current = len(self.fragments) - 1
                        if self._pending:
                            for piece in self._pending:
                                frag["content"] = (frag.get("content") or "") + piece
                                yield self._kind(frag), piece
                            self._pending = []
                        elif frag.get("content"):
                            # fragment arrives already carrying text (e.g. the
                            # final RESPONSE append) — emit it
                            yield self._kind(frag), str(frag["content"])
                return
            if op == "DELETE":
                self.fragments = []
                self.current = -1
                return

        if path.startswith("response/fragments/"):
            frag = self._frag_at(path)
            tail = path.split("/")[-1]
            if frag is not None and tail != "-1":
                if op == "SET":
                    frag[tail] = value
                    if tail == "content":
                        yield self._kind(frag), None  # marker; consumers ignore
                elif op == "APPEND":
                    frag[tail] = (frag.get(tail) or "") + str(value)
                    yield self._kind(frag), str(value)
                return

        if path == "response/status" or path == "response/quasi_status":
            return

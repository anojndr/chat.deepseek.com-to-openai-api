"""Rewrite DeepSeek [citation:N] markers into markdown links.

DeepSeek's search-enabled answers embed markers like ``[citation:3]`` where N
is the 1-based position of a search result in the order the results arrived
(each TOOL_SEARCH fragment's ``results`` list, flattened across stages). The
aggregator collects those URLs; this module rewrites the answer text so every
marker becomes ``[citation:N](https://…)``.

Streaming-safe: a marker can be split across two SSE deltas, so feed() holds
back a trailing partial marker and finish() flushes whatever remains.
"""

from __future__ import annotations

import re

_CITE = re.compile(r"\[citation:(\d+)\]")
_ADJACENT = re.compile(r"(\[citation:\d+\])(?=\[citation:)")
_HEAD = "[citation:]"


def _holdback_len(text: str) -> int:
    """Length of a trailing partial '[citation:…' marker, or 0."""
    idx = text.rfind("[")
    if idx == -1:
        return 0
    suffix = text[idx:]
    if "]" in suffix:
        return 0
    if len(suffix) > len(_HEAD) + 6:  # longest possible partial
        return 0
    if _HEAD.startswith(suffix) or suffix.startswith("[citation:") and suffix[len("[citation:") :].isdigit():
        return len(suffix)
    return 0


class CitationRewriter:
    """Incrementally rewrite [citation:N] markers as they stream past."""

    def __init__(self, reference_urls: list[str | None]) -> None:
        # live list from FragmentAggregator.reference_urls — grows as TOOL_SEARCH
        # fragments arrive, always before any content that cites them
        self._refs = reference_urls
        self._pending = ""
        self._ended_with_cite = False

    def _replace(self, match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(self._refs):
            url = self._refs[n - 1]
            if url:
                return f"[{match.group(0)[1:-1]}]({url})"
        return match.group(0)

    def feed(self, text: str) -> str:
        buf = self._pending + text
        self._pending = ""
        if self._ended_with_cite and buf.startswith("[citation:"):
            buf = " " + buf
            self._ended_with_cite = False
        buf = _ADJACENT.sub(r"\1 ", buf)
        out = _CITE.sub(self._replace, buf)
        hold = _holdback_len(out)
        if hold:
            self._pending = out[-hold:]
            out = out[: -hold]
        if out:
            self._ended_with_cite = bool(re.search(r"(\[citation:\d+\](?:\(\S*\))?)$", out))
        return out

    def finish(self) -> str:
        rest = self._pending
        self._pending = ""
        self._ended_with_cite = False
        return rest


def rewrite_citations(text: str, reference_urls: list[str | None]) -> str:
    """One-shot variant for non-streaming answers."""
    rw = CitationRewriter(reference_urls)
    return rw.feed(text) + rw.finish()

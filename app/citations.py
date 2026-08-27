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
from typing import Any
from urllib.parse import urlparse

SOURCE_APPENDIX_MAX = 50


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def source_appendix(sources: list[Any], queries: list[str] | str | None = None) -> str:
    """Bridge source appendix for llmcord-go's "Show Sources" button.

    Matches the appendix contract parsed by llmcord-go across bridge providers:
        \n\nSources
        1. [Title](url) (domain) via `query`

        Search Queries
        1. `query`
    """
    entries: list[str] = []
    seen_urls: set[str] = set()

    query_str = ""
    if isinstance(queries, list) and queries:
        first_q = queries[0]
        query_str = first_q if isinstance(first_q, str) else str(first_q) if first_q is not None else ""
    elif isinstance(queries, str):
        query_str = queries
    clean_query = " ".join(query_str.split()).replace("`", "'").strip() if query_str else ""

    for src in sources[:SOURCE_APPENDIX_MAX]:
        raw_url = None
        title = ""
        if isinstance(src, dict):
            raw_url = src.get("url")
            title = str(src.get("title") or "")
        elif isinstance(src, str):
            raw_url = src

        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = raw_url.strip().replace("\n", "").replace("\r", "").replace("\t", "")
        url = url.replace(")", "%29").replace(" ", "%20")
        if not url or url.lower() in seen_urls:
            continue
        seen_urls.add(url.lower())

        clean_title = " ".join(title.split()).replace("[", "(").replace("]", ")")
        clean_title = clean_title or url

        entry = f"[{clean_title}]({url})"
        host = _host_of(url)
        if clean_title != url and host:
            entry += f" ({host})"
        if clean_query:
            entry += f" via `{clean_query}`"
        entries.append(entry)

    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))
    if clean_query:
        lines.append("")
        lines.append("Search Queries")
        lines.append(f"1. `{clean_query}`")
    return "\n\n" + "\n".join(lines)

_CITE = re.compile(r"\[!?citation:(\d+)\]")
_ADJACENT = re.compile(r"(\[!?[cC]itation:\d+\])(?=\[!?[cC]itation:)")
_HEADS = ("[citation:", "[!citation:")


def _fix_space_before_punct(text: str) -> str:
    """Remove spaces before punctuation marks (periods, commas, etc.)."""
    return re.sub(r" +([.,!?:;])(?=[\s\n\r]|$)", r"\1", text)


def _holdback_len(text: str) -> int:
    """Length of a trailing partial '[citation:…' or '[!citation:…' marker, or 0."""
    idx = text.rfind("[")
    if idx == -1:
        return 0
    suffix = text[idx:]
    if "]" in suffix:
        return 0
    if len(suffix) > 20:  # longest possible partial
        return 0
    for h in _HEADS:
        if h.startswith(suffix) or (suffix.lower().startswith(h) and suffix[len(h) :].isdigit()):
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
                return f"[citation:{n}]({url})"
        return match.group(0)

    def feed(self, text: str) -> str:
        buf = self._pending + text
        self._pending = ""
        if self._ended_with_cite and re.match(r"^\[!?citation:", buf, re.IGNORECASE):
            buf = " " + buf
            self._ended_with_cite = False
        buf = _ADJACENT.sub(r"\1 ", buf)
        out = _CITE.sub(self._replace, buf)
        out = _fix_space_before_punct(out)
        hold = _holdback_len(out)
        if hold:
            self._pending = out[-hold:]
            out = out[: -hold]
        space_hold = 0
        while len(out) > space_hold and out[-1 - space_hold] == " ":
            space_hold += 1
        if space_hold > 0:
            self._pending = out[-space_hold:] + self._pending
            out = out[:-space_hold]
        if out:
            self._ended_with_cite = bool(re.search(r"(\[!?citation:\d+\](?:\(\S*\))?)$", out))
        return out

    def finish(self) -> str:
        rest = self._pending
        self._pending = ""
        self._ended_with_cite = False
        return _fix_space_before_punct(rest)


def rewrite_citations(text: str, reference_urls: list[str | None]) -> str:
    """One-shot variant for non-streaming answers."""
    rw = CitationRewriter(reference_urls)
    return rw.feed(text) + rw.finish()

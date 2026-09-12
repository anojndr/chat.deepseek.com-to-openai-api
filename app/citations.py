# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
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
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Sequence

SOURCE_APPENDIX_MAX = 50

# Longest possible trailing partial marker held back during streaming.
_MAX_PARTIAL_MARKER_LEN = 20


def _host_of(url: str) -> str:
    """Extract the lowercase host from a URL.

    Returns:
        Host portion, or empty string when unparsable.

    """
    try:
        return (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""


def _normalize_query(queries: Sequence[object] | str | None) -> str:
    """Normalize the first query into a clean display string.

    Returns:
        Cleaned query text, or empty string when absent.

    """
    query_str = ""
    if isinstance(queries, str):
        query_str = queries
    elif isinstance(queries, list) and queries:
        first = queries[0]
        if isinstance(first, str):
            query_str = first
        elif first is not None:
            query_str = str(first)
    if not query_str:
        return ""
    return " ".join(query_str.split()).replace("`", "'").strip()


def _source_url_and_title(src: object) -> tuple[str, str] | None:
    """Extract and clean the URL and title from a source entry.

    Returns:
        Tuple of cleaned URL and raw title, or None when unusable.

    """
    raw_url: object = None
    title = ""
    if isinstance(src, dict):
        raw_url = src.get("url")
        title = str(src.get("title") or "")
    elif isinstance(src, str):
        raw_url = src
    if not isinstance(raw_url, str):
        return None
    stripped = raw_url.strip()
    if not stripped:
        return None
    url = stripped.replace("\n", "").replace("\r", "").replace("\t", "")
    url = url.replace(")", "%29").replace(" ", "%20")
    if not url:
        return None
    return url, title


def _build_source_entry(
    src: object,
    seen_urls: set[str],
    clean_query: str,
) -> str | None:
    """Build a single markdown source line for the appendix.

    Returns:
        Formatted entry, or None when the source is skipped.

    """
    parts = _source_url_and_title(src)
    if parts is None:
        return None
    url, title = parts
    lowered = url.lower()
    if lowered in seen_urls:
        return None
    seen_urls.add(lowered)
    clean_title = " ".join(title.split()).replace("[", "(").replace("]", ")")
    clean_title = clean_title or url
    entry = f"[{clean_title}]({url})"
    host = _host_of(url)
    if clean_title != url and host:
        entry += f" ({host})"
    if clean_query:
        entry += f" via `{clean_query}`"
    return entry


def source_appendix(
    sources: Sequence[object],
    queries: Sequence[object] | str | None = None,
) -> str:
    """Create a source appendix for the Show Sources button.

    Formats collected search results as a Sources list with an
    optional Search Queries section.

    Returns:
        Appendix text starting with blank lines, or empty string.

    """
    clean_query = _normalize_query(queries)
    entries: list[str] = []
    seen_urls: set[str] = set()
    for src in list(sources)[:SOURCE_APPENDIX_MAX]:
        entry = _build_source_entry(src, seen_urls, clean_query)
        if entry is not None:
            entries.append(entry)
    if not entries:
        return ""
    lines = ["Sources"]
    lines.extend(f"{i}. {entry}" for i, entry in enumerate(entries, start=1))
    if clean_query:
        lines.extend(["", "Search Queries", f"1. `{clean_query}`"])
    return "\n\n" + "\n".join(lines)


_CITE = re.compile(r"\[!?citation:(\d+)\]")
_ADJACENT = re.compile(r"(\[!?[cC]itation:\d+\])(?=\[!?[cC]itation:)")
_HEADS = ("[citation:", "[!citation:")


def _fix_space_before_punct(text: str) -> str:
    """Remove spaces before punctuation marks.

    Returns:
        Text with stray spaces before punctuation removed.

    """
    return re.sub(r" +([.,!?:;])(?=[\s\n\r]|$)", r"\1", text)


def _holdback_len(text: str) -> int:
    """Measure a trailing partial citation marker.

    Returns:
        Length of the partial marker, or zero when absent.

    """
    idx = text.rfind("[")
    if idx == -1:
        return 0
    suffix = text[idx:]
    if "]" in suffix:
        return 0
    if len(suffix) > _MAX_PARTIAL_MARKER_LEN:
        return 0
    for head in _HEADS:
        if head.startswith(suffix) or (
            suffix.lower().startswith(head) and suffix[len(head) :].isdigit()
        ):
            return len(suffix)
    return 0


class CitationRewriter:
    """Incrementally rewrite [citation:N] markers as they stream past."""

    def __init__(self, reference_urls: list[str | None]) -> None:
        """Create a rewriter over the shared reference URL list."""
        # Live list from FragmentAggregator.reference_urls — grows as
        # TOOL_SEARCH fragments arrive, always before any content
        # that cites them.
        self._refs = reference_urls
        self._pending = ""
        self._ended_with_cite = False

    def _replace(self, match: re.Match[str]) -> str:
        """Replace one citation marker with its markdown link.

        Returns:
            Linked marker, or the original text when unresolvable.

        """
        num = int(match.group(1))
        if 1 <= num <= len(self._refs):
            url = self._refs[num - 1]
            if url:
                return f"[citation:{num}]({url})"
        return match.group(0)

    def feed(self, text: str) -> str:
        """Feed a text chunk through the rewriter.

        Returns:
            Rewritten text with complete markers linked.

        """
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
            out = out[:-hold]
        space_hold = 0
        while len(out) > space_hold and out[-1 - space_hold] == " ":
            space_hold += 1
        if space_hold > 0:
            self._pending = out[-space_hold:] + self._pending
            out = out[:-space_hold]
        if out:
            self._ended_with_cite = bool(
                re.search(r"(\[!?citation:\d+\](?:\(\S*\))?)$", out),
            )
        return out

    def finish(self) -> str:
        """Flush any held-back trailing text.

        Returns:
            Remaining buffered text.

        """
        rest = self._pending
        self._pending = ""
        self._ended_with_cite = False
        return _fix_space_before_punct(rest)


def rewrite_citations(text: str, reference_urls: list[str | None]) -> str:
    """Rewrite all citation markers in one shot.

    Returns:
        Text with every marker replaced by a markdown link.

    """
    rw = CitationRewriter(reference_urls)
    return rw.feed(text) + rw.finish()

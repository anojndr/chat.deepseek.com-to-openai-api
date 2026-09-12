# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Translate OpenAI message lists into (prompt, files, session-key).

Multi-turn strategy: the first request of a conversation creates a DeepSeek
chat_session; follow-ups pass parent_message_id so DeepSeek keeps native
context. The caller only needs the last user turn as the prompt. Files are
uploaded once per new conversation and referenced via ref_file_ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from .models import decode_data_url


@dataclass
class PreparedTurn:
    """Prepared prompt and upload files for one turn."""

    prompt: str
    files: list[tuple[str, bytes, str | None]] = field(default_factory=list)
    system_suffix: str | None = None


class FilePart(TypedDict, total=False):
    """File or image part extracted from message content."""

    kind: str
    url: str | None
    filename: str | None
    file_data: str | None
    file_id: str | None


_TEXTUAL_TYPES = {"text", "input_text", "output_text", "summary_text"}
_IMAGE_TYPES = {"image_url", "input_image"}
# Tuple (not set): membership stays hash-free for non-string roles.
_SYSTEM_ROLES = ("system", "developer")
_ROLE_PREFIXES = {
    "user": "[user]",
    "assistant": "[assistant]",
    "tool": "[tool result]",
}
_MIME_EXT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/svg+xml": "svg",
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
    "application/json": "json",
    "text/html": "html",
}


def _canonical_role(role: object) -> str:
    """Map an OpenAI role to the canonical history role name.

    Returns:
        str: One of "system", "assistant", "tool", or "user".

    """
    if role in _SYSTEM_ROLES:
        return "system"
    if role == "assistant":
        return "assistant"
    if role == "tool":
        return "tool"
    return "user"


def _with_tool_markers(text: str, msg: dict[str, Any]) -> str:
    """Append called-tool and tool-result markers to message text.

    Returns:
        str: Text with markers applied.

    """
    if msg.get("tool_calls"):
        calls = ", ".join(
            (c.get("function") or {}).get("name", "?") for c in msg["tool_calls"]
        )
        text = (text + f"\n[called tools: {calls}]").strip()
    if msg.get("tool_call_id"):
        text = f"[result for {msg['tool_call_id']}] {text}".strip()
    return text


def _image_file_part(part: dict[str, Any]) -> FilePart:
    """Build a file part for an image content part.

    Returns:
        FilePart: Image part with resolved URL.

    """
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return {"kind": "image", "url": url}


def _input_file_part(part: dict[str, Any]) -> FilePart:
    """Build a file part for an input_file content part.

    Returns:
        FilePart: File part with metadata.

    """
    file_info = part.get("file") or {}
    return {
        "kind": "file",
        "filename": file_info.get("filename"),
        "file_data": part.get("file_data") or file_info.get("file_data"),
        "file_id": part.get("file_id") or file_info.get("file_id"),
    }


def _upload_file_part(part: dict[str, Any]) -> FilePart:
    """Build a file part for a file content part.

    Returns:
        FilePart: File part with metadata.

    """
    file_info = part.get("file") or {}
    return {
        "kind": "file",
        "filename": file_info.get("filename"),
        "file_data": part.get("file_data") or file_info.get("file_data"),
    }


def _flatten_part(part: object, texts: list[str], files: list[FilePart]) -> None:
    """Sort one content part into the text or file accumulators."""
    if not isinstance(part, dict):
        texts.append(str(part))
        return
    entry = cast("dict[str, Any]", part)
    ptype = entry.get("type")
    if ptype in _TEXTUAL_TYPES or ptype == "text":
        texts.append(entry.get("text") or "")
    elif ptype in _IMAGE_TYPES:
        files.append(_image_file_part(entry))
    elif ptype == "input_file":
        files.append(_input_file_part(entry))
    elif ptype == "file":
        files.append(_upload_file_part(entry))
    elif ptype == "refusal":
        texts.append(entry.get("refusal") or "")
    else:
        # tool/function output etc. rendered as JSON text
        texts.append(json.dumps(entry, ensure_ascii=False))


def _flatten_content(content: object) -> tuple[str, list[FilePart]]:
    """Return (text, file_parts) from OpenAI content: str or parts list.

    Returns:
        tuple[str, list[FilePart]]: Joined text and extracted file parts.

    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    files: list[FilePart] = []
    if isinstance(content, list):
        for part in content:
            _flatten_part(part, texts, files)
    return "\n".join(p for p in texts if p), files


def ensure_extension(filename: str, mime: str | None) -> str:
    """DeepSeek rejects uploads whose names lack a known extension.

    Returns:
        str: Filename with a suitable extension.

    """
    if "." in filename:
        return filename
    ext = _MIME_EXT.get(mime or "", "bin")
    return f"{filename}.{ext}"


def item_hash(prev: str, role: str, canon: str) -> str:
    """Compute the rolling hash for one history item.

    Returns:
        str: Truncated hex digest chaining the previous hash.

    """
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(f"|{role}|".encode())
    h.update(canon.encode())
    return h.hexdigest()[:32]


def _history_item(msg: dict[str, Any]) -> tuple[str, str, list[FilePart]]:
    """Split one history message into role, text, and file parts.

    Returns:
        tuple[str, str, list[FilePart]]: Canonical role, marker text, files.

    """
    role_name = _canonical_role(msg.get("role", "user"))
    text, files = _flatten_content(msg.get("content"))
    return role_name, _with_tool_markers(text, msg), files


def _canon_with_files(text: str, files: list[FilePart]) -> str:
    """Append file fingerprints to canonical message text.

    Returns:
        str: Text suffixed with file descriptors.

    """
    if not files:
        return text
    extras: list[str] = []
    for part in files:
        part_kind = part.get("kind", "")
        part_name = part.get("filename", "")
        part_data = str(part.get("file_data") or part.get("url") or "")
        extras.append(f"{part_kind}:{part_name}:{len(part_data)}")
    return text + "\x00" + "\x00".join(extras)


def compute_history_hashes(
    messages: list[dict[str, Any]],
    instructions: str | None = None,
) -> tuple[list[str], str]:
    """Compute prefix rolling hashes for a message history.

    Returns:
        tuple[list[str], str]: Rolling hashes and joined system text.

    """
    system_chunks: list[str] = []
    if instructions:
        system_chunks.append(instructions)

    items: list[tuple[str, str]] = []
    for msg in messages:
        role_name, text, files = _history_item(msg)
        if role_name == "system":
            system_chunks.append(text)
        else:
            items.append((role_name, _canon_with_files(text, files)))

    system_text = "\n\n".join(s for s in system_chunks if s)
    prev = item_hash("", "system", system_text) if system_text else ""
    hashes: list[str] = []
    for role_name, canon in items:
        prev = item_hash(prev, role_name, canon)
        hashes.append(prev)
    return hashes, system_text


def _split_turn(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[tuple[str, str]], str, list[FilePart]]:
    """Split messages into system chunks, history, and latest user turn.

    Returns:
        tuple: System chunks, (role, text) history, latest user text, files.

    """
    system_chunks: list[str] = []
    history: list[tuple[str, str]] = []
    latest_user_text = ""
    latest_user_files: list[FilePart] = []
    for msg in messages:
        role_name = _canonical_role(msg.get("role", "user"))
        text, files = _flatten_content(msg.get("content"))
        text = _with_tool_markers(text, msg)
        if role_name == "system":
            system_chunks.append(text)
        else:
            history.append((role_name, text))
            if role_name == "user":
                latest_user_text = text
                latest_user_files = files
    return system_chunks, history, latest_user_text, latest_user_files


def _first_turn_prompt(
    system_chunks: list[str],
    history: list[tuple[str, str]],
) -> str:
    """Join system chunks and labeled history into a first-turn prompt.

    Returns:
        str: Full conversation text.

    """
    lines: list[str] = list(system_chunks)
    for role_name, text in history:
        prefix = _ROLE_PREFIXES[role_name]
        lines.append(
            f"{prefix} {text}" if text and not text.startswith(prefix) else text,
        )
    return "\n\n".join(lines).strip()


def _followup_prompt(system_chunks: list[str], latest_user_text: str) -> str:
    """Build a follow-up prompt from the newest user message.

    Returns:
        str: User text with an optional system reminder.

    """
    prompt = latest_user_text.strip()
    if system_chunks:
        prompt = "[system reminder]\n" + "\n".join(system_chunks) + "\n\n" + prompt
    return prompt


def _all_message_files(messages: list[dict[str, Any]]) -> list[FilePart]:
    """Collect file parts across a first-turn conversation.

    Returns:
        list[FilePart]: Every file part in message order.

    """
    all_files: list[FilePart] = []
    for msg in messages:
        _, files = _flatten_content(msg.get("content"))
        all_files.extend(files)
    return all_files


def _decode_part_url(url: str, file_data: str) -> tuple[bytes, str] | None:
    """Decode the first data: URL of a file part, if any.

    Remote URLs are left alone on purpose: downloading arbitrary remote
    content server-side would be a surprise side effect.

    Returns:
        tuple[bytes, str] | None: Decoded bytes and mime, or None.

    """
    if url.startswith("data:"):
        return decode_data_url(url)
    if file_data.startswith("data:"):
        return decode_data_url(file_data)
    return None


def _decode_turn_files(
    all_files: list[FilePart],
) -> list[tuple[str, bytes, str | None]]:
    """Decode data: URLs of turn files into upload payloads.

    Returns:
        list[tuple[str, bytes, str | None]]: Filename, bytes, and mime.

    """
    prepared_files: list[tuple[str, bytes, str | None]] = []
    for index, part in enumerate(all_files):
        filename = part.get("filename") or f"file-{index + 1}"
        url = part.get("url") or ""
        file_data = part.get("file_data") or ""
        decoded = _decode_part_url(url, file_data)
        if decoded:
            raw, mime = decoded
            prepared_files.append((ensure_extension(filename, mime), raw, mime))
    return prepared_files


def prepare_turn(
    messages: list[dict[str, Any]],
    *,
    is_first_turn: bool,
    instructions: str | None = None,
) -> PreparedTurn:
    """Build the prompt for the current turn.

    First turn: full conversation text (system + history + user) because the
    fresh DeepSeek session has no context yet. Later turns: only the newest
    user message - DeepSeek holds prior turns via parent_message_id.

    Returns:
        PreparedTurn: Prompt with decoded upload files.

    """
    system_chunks, history, latest_user_text, latest_user_files = _split_turn(
        messages,
    )
    if instructions:
        system_chunks.insert(0, instructions)
    if is_first_turn:
        prompt = _first_turn_prompt(system_chunks, history)
        all_files = _all_message_files(messages)
    else:
        prompt = _followup_prompt(system_chunks, latest_user_text)
        all_files = latest_user_files
    return PreparedTurn(prompt=prompt, files=_decode_turn_files(all_files))

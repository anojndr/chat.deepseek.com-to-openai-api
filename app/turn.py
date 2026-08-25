"""Translate OpenAI message lists into (prompt, files, session-key).

Multi-turn strategy: the first request of a conversation creates a DeepSeek
chat_session; follow-ups pass parent_message_id so DeepSeek keeps native
context. The caller only needs the last user turn as the prompt. Files are
uploaded once per new conversation and referenced via ref_file_ids.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedTurn:
    prompt: str
    files: list[tuple[str, bytes, str | None]] = field(default_factory=list)
    system_suffix: str | None = None


_TEXTUAL_TYPES = {"text", "input_text", "output_text", "summary_text"}
_IMAGE_TYPES = {"image_url", "input_image"}


def _flatten_content(content: Any) -> tuple[str, list[dict[str, Any]]]:
    """Return (text, file_parts) from OpenAI content: str or parts list."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    parts_out: list[str] = []
    files: list[dict[str, Any]] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                parts_out.append(str(part))
                continue
            ptype = part.get("type")
            if ptype in _TEXTUAL_TYPES:
                parts_out.append(part.get("text") or "")
            elif ptype == "text":
                parts_out.append(part.get("text") or "")
            elif ptype in _IMAGE_TYPES:
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                files.append({"kind": "image", "url": url})
            elif ptype == "input_file":
                file_info = part.get("file") or {}
                files.append(
                    {
                        "kind": "file",
                        "filename": file_info.get("filename"),
                        "file_data": part.get("file_data") or file_info.get("file_data"),
                        "file_id": part.get("file_id") or file_info.get("file_id"),
                    }
                )
            elif ptype == "file":
                file_info = part.get("file") or {}
                files.append(
                    {
                        "kind": "file",
                        "filename": file_info.get("filename"),
                        "file_data": part.get("file_data") or file_info.get("file_data"),
                    }
                )
            elif ptype == "refusal":
                parts_out.append(part.get("refusal") or "")
            else:
                # tool/function output etc. rendered as JSON text
                parts_out.append(json.dumps(part, ensure_ascii=False))
    return "\n".join(p for p in parts_out if p), files


def _data_to_bytes(url: str) -> tuple[bytes, str] | None:
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url or "", re.DOTALL)
    if not match:
        return None
    mime = match.group(1) or "application/octet-stream"
    payload = match.group(3)
    try:
        if match.group(2):
            return base64.b64decode(payload, validate=False), mime
        from urllib.parse import unquote_to_bytes

        return unquote_to_bytes(payload), mime
    except (binascii.Error, ValueError):
        return None


_MIME_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
    "image/webp": "webp", "image/bmp": "bmp", "image/svg+xml": "svg",
    "application/pdf": "pdf", "text/plain": "txt", "text/markdown": "md",
    "text/csv": "csv", "application/json": "json", "text/html": "html",
}


def ensure_extension(filename: str, mime: str | None) -> str:
    """DeepSeek rejects uploads whose names lack a known extension."""
    if "." in filename:
        return filename
    ext = _MIME_EXT.get(mime or "", "bin")
    return f"{filename}.{ext}"


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
    """
    system_chunks: list[str] = []
    history: list[tuple[str, str]] = []
    latest_user_text = ""
    latest_user_files: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            role_name = "tool"
        elif role == "assistant":
            role_name = "assistant"
        elif role in ("system", "developer"):
            role_name = "system"
        else:
            role_name = "user"
        text, files = _flatten_content(msg.get("content"))
        if msg.get("tool_calls"):
            calls = ", ".join(
                (c.get("function") or {}).get("name", "?") for c in msg["tool_calls"]
            )
            text = (text + f"\n[called tools: {calls}]").strip()
        if msg.get("tool_call_id"):
            text = f"[result for {msg['tool_call_id']}] {text}".strip()
        if role_name == "system":
            system_chunks.append(text)
        else:
            history.append((role_name, text))
            if role_name == "user":
                latest_user_text = text
                latest_user_files = files

    if instructions:
        system_chunks.insert(0, instructions)

    if is_first_turn:
        lines: list[str] = []
        if system_chunks:
            lines.extend(system_chunks)
        for role_name, text in history:
            prefix = {"user": "[user]", "assistant": "[assistant]", "tool": "[tool result]"}[
                role_name
            ]
            lines.append(f"{prefix} {text}" if text and not text.startswith(prefix) else text)
        prompt = "\n\n".join(lines).strip()
    else:
        prompt = latest_user_text.strip()
        if system_chunks:
            prompt = (
                "[system reminder]\n"
                + "\n".join(system_chunks)
                + "\n\n"
                + prompt
            )

    all_files: list[dict[str, Any]] = []
    if is_first_turn:
        # collect every file part across the whole first-turn conversation
        for msg in messages:
            _, files = _flatten_content(msg.get("content"))
            all_files.extend(files)
    else:
        all_files = latest_user_files

    prepared_files: list[tuple[str, bytes, str | None]] = []
    for i, f in enumerate(all_files):
        filename = f.get("filename") or f"file-{i + 1}"
        url = f.get("url") or ""
        file_data = f.get("file_data") or ""
        if url.startswith("data:"):
            decoded = _data_to_bytes(url)
        elif file_data.startswith("data:"):
            decoded = _data_to_bytes(file_data)
        else:
            # http(s) URLs are left alone on purpose: downloading arbitrary
            # remote content server-side would be a surprise side effect.
            decoded = None
        if decoded:
            raw, mime = decoded
            prepared_files.append((ensure_extension(filename, mime), raw, mime))

    return PreparedTurn(prompt=prompt, files=prepared_files)

    return PreparedTurn(prompt=prompt, files=prepared_files)

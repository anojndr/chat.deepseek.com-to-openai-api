"""OpenAI-compatible request/response translation models."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

DEEPTHINK_SUFFIX = "-deepthink"
MODEL_BASE = "deepseek-chat"
MODEL_ALIASES = {
    "deepseek-chat": None,
    "deepseek-v3": None,
    "deepseek-v3.2": None,
    "deepseek-reasoner": None,
    "deepseek-r1": None,
    "default": None,
    "expert": "expert",
    "vision": "vision",
}


class ModelSpec(BaseModel):
    base: str = MODEL_BASE
    deepthink: bool = False
    model_type: str | None = None
    requested: str = MODEL_BASE

    @property
    def wire_id(self) -> str:
        """Echo the caller's model id back (normalised case)."""
        return self.requested


# models whose DeepSeek semantics imply thinking by default
_THINK_BY_DEFAULT = {"deepseek-reasoner", "deepseek-r1"}


def parse_model(model: str | None) -> ModelSpec:
    raw = (model or MODEL_BASE).strip()
    spec = ModelSpec(requested=raw)
    lowered = raw.lower()

    # strip thinking suffixes case-insensitively ("-deepthink", "-think",
    # "-thinking" in any casing)
    stripped = re.sub(r"-(?:deep)?think(?:ing)?$", "", lowered)
    if stripped != lowered:
        spec.deepthink = True
    lowered = stripped

    alias = MODEL_ALIASES.get(lowered, "__unknown__")
    if alias == "__unknown__":
        # Unknown ids still route to the default model; echo the requested id.
        return spec
    spec.base = lowered
    spec.model_type = alias
    if lowered in _THINK_BY_DEFAULT:
        spec.deepthink = True
    return spec


# -- content part helpers ---------------------------------------------------


MAX_DATA_URL_BYTES = 25 * 1024 * 1024  # 25 MB decoded cap


def decode_data_url(url: str) -> tuple[bytes, str] | None:
    """Decode a data: URL, tolerating media-type params (charset etc.).

    Returns (bytes, mime) or None when the URL is malformed or oversized.
    """
    head, sep, payload = url.partition(",")
    if not sep or not head.startswith("data:"):
        return None
    meta = head[len("data:") :]
    parts = [s.strip().lower() for s in meta.split(";") if s.strip()]
    mime = parts[0] if parts and "/" in parts[0] else "text/plain"
    is_base64 = "base64" in parts[1:]
    if len(payload) > MAX_DATA_URL_BYTES * 2:  # rough pre-decode guard
        return None
    try:
        if is_base64:
            decoded = base64.b64decode(payload, validate=False)
        else:
            from urllib.parse import unquote_to_bytes

            decoded = unquote_to_bytes(payload)
    except (binascii.Error, ValueError):
        return None
    if len(decoded) > MAX_DATA_URL_BYTES:
        return None
    return decoded, mime


def guess_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    table = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
        "bmp": "image/bmp",
        "pdf": "application/pdf",
        "txt": "text/plain",
        "md": "text/markdown",
        "csv": "text/csv",
        "json": "application/json",
        "html": "text/html",
        "xml": "application/xml",
        "yaml": "text/yaml",
        "yml": "text/yaml",
        "py": "text/x-python",
        "js": "text/javascript",
        "ts": "text/typescript",
        "sh": "text/x-shellscript",
    }
    return table.get(ext, "application/octet-stream")


# -- Chat Completions -------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Any = None  # str | parts | None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = MODEL_BASE
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "ignore"}

    @property
    def include_usage(self) -> bool:
        return bool((self.stream_options or {}).get("include_usage"))


# -- Responses API ----------------------------------------------------------


class ResponsesRequest(BaseModel):
    model: str | None = MODEL_BASE
    input: Any = None  # str | list of items
    instructions: str | None = None
    previous_response_id: str | None = None
    conversation: dict[str, Any] | str | None = None
    stream: bool = False
    store: bool | None = None
    reasoning: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "ignore"}

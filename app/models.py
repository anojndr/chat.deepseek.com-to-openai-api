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


def parse_model(model: str | None) -> ModelSpec:
    raw = (model or MODEL_BASE).strip()
    spec = ModelSpec(requested=raw)
    if raw.endswith(DEEPTHINK_SUFFIX):
        spec.deepthink = True
        raw = raw[: -len(DEEPTHINK_SUFFIX)]
    elif raw.endswith("-thinking") or raw.endswith("-think"):
        spec.deepthink = True
        raw = re.sub(r"-(?:deep)?think(?:ing)?$", "", raw)
    alias = MODEL_ALIASES.get(raw.lower(), "__unknown__")
    spec.base = MODEL_BASE if alias == "__unknown__" else raw.lower()
    if alias != "__unknown__":
        spec.model_type = alias
    return spec


# -- content part helpers ---------------------------------------------------


def decode_data_url(url: str) -> tuple[bytes, str] | None:
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
    if not match:
        return None
    mime = match.group(1) or "text/plain"
    payload = match.group(3)
    if match.group(2):  # base64 flag
        try:
            return base64.b64decode(payload, validate=False), mime
        except (binascii.Error, ValueError):
            return None
    from urllib.parse import unquote_to_bytes

    return unquote_to_bytes(payload), mime


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

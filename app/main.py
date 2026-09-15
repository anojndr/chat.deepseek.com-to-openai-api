# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""FastAPI application: OpenAI-compatible Chat Completions + Responses proxy."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .accounts import AccountPool
from .citations import source_appendix
from .conversations import ConversationManager, StreamEvent, TurnResult
from .deepseek import (
    _BIZ_CODE_GLOBAL_MUTED,
    _BIZ_CODE_MUTED,
    _NON_RETRYABLE_COMPLETION_CODES,
    DeepSeekError,
)
from .models import ChatCompletionRequest, ModelSpec, ResponsesRequest, parse_model
from .pow_solver import PowSolver
from .storage import ConvRef, Storage
from .turn import PreparedTurn, compute_history_hashes, prepare_turn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    _BaseResponseFn = Callable[..., dict[str, Any]]

# HTTP status codes returned by this proxy.
_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_HTTP_UNPROCESSABLE = 422
_HTTP_RATE_LIMITED = 429
_HTTP_BAD_GATEWAY = 502

# Upstream status range passed through unchanged.
_HTTP_ERROR_MIN = 400
_HTTP_ERROR_MAX = 600

# Upstream business codes mapped to dedicated statuses.
_AUTH_FAILURE_BIZ_CODES: frozenset[int] = frozenset({40002, 40003})
_RATE_LIMIT_BIZ_CODE = 40029
# Single source of truth lives in app.deepseek; these aliases keep the
# HTTP mapping readable next to the status table.
_MUTED_BIZ_CODES: frozenset[int] = frozenset({_BIZ_CODE_MUTED, _BIZ_CODE_GLOBAL_MUTED})
_UNPROCESSABLE_BIZ_CODES: frozenset[int] = _NON_RETRYABLE_COMPLETION_CODES

# Truthy tokens for the include-sources flags.
_TRUTHY_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})

# Chat roles accepted verbatim in Responses input items.
_CHAT_ROLES: frozenset[str] = frozenset({"user", "assistant", "system", "developer"})

# Error type identifiers in OpenAI-style error payloads.
_ERROR_UPSTREAM = "upstream_error"
_ERROR_INVALID_REQUEST = "invalid_request_error"
_ERROR_INVALID_API_KEY = "invalid_api_key"
_ERROR_RATE_LIMIT = "rate_limit_exceeded"
_ERROR_MODERATION = "moderation_blocked"


class MissingApiKeyError(Exception):
    """Raised when a request lacks the required API key."""

    def __init__(self) -> None:
        """Initialize with the default invalid-key message."""
        super().__init__("invalid or missing API key")


class StartupError(RuntimeError):
    """Raised when the shared manager is used before startup."""

    def __init__(self) -> None:
        """Initialize with the default not-ready message."""
        super().__init__("startup hook has not run")


ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_PATH = Path(
    "/home/sweetpotet/Desktop/chat.deepseek.com-to-openai-api/accounts.txt",
)
if not ACCOUNTS_PATH.exists():
    ACCOUNTS_PATH = ROOT / "accounts.txt"
DB_PATH = ROOT / "data.sqlite"

_pool = AccountPool(ACCOUNTS_PATH)
# Legacy aliases retained for backward compatibility with direct assignment
# (e.g. older tests); new code goes through _runtime, prefer test hooks.
_solver: PowSolver | None = None
_storage = Storage(DB_PATH)
_manager: ConversationManager | None = None


class _RuntimeState:
    """Mutable holder for the lazily created solver/manager singletons."""

    def __init__(self) -> None:
        """Seed the holder from the import-time pool and storage."""
        self.pool: AccountPool = _pool
        self.storage: Storage = _storage
        self.solver: PowSolver | None = None
        self.manager: ConversationManager | None = None


_runtime = _RuntimeState()
_IMPORT_POOL = _pool
_IMPORT_STORAGE = _storage
_IMPORT_MANAGER: ConversationManager | None = _manager
_IMPORT_SOLVER: PowSolver | None = _solver


def _active_pool() -> AccountPool:
    """Return the effective account pool, honoring legacy patches.

    Returns:
        The patched pool when tests assign ``main_mod._pool`` directly,
        otherwise the runtime holder pool.

    """
    if _pool is not _IMPORT_POOL:
        return _pool
    return _runtime.pool


def _active_storage() -> Storage:
    """Return the effective storage, honoring legacy patches.

    Returns:
        The patched storage when tests assign ``main_mod._storage``
        directly, otherwise the runtime holder storage.

    """
    if _storage is not _IMPORT_STORAGE:
        return _storage
    return _runtime.storage


def _active_solver() -> PowSolver | None:
    """Return the effective solver, honoring legacy patches.

    Returns:
        The patched or runtime solver, if any.

    """
    if _solver is not _IMPORT_SOLVER:
        return _solver
    return _runtime.solver


def _active_manager_raw() -> ConversationManager | None:
    """Return the effective manager without raising, honoring patches.

    Returns:
        The patched manager when tests assign ``main_mod._manager``
        directly, otherwise the runtime holder manager.

    """
    if _manager is not _IMPORT_MANAGER:
        return _manager
    return _runtime.manager if _runtime.manager is not None else _manager


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prepare shared singletons on startup and close them on shutdown.

    Args:
        _app: Application instance (unused; required by FastAPI).

    Yields:
        Control to the running application.

    """
    _ensure_runtime()
    try:
        yield
    finally:
        await manager().aclose()


app = FastAPI(
    title="DeepSeek OpenAI-Compatible API",
    version="1.0.0",
    lifespan=lifespan,
)


def _ensure_runtime() -> None:
    """Create the shared solver and manager once, honoring test overrides."""
    solver = _active_solver()
    if solver is None:
        solver = PowSolver()
        _runtime.solver = solver
    existing = _active_manager_raw()
    if existing is None:
        _runtime.manager = ConversationManager(
            _active_pool(),
            solver,
            storage=_active_storage(),
        )
    else:
        existing.ensure_sweeper()


def manager() -> ConversationManager:
    """Return the shared conversation manager.

    Returns:
        The manager created during startup (or injected by tests).

    Raises:
        StartupError: If startup has not run and no manager was injected.

    """
    active = _active_manager_raw()
    if active is None:
        raise StartupError
    return active


def hook_set_state(
    storage: Storage | None = None,
    pool: AccountPool | None = None,
    solver: PowSolver | None = None,
    manager: ConversationManager | None = None,
) -> None:
    """Replace runtime state pieces for tests.

    Only pieces passed as non-None are replaced; the rest is left unchanged.

    Args:
        storage: Replacement storage, if any.
        pool: Replacement account pool, if any.
        solver: Replacement PoW solver, if any.
        manager: Replacement conversation manager, if any.

    """
    if storage is not None:
        _runtime.storage = storage
    if pool is not None:
        _runtime.pool = pool
    if solver is not None:
        _runtime.solver = solver
    if manager is not None:
        _runtime.manager = manager


def test_hook_clear_response_links() -> None:
    """Clear the in-memory response-link map for tests."""
    _response_links.clear()


def test_hook_store_response_link(
    response_id: str,
    conversation_key: str,
    model: str,
) -> None:
    """Store a response link for tests.

    Args:
        response_id: Response identifier to store.
        conversation_key: Conversation key the response belongs to.
        model: Model identifier recorded with the link.

    """
    _store_response_link(response_id, conversation_key, model)


def test_hook_get_response_links() -> dict[str, dict[str, str]]:
    """Return a snapshot of the in-memory response links for tests.

    Returns:
        Mapping of response id to conversation/model pairs.

    """
    return {
        response_id: {
            "conversation": str(link["conversation"]),
            "model": str(link["model"]),
        }
        for response_id, link in _response_links.items()
    }


API_KEY: str | None = os.environ.get("API_KEY")
_INCLUDE_SOURCES_RAW = os.environ.get("DEEPSEEK_INCLUDE_SOURCES")
if _INCLUDE_SOURCES_RAW is None:
    _INCLUDE_SOURCES_RAW = os.environ.get("INCLUDE_SOURCES", "0")
INCLUDE_SOURCES: bool = _INCLUDE_SOURCES_RAW.strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _include_sources(flag: object) -> bool:
    """Resolve the per-request sources flag against the server default.

    Args:
        flag: Per-request override; None falls back to server config.

    Returns:
        True when sources should be appended.

    """
    if flag is None:
        return INCLUDE_SOURCES
    if isinstance(flag, str):
        return flag.strip().lower() in _TRUTHY_TOKENS
    return bool(flag)


def should_include_sources(flag: object) -> bool:
    """Decide whether to append source citations for a request.

    Args:
        flag: Per-request override; None falls back to server config.

    Returns:
        True when sources should be appended.

    """
    return _include_sources(flag)


def require_api_key(request: Request) -> None:
    """Enforce the bearer key gate when the API_KEY env var is set.

    Args:
        request: Incoming request carrying the credentials.

    Raises:
        MissingApiKeyError: If a key is configured but missing or wrong.

    """
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    provided = (
        header[7:].strip()
        if header.startswith("Bearer ")
        else request.headers.get("x-api-key", "")
    )
    if provided != API_KEY:
        # Must raise: returning a response from a dependency does not
        # short-circuit the route in FastAPI.
        raise MissingApiKeyError


@app.exception_handler(MissingApiKeyError)
def _missing_api_key_handler(
    _request: Request,
    exc: MissingApiKeyError,
) -> JSONResponse:
    """Render invalid-key failures as an OpenAI-style 401 payload.

    Args:
        _request: Incoming request (unused; required by Starlette).
        exc: Raised key error carrying the message.

    Returns:
        JSON error response with status 401.

    """
    return JSONResponse(
        status_code=_HTTP_UNAUTHORIZED,
        content={
            "error": {
                "message": str(exc),
                "type": _ERROR_INVALID_API_KEY,
                "code": _ERROR_INVALID_API_KEY,
            },
        },
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _error(
    message: str,
    status: int = _HTTP_BAD_GATEWAY,
    err_type: str = _ERROR_UPSTREAM,
) -> JSONResponse:
    """Build an OpenAI-style error response.

    Args:
        message: Human-readable failure description.
        status: HTTP status code.
        err_type: OpenAI-style error type identifier.

    Returns:
        JSON error response with the given status.

    """
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": None}},
    )


def _http_error(exc: DeepSeekError) -> JSONResponse:
    """Map an upstream DeepSeek failure to an OpenAI-style response.

    Args:
        exc: Upstream error carrying status and business codes.

    Returns:
        JSON error response with the mapped status.

    """
    status = exc.status or _HTTP_BAD_GATEWAY
    if exc.biz_code in _AUTH_FAILURE_BIZ_CODES:
        mapped = _HTTP_UNAUTHORIZED
        etype = _ERROR_INVALID_API_KEY
    elif exc.biz_code == _RATE_LIMIT_BIZ_CODE:
        mapped = _HTTP_RATE_LIMITED
        etype = _ERROR_RATE_LIMIT
    elif exc.biz_code in _MUTED_BIZ_CODES:
        mapped = _HTTP_FORBIDDEN
        etype = _ERROR_MODERATION
    elif exc.biz_code in _UNPROCESSABLE_BIZ_CODES:
        mapped = _HTTP_UNPROCESSABLE
        etype = _ERROR_INVALID_REQUEST
    else:
        if _HTTP_ERROR_MIN <= status < _HTTP_ERROR_MAX:
            mapped = status
        else:
            mapped = _HTTP_BAD_GATEWAY
        etype = _ERROR_UPSTREAM
    return _error(str(exc), mapped, etype)


def _sse(data: dict[str, Any]) -> str:
    """Encode a payload as one SSE data frame.

    Args:
        data: Payload to serialize.

    Returns:
        SSE-encoded frame string.

    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _conversation_key(req: Request, body_user: str | None) -> str:
    """Derive the stable per-client conversation key for a request.

    Args:
        req: Incoming request (session headers take precedence).
        body_user: Optional user identifier from the request body.

    Returns:
        Header session id, or an auto key scoped to client and user.

    """
    header_id = req.headers.get("x-session-id") or req.headers.get("x-conversation-id")
    if header_id:
        return header_id
    return f"auto:{req.client.host if req.client else 'anon'}:{body_user or 'default'}"


def _has_custom_session(request: Request) -> bool:
    """Check whether the request pins a conversation via session headers.

    Args:
        request: Incoming request to inspect.

    Returns:
        True when a session or conversation id header is present.

    """
    return bool(
        request.headers.get("x-session-id") or request.headers.get("x-conversation-id"),
    )


def _estimate(text: str) -> int:
    """Roughly estimate token usage as one token per four characters.

    DeepSeek's accumulated_token_usage is not attributable per request.

    Args:
        text: Text to estimate.

    Returns:
        Token estimate of at least one.

    """
    return max(1, len(text) // 4)


def _sources_appendix(
    prompt: str,
    sources: list[Any],
    queries: list[str],
    *,
    include: bool,
) -> str:
    """Build the sources appendix for a turn when enabled.

    Args:
        prompt: Original prompt, used as the fallback appendix query.
        sources: Collected source entries.
        queries: Search queries issued during the turn.
        include: Whether appending sources was requested.

    Returns:
        Appendix text, or an empty string when disabled or sourceless.

    """
    if not include or not sources:
        return ""
    query = queries[0] if queries else prompt
    return source_appendix(sources, query)


def _responses_output_items(
    reasoning: str,
    content: str,
    msg_item_id: str,
) -> list[dict[str, Any]]:
    """Build the output item list for a completed Responses turn.

    Args:
        reasoning: Accumulated reasoning text, possibly empty.
        content: Final answer text.
        msg_item_id: Identifier for the message item.

    Returns:
        Reasoning item (when non-empty) followed by the message item.

    """
    items: list[dict[str, Any]] = []
    if reasoning:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex}",
                "summary": [{"type": "summary_text", "text": reasoning}],
            },
        )
    items.append(
        {
            "type": "message",
            "id": msg_item_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        },
    )
    return items


async def _resolve_conversation(hashes: list[str]) -> tuple[str, bool]:
    """Fork the longest-prefix match into a fresh key, or start new.

    Persists the inherited checkpoint immediately so mid-flight rows stay
    visible before the first upstream event arrives.

    Args:
        hashes: History hashes identifying the requested prefix.

    Returns:
        Tuple of the conversation key and whether this is a first turn.

    """
    match = _active_storage().find_prefix(hashes)
    matched_len = match[0] if match else 0
    ref: ConvRef | None = match[1] if match else None
    if ref is not None and matched_len >= len(hashes):
        # Exact duplicate request: re-match at matched_len - 1.
        matched_len = len(hashes) - 1
        rematch = (
            _active_storage().find_prefix(hashes[:matched_len])
            if matched_len > 0
            else None
        )
        ref = rematch[1] if rematch else None
    if ref is None:
        # Brand new conversation.
        key = f"auto:{uuid.uuid4().hex}"
        await manager().get_or_create(key)
        return key, True
    # Fork into a distinct instance referencing the parent checkpoint; only
    # continue incrementally when the prefix reaches the parent message.
    is_immediate_parent = matched_len == len(hashes) - 1
    key = f"auto:{uuid.uuid4().hex}"
    conv = await manager().get_or_create(key)
    conv.account_index = ref.account_index
    conv.account_token = ref.account_token
    conv.deepseek_session_id = ref.deepseek_session_id
    conv.parent_message_id = ref.parent_message_id
    # Persist the inherited checkpoint now: until the first upstream event
    # the row otherwise looks brand-new, hiding mid-flight state.
    manager().persist_conversation(conv)
    return key, not is_immediate_parent


async def _chat_turn_key(
    request: Request,
    user: str | None,
    hashes: list[str],
) -> tuple[str, bool]:
    """Resolve the conversation key and first-turn flag for a chat turn.

    Args:
        request: Incoming request carrying session headers and identity.
        user: Optional user identifier from the request body.
        hashes: History hashes identifying the requested prefix.

    Returns:
        Tuple of the conversation key and whether this is a first turn.

    """
    key = _conversation_key(request, user)
    if _has_custom_session(request):
        conv = await manager().get_or_create(key)
        return key, conv.deepseek_session_id is None
    return await _resolve_conversation(hashes)


def _resolve_previous_key(
    response_id: str | None,
    fallback: str,
) -> str | JSONResponse:
    """Resolve the conversation key for a previous response id.

    Args:
        response_id: Previous response identifier, if any.
        fallback: Key to use when no previous response was given.

    Returns:
        Conversation key, or an error response when the id is unknown.

    """
    if response_id is None:
        return fallback
    stored = _response_links.get(response_id)
    if stored is None:
        return _error(
            f"previous_response_id {response_id} not found",
            _HTTP_NOT_FOUND,
            _ERROR_INVALID_REQUEST,
        )
    return str(stored["conversation"])


# ---------------------------------------------------------------------------
# basic endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, Any]:
    """Describe the proxy service and its endpoints.

    Returns:
        Service status, endpoint list, and supported models.

    """
    return {
        "status": "ok",
        "service": "chat.deepseek.com → OpenAI-compatible API",
        "endpoints": ["/v1/chat/completions", "/v1/responses", "/v1/models", "/health"],
        "models": [
            "deepseek-chat",
            "deepseek-chat-deepthink",
            "deepseek-reasoner",
            "deepseek-reasoner-deepthink",
            "deepseek-vision",
            "deepseek-vision-deepthink",
        ],
    }


@app.get("/health")
@app.get("/v1/health")
async def health() -> dict[str, Any]:
    """Report process health and account pool status.

    Returns:
        Health status plus an account pool snapshot.

    """
    return {"status": "ok", "accounts": _active_pool().snapshot()}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List the OpenAI-style model entries served by this proxy.

    Returns:
        Model list payload in OpenAI list format.

    """
    now = int(time.time())
    models = [
        {
            "id": "deepseek-chat",
            "object": "model",
            "created": now,
            "owned_by": "deepseek-proxy",
        },
        {
            "id": "deepseek-chat-deepthink",
            "object": "model",
            "created": now,
            "owned_by": "deepseek-proxy",
        },
        {
            "id": "deepseek-reasoner",
            "object": "model",
            "created": now,
            "owned_by": "deepseek-proxy",
        },
        {
            "id": "deepseek-reasoner-deepthink",
            "object": "model",
            "created": now,
            "owned_by": "deepseek-proxy",
        },
        {
            "id": "deepseek-vision",
            "object": "model",
            "created": now,
            "owned_by": "deepseek-proxy",
        },
        {
            "id": "deepseek-vision-deepthink",
            "object": "model",
            "created": now,
            "owned_by": "deepseek-proxy",
        },
    ]
    return {"object": "list", "data": models}


@app.post("/accounts/reload", dependencies=[Depends(require_api_key)])
async def reload_accounts() -> dict[str, Any]:
    """Reload account tokens and drop clients for removed accounts.

    Returns:
        Reload counts plus a fresh account pool snapshot.

    """
    count = _active_pool().reload()
    removed = await manager().invalidate_clients(set(_active_pool().tokens()))
    return {
        "status": "reloaded",
        "accounts": count,
        "stale_clients_closed": removed,
        "detail": _active_pool().snapshot(),
    }


# ---------------------------------------------------------------------------
# Chat Completions API
# ---------------------------------------------------------------------------


@dataclass
class _ChatStreamContext:
    """Parameters for streaming one Chat Completions turn."""

    key: str
    prepared: PreparedTurn
    spec: ModelSpec
    completion_id: str
    created: int
    model_id: str
    include_usage: bool
    include_sources: bool = False
    hashes: list[str] | None = None


@dataclass
class _ChatStreamState:
    """Mutable accumulation state for one chat stream."""

    prompt: str
    completion_est: int = 0
    search_queries: list[str] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)


def _chat_chunk(
    stream: _ChatStreamContext,
    delta: dict[str, Any],
    finish: str | None = None,
) -> str:
    """Format one chat completion chunk as an SSE payload.

    Args:
        stream: Stream parameters carrying ids and model info.
        delta: Content delta for the single choice.
        finish: Optional finish reason for the final chunk.

    Returns:
        SSE-encoded chunk string.

    """
    return _sse(
        {
            "id": stream.completion_id,
            "object": "chat.completion.chunk",
            "created": stream.created,
            "model": stream.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish,
                },
            ],
        },
    )


def _apply_chat_event(
    stream: _ChatStreamContext,
    state: _ChatStreamState,
    event: StreamEvent,
) -> list[str]:
    """Translate one turn event into chat SSE chunks, updating state.

    Args:
        stream: Stream parameters carrying ids and model info.
        state: Mutable accumulation state to update.
        event: Turn event emitted by the conversation manager.

    Returns:
        Chunk strings to emit for the event, possibly empty.

    """
    if event.kind == "reasoning":
        return [_chat_chunk(stream, {"reasoning_content": str(event.value)})]
    if event.kind == "content":
        text = str(event.value)
        state.completion_est += _estimate(text)
        return [_chat_chunk(stream, {"content": text})]
    if event.kind == "search" and isinstance(event.value, list):
        state.search_queries.extend(str(query) for query in event.value if query)
        return []
    if event.kind == "sources" and isinstance(event.value, list):
        state.sources = list(event.value)
        return []
    return []


def _finish_chat_stream(
    stream: _ChatStreamContext,
    state: _ChatStreamState,
) -> list[str]:
    """Emit the closing chunks for a chat stream.

    Args:
        stream: Stream parameters carrying ids and model info.
        state: Accumulated stream state.

    Returns:
        Closing chunk strings ending with the done sentinel.

    """
    pieces: list[str] = []
    if stream.include_sources and state.sources:
        appendix = _sources_appendix(
            state.prompt,
            state.sources,
            state.search_queries,
            include=True,
        )
        if appendix:
            state.completion_est += _estimate(appendix)
            pieces.append(_chat_chunk(stream, {"content": appendix}))
    prompt_est = _estimate(state.prompt)
    pieces.append(_chat_chunk(stream, {}, finish="stop"))
    if stream.include_usage:
        pieces.append(
            _sse(
                {
                    "id": stream.completion_id,
                    "object": "chat.completion.chunk",
                    "created": stream.created,
                    "model": stream.model_id,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_est,
                        "completion_tokens": state.completion_est,
                        "total_tokens": prompt_est + state.completion_est,
                    },
                },
            ),
        )
    pieces.append("data: [DONE]\n\n")
    return pieces


def _finalize_chat(
    stream: _ChatStreamContext,
    result: TurnResult,
) -> JSONResponse:
    """Build the completed chat payload with usage estimates.

    Args:
        stream: Stream parameters carrying ids and model info.
        result: Completed turn result.

    Returns:
        Completed chat completion response.

    """
    content = result.content
    content += _sources_appendix(
        stream.prepared.prompt,
        result.sources,
        result.search_queries,
        include=stream.include_sources,
    )
    prompt_est = _estimate(stream.prepared.prompt)
    completion_est = _estimate(content)
    return JSONResponse(
        {
            "id": stream.completion_id,
            "object": "chat.completion",
            "created": stream.created,
            "model": stream.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": prompt_est,
                "completion_tokens": completion_est,
                "total_tokens": prompt_est + completion_est,
            },
        },
    )


@app.post(
    "/v1/chat/completions",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    """Accept a Chat Completions request and proxy it to DeepSeek.

    Args:
        request: Incoming OpenAI-style chat request.

    Returns:
        Streaming SSE response or the completed chat payload.

    """
    try:
        body = ChatCompletionRequest(**(await request.json()))
    except (TypeError, ValueError) as exc:
        return _error(
            f"invalid request body: {exc}",
            _HTTP_BAD_REQUEST,
            _ERROR_INVALID_REQUEST,
        )
    if not body.messages:
        return _error(
            "messages must not be empty",
            _HTTP_BAD_REQUEST,
            _ERROR_INVALID_REQUEST,
        )

    spec = parse_model(body.model)
    raw_messages = [m.model_dump(exclude_none=True) for m in body.messages]
    hashes, _system_text = compute_history_hashes(raw_messages)
    key, is_first = await _chat_turn_key(request, body.user, hashes)

    prepared = prepare_turn(raw_messages, is_first_turn=is_first)
    if not prepared.prompt:
        return _error(
            "no usable prompt in messages",
            _HTTP_BAD_REQUEST,
            _ERROR_INVALID_REQUEST,
        )

    context = _ChatStreamContext(
        key=key,
        prepared=prepared,
        spec=spec,
        completion_id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
        created=int(time.time()),
        model_id=spec.wire_id,
        include_usage=body.include_usage,
        include_sources=_include_sources(body.include_sources),
        hashes=hashes,
    )
    if body.stream:
        return StreamingResponse(
            _chat_stream(context),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await manager().run_turn(
            context.key,
            context.prepared,
            deepthink=context.spec.deepthink,
            model_type=context.spec.model_type,
            hashes=context.hashes,
        )
    except DeepSeekError as exc:
        return _http_error(exc)
    return _finalize_chat(context, result)


async def _chat_stream(stream: _ChatStreamContext) -> AsyncIterator[str]:
    """Stream chat completion chunks for one turn.

    Args:
        stream: Stream parameters for the turn.

    Yields:
        SSE-encoded chunk strings.

    """
    yield _chat_chunk(stream, {"role": "assistant", "content": ""})
    state = _ChatStreamState(prompt=stream.prepared.prompt)
    try:
        async for event in manager().stream_turn(
            stream.key,
            stream.prepared,
            deepthink=stream.spec.deepthink,
            model_type=stream.spec.model_type,
            hashes=stream.hashes,
        ):
            for piece in _apply_chat_event(stream, state, event):
                yield piece
    except DeepSeekError as exc:
        # OpenAI SDKs raise when an SSE payload carries a top-level "error";
        # emit that (no finish_reason, no [DONE]) so truncation is visible.
        yield _sse(
            {
                "error": {
                    "message": str(exc),
                    "type": _ERROR_UPSTREAM,
                    "code": exc.biz_code,
                },
            },
        )
        return
    for piece in _finish_chat_stream(stream, state):
        yield piece


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


@dataclass
class _ResponsesStreamContext:
    """Parameters for streaming one Responses API turn."""

    key: str
    prepared: PreparedTurn
    spec: ModelSpec
    response_id: str
    msg_item_id: str
    base_response: _BaseResponseFn
    include_sources: bool = False
    hashes: list[str] | None = None


@dataclass
class _ResponsesStreamState:
    """Mutable accumulation state for one responses stream."""

    prompt: str
    full_text: list[str] = field(default_factory=list)
    reasoning_text: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)


def _response_event(name: str, payload: dict[str, Any]) -> str:
    """Format one Responses API event as an SSE payload.

    Args:
        name: Event type name.
        payload: Event body fields.

    Returns:
        SSE-encoded event string.

    """
    body = {"type": name, **payload}
    return f"event: {name}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


def _apply_responses_event(
    stream: _ResponsesStreamContext,
    state: _ResponsesStreamState,
    event: StreamEvent,
) -> list[str]:
    """Translate one turn event into Responses SSE events, updating state.

    Args:
        stream: Stream parameters carrying ids and envelope builder.
        state: Mutable accumulation state to update.
        event: Turn event emitted by the conversation manager.

    Returns:
        Event strings to emit for the event, possibly empty.

    """
    if event.kind == "reasoning":
        state.reasoning_text.append(str(event.value))
        return [
            _response_event(
                "response.reasoning_text.delta",
                {
                    "item_id": stream.msg_item_id,
                    "output_index": 0,
                    "delta": str(event.value),
                },
            ),
        ]
    if event.kind == "content":
        state.full_text.append(str(event.value))
        return [
            _response_event(
                "response.output_text.delta",
                {
                    "item_id": stream.msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": str(event.value),
                },
            ),
        ]
    if event.kind == "search" and isinstance(event.value, list):
        state.search_queries.extend(str(query) for query in event.value if query)
        return []
    if event.kind == "sources" and isinstance(event.value, list):
        state.sources = list(event.value)
        return []
    return []


def _finish_responses_stream(
    stream: _ResponsesStreamContext,
    state: _ResponsesStreamState,
) -> list[str]:
    """Emit the closing events for a Responses stream.

    Args:
        stream: Stream parameters carrying ids and envelope builder.
        state: Accumulated stream state.

    Returns:
        Closing event strings ending with response.completed.

    """
    pieces: list[str] = []
    if stream.include_sources and state.sources:
        appendix = _sources_appendix(
            state.prompt,
            state.sources,
            state.search_queries,
            include=True,
        )
        if appendix:
            state.full_text.append(appendix)
            pieces.append(
                _response_event(
                    "response.output_text.delta",
                    {
                        "item_id": stream.msg_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": appendix,
                    },
                ),
            )
    text = "".join(state.full_text)
    pieces.extend(
        [
            _response_event(
                "response.output_text.done",
                {
                    "item_id": stream.msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
            ),
            _response_event(
                "response.content_part.done",
                {
                    "item_id": stream.msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                },
            ),
            _response_event(
                "response.output_item.done",
                {
                    "output_index": 0,
                    "item": {
                        "id": stream.msg_item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []},
                        ],
                    },
                },
            ),
        ],
    )
    reasoning = "".join(state.reasoning_text)
    final = stream.base_response("completed")
    final["output"] = _responses_output_items(reasoning, text, stream.msg_item_id)
    final["completed_at"] = int(time.time())
    final["usage"] = {
        "input_tokens": _estimate(state.prompt),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": _estimate(text),
        "output_tokens_details": {"reasoning_tokens": _estimate(reasoning)},
        "total_tokens": _estimate(state.prompt) + _estimate(text),
    }
    _store_response_link(stream.response_id, stream.key, stream.spec.wire_id)
    pieces.append(_response_event("response.completed", {"response": final}))
    return pieces


async def _complete_responses_turn(
    stream: _ResponsesStreamContext,
) -> JSONResponse:
    """Run one non-streaming Responses turn to completion.

    Args:
        stream: Shared turn parameters and envelope builder.

    Returns:
        Completed response payload.

    """
    try:
        result = await manager().run_turn(
            stream.key,
            stream.prepared,
            deepthink=stream.spec.deepthink,
            model_type=stream.spec.model_type,
            hashes=stream.hashes,
        )
    except DeepSeekError as exc:
        return _http_error(exc)
    content = result.content
    content += _sources_appendix(
        stream.prepared.prompt,
        result.sources,
        result.search_queries,
        include=stream.include_sources,
    )
    final = stream.base_response(
        "completed",
        _responses_output_items(result.reasoning or "", content, stream.msg_item_id),
    )
    final["completed_at"] = int(time.time())
    final["usage"] = {
        "input_tokens": _estimate(stream.prepared.prompt),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": _estimate(content),
        "output_tokens_details": {
            "reasoning_tokens": _estimate(result.reasoning or ""),
        },
        "total_tokens": _estimate(stream.prepared.prompt) + _estimate(content),
    }
    _store_response_link(stream.response_id, stream.key, stream.spec.wire_id)
    return JSONResponse(final)


@app.post("/v1/responses", response_model=None, dependencies=[Depends(require_api_key)])
async def responses_api(request: Request) -> StreamingResponse | JSONResponse:
    """Accept a Responses request and proxy it to DeepSeek.

    Args:
        request: Incoming OpenAI-style responses request.

    Returns:
        Streaming SSE response or the completed response payload.

    """
    try:
        body = ResponsesRequest(**(await request.json()))
    except (TypeError, ValueError) as exc:
        return _error(
            f"invalid request body: {exc}",
            _HTTP_BAD_REQUEST,
            _ERROR_INVALID_REQUEST,
        )

    spec = parse_model(body.model)
    key = _conversation_key(request, None)
    key_or_error = _resolve_previous_key(body.previous_response_id, key)
    if isinstance(key_or_error, JSONResponse):
        return key_or_error
    key = key_or_error

    items = _normalize_responses_input(body.input)
    if body.instructions:
        items.insert(0, {"role": "system", "content": body.instructions})

    hashes, _system_text = compute_history_hashes(
        items,
        instructions=body.instructions,
    )
    if body.previous_response_id or _has_custom_session(request):
        conv = await manager().get_or_create(key)
        is_first = conv.deepseek_session_id is None
    else:
        key, is_first = await _resolve_conversation(hashes)

    prepared = prepare_turn(items, is_first_turn=is_first)
    if not prepared.prompt:
        return _error(
            "no usable input",
            _HTTP_BAD_REQUEST,
            _ERROR_INVALID_REQUEST,
        )

    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())

    def base_response(
        status: str = "in_progress",
        output: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the response envelope snapshot for this turn.

        Args:
            status: Lifecycle status label.
            output: Output items to embed.

        Returns:
            Response envelope dict.

        """
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": status,
            "error": None,
            "incomplete_details": None,
            "instructions": body.instructions,
            "max_output_tokens": body.max_output_tokens,
            "model": spec.wire_id,
            "output": output or [],
            "parallel_tool_calls": True,
            "previous_response_id": body.previous_response_id,
            "reasoning": {
                "effort": "high" if spec.deepthink else None,
                "summary": None,
            },
            "store": False,
            "temperature": body.temperature,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": body.top_p,
            "truncation": "disabled",
            "usage": None,
            "user": None,
            "metadata": {},
        }

    context = _ResponsesStreamContext(
        key=key,
        prepared=prepared,
        spec=spec,
        response_id=response_id,
        msg_item_id=f"msg_{uuid.uuid4().hex}",
        base_response=base_response,
        include_sources=_include_sources(body.include_sources),
        hashes=hashes,
    )
    if body.stream:
        return StreamingResponse(
            _responses_stream(context),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await _complete_responses_turn(context)


async def _responses_stream(stream: _ResponsesStreamContext) -> AsyncIterator[str]:
    """Stream Responses API events for one turn.

    Args:
        stream: Stream parameters for the turn.

    Yields:
        SSE-encoded event strings.

    """
    snapshot = stream.base_response()
    yield _response_event("response.created", {"response": snapshot})
    yield _response_event("response.in_progress", {"response": snapshot})
    yield _response_event(
        "response.output_item.added",
        {
            "output_index": 0,
            "item": {
                "id": stream.msg_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield _response_event(
        "response.content_part.added",
        {
            "item_id": stream.msg_item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    state = _ResponsesStreamState(prompt=stream.prepared.prompt)
    try:
        async for event in manager().stream_turn(
            stream.key,
            stream.prepared,
            deepthink=stream.spec.deepthink,
            model_type=stream.spec.model_type,
            hashes=stream.hashes,
        ):
            for piece in _apply_responses_event(stream, state, event):
                yield piece
    except DeepSeekError as exc:
        failed = stream.base_response("failed")
        failed["error"] = {"code": _ERROR_UPSTREAM, "message": str(exc)}
        yield _response_event("response.failed", {"response": failed})
        return
    for piece in _finish_responses_stream(stream, state):
        yield piece


def _normalize_responses_input(value: object) -> list[dict[str, Any]]:
    """Accept string input or the documented item shapes; flatten to messages.

    Args:
        value: Raw input payload in any documented shape.

    Returns:
        Message dicts in chat shape.

    """
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    out: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else [value]:
        normalized = _normalize_responses_item(item)
        if normalized is not None:
            out.append(normalized)
    return out


def _normalize_responses_item(item: object) -> dict[Any, Any] | None:
    """Normalize one Responses input item to chat shape, or skip it.

    Args:
        item: Single input item in any documented shape.

    Returns:
        Message dict, or None when the item carries no usable content.

    """
    if isinstance(item, str):
        return {"role": "user", "content": item}
    if not isinstance(item, dict):
        return None
    if item.get("role") in _CHAT_ROLES:
        return item
    return _normalize_typed_item(item)


def _normalize_typed_item(item: dict[Any, Any]) -> dict[str, Any] | None:
    """Normalize a dict-shaped Responses input item without a chat role.

    Args:
        item: Input item carrying a type discriminator.

    Returns:
        Message dict, or None when the item carries no usable content.

    """
    itype = item.get("type")
    if itype == "message":
        return {**item, "role": item.get("role", "assistant")}
    if itype == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": item.get("output"),
        }
    if itype == "reasoning":
        summary = "".join(
            part.get("text", "")
            for part in (item.get("summary") or [])
            if isinstance(part, dict)
        )
        if summary:
            return {"role": "assistant", "content": f"[reasoning] {summary}"}
    # Function_call / tool echoes without output are skipped.
    return None


_RESPONSE_LINK_MAX = 10_000
_response_links: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _store_response_link(response_id: str, conversation_key: str, model: str) -> None:
    """LRU-bounded map so old ids age out one-by-one (never a mass wipe).

    Args:
        response_id: Response identifier to store.
        conversation_key: Conversation key the response belongs to.
        model: Model identifier recorded with the link.

    """
    _response_links[response_id] = {"conversation": conversation_key, "model": model}
    _response_links.move_to_end(response_id)
    while len(_response_links) > _RESPONSE_LINK_MAX:
        _response_links.popitem(last=False)
    _active_storage().store_response_link(
        response_id,
        conversation_key,
        model,
        limit=_RESPONSE_LINK_MAX,
    )


@app.get(
    "/v1/responses/{response_id}",
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
async def get_response(response_id: str) -> JSONResponse | dict[str, Any]:
    """Fetch a stored response envelope by id.

    Args:
        response_id: Response identifier from a prior turn.

    Returns:
        Error payload when unknown, otherwise the response envelope.

    """
    stored = _response_links.get(response_id)
    if not stored:
        stored = _active_storage().get_response_link(response_id)
        if stored:
            _response_links[response_id] = stored
    if not stored:
        return JSONResponse(
            status_code=_HTTP_NOT_FOUND,
            content={
                "error": {
                    "message": f"no response found with id '{response_id}'",
                    "type": _ERROR_INVALID_REQUEST,
                    "code": None,
                },
            },
        )
    transcript = manager().transcript(stored["conversation"])
    last_answer = next(
        (m["content"] for m in reversed(transcript) if m["role"] == "assistant"),
        "",
    )
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": stored["model"],
        "output": [
            {
                "type": "message",
                "id": f"msg_{response_id[5:]}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": last_answer, "annotations": []},
                ],
            },
        ],
    }


@app.delete("/v1/sessions/{key}", dependencies=[Depends(require_api_key)])
async def delete_session(key: str) -> dict[str, Any]:
    """Delete a stored conversation session.

    Args:
        key: Session key to delete.

    Returns:
        Deletion confirmation payload.

    """
    await manager().reset(key)
    return {"status": "deleted", "session": key}

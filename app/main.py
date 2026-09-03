"""FastAPI application: OpenAI-compatible Chat Completions + Responses proxy."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .accounts import AccountPool
from .conversations import ConversationManager
from .deepseek import DeepSeekError
from .citations import source_appendix
from .models import (
    MODEL_BASE,
    ChatCompletionRequest,
    ResponsesRequest,
    parse_model,
)
from .pow_solver import PowSolver
from .storage import Storage, ConvRef
from .turn import prepare_turn, compute_history_hashes

ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_PATH = Path(
    "/home/sweetpotet/Desktop/chat.deepseek.com-to-openai-api/accounts.txt"
)
if not ACCOUNTS_PATH.exists():
    ACCOUNTS_PATH = ROOT / "accounts.txt"
DB_PATH = ROOT / "data.sqlite"

_pool = AccountPool(ACCOUNTS_PATH)
_solver: PowSolver | None = None
_storage = Storage(DB_PATH)
_manager: ConversationManager | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _solver, _manager
    if _solver is None:
        _solver = PowSolver()
    if _manager is None:
        _manager = ConversationManager(_pool, _solver, storage=_storage)
    else:
        _manager.ensure_sweeper()
    yield
    if _manager is not None:
        await manager().aclose()


app = FastAPI(
    title="DeepSeek OpenAI-Compatible API", version="1.0.0", lifespan=lifespan
)


def manager() -> ConversationManager:
    assert _manager is not None, "startup hook has not run"
    return _manager


API_KEY: str | None = os.environ.get("API_KEY")
_INCLUDE_SOURCES_RAW = os.environ.get("DEEPSEEK_INCLUDE_SOURCES")
if _INCLUDE_SOURCES_RAW is None:
    _INCLUDE_SOURCES_RAW = os.environ.get("INCLUDE_SOURCES", "0")
INCLUDE_SOURCES: bool = _INCLUDE_SOURCES_RAW.strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _include_sources(flag: Any | None) -> bool:
    """Per-request override; falls back to the DEEPSEEK_INCLUDE_SOURCES config flag."""
    if flag is None:
        return INCLUDE_SOURCES
    if isinstance(flag, str):
        return flag.strip().lower() in ("1", "true", "yes", "on")
    return bool(flag)


class MissingApiKey(Exception):
    pass


async def require_api_key(request: Request) -> None:
    """Bearer-key gate, active only when the API_KEY env var is set."""
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    provided = (
        header[7:].strip()
        if header.startswith("Bearer ")
        else request.headers.get("x-api-key", "")
    )
    if provided != API_KEY:
        # must raise: returning a Response from a dependency does not
        # short-circuit the route in FastAPI
        raise MissingApiKey("invalid or missing API key")


@app.exception_handler(MissingApiKey)
async def _missing_api_key_handler(
    request: Request, exc: MissingApiKey
) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_api_key",
                "code": "invalid_api_key",
            }
        },
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _error(
    message: str, status: int = 502, err_type: str = "upstream_error"
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": None}},
    )


def _http_error(exc: DeepSeekError) -> JSONResponse:
    status = exc.status or 502
    if exc.biz_code in (40002, 40003):
        mapped = 401
        etype = "invalid_api_key"
    elif exc.biz_code == 40029:
        mapped = 429
        etype = "rate_limit_exceeded"
    else:
        mapped = status if 400 <= status < 600 else 502
        etype = "upstream_error"
    return _error(str(exc), mapped, etype)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _conversation_key(req: Request, body_user: str | None) -> str:
    """Stable per-client conversation: X-Session-Id header > user field."""
    header_id = req.headers.get("x-session-id") or req.headers.get("x-conversation-id")
    if header_id:
        return header_id
    return f"auto:{req.client.host if req.client else 'anon'}:{body_user or 'default'}"


# ---------------------------------------------------------------------------
# basic endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "chat.deepseek.com → OpenAI-compatible API",
        "endpoints": ["/v1/chat/completions", "/v1/responses", "/v1/models", "/health"],
        "models": ["deepseek-chat", "deepseek-chat-deepthink", "deepseek-reasoner"],
    }


@app.get("/health")
@app.get("/v1/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "accounts": _pool.snapshot()}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
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
    count = _pool.reload()
    removed = await manager().invalidate_clients(set(_pool.tokens()))
    return {
        "status": "reloaded",
        "accounts": count,
        "stale_clients_closed": removed,
        "detail": _pool.snapshot(),
    }


# ---------------------------------------------------------------------------
# Chat Completions API
# ---------------------------------------------------------------------------


@app.post(
    "/v1/chat/completions", response_model=None, dependencies=[Depends(require_api_key)]
)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    try:
        body = ChatCompletionRequest(**(await request.json()))
    except Exception as exc:
        return _error(f"invalid request body: {exc}", 400, "invalid_request_error")
    if not body.messages:
        return _error("messages must not be empty", 400, "invalid_request_error")

    spec = parse_model(body.model)
    key = _conversation_key(request, body.user)

    raw_messages = [m.model_dump(exclude_none=True) for m in body.messages]
    hashes, system_text = compute_history_hashes(raw_messages)

    has_custom_session_header = bool(
        request.headers.get("x-session-id") or request.headers.get("x-conversation-id")
    )

    if has_custom_session_header:
        conv = await manager().get_or_create(key)
        is_first = conv.deepseek_session_id is None
    else:
        # Match longest prefix
        match = _storage.find_prefix(hashes) if _storage is not None else None
        matched_len = match[0] if match else 0
        ref: ConvRef | None = match[1] if match else None

        if ref is not None and matched_len >= len(hashes):
            # Exact duplicate request: re-match at matched_len - 1
            matched_len = len(hashes) - 1
            rematch = (
                _storage.find_prefix(hashes[:matched_len])
                if (matched_len > 0 and _storage is not None)
                else None
            )
            ref = rematch[1] if rematch else None

        if ref is not None:
            # Fork into a distinct conversation instance referencing the parent checkpoint
            # Only continue incrementally if the prefix matches up to the immediate parent message
            is_immediate_parent = matched_len == len(hashes) - 1
            key = f"auto:{uuid.uuid4().hex}"
            conv = await manager().get_or_create(key)
            conv.account_index = ref.account_index
            conv.account_token = ref.account_token
            conv.deepseek_session_id = ref.deepseek_session_id
            conv.parent_message_id = ref.parent_message_id
            is_first = not is_immediate_parent
        else:
            # Brand new conversation
            key = f"auto:{uuid.uuid4().hex}"
            conv = await manager().get_or_create(key)
            is_first = True
    prepared = prepare_turn(raw_messages, is_first_turn=is_first)
    if not prepared.prompt:
        return _error("no usable prompt in messages", 400, "invalid_request_error")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    model_id = spec.wire_id
    req_include_sources = _include_sources(body.include_sources)

    if body.stream:
        return StreamingResponse(
            _chat_stream(
                key,
                prepared,
                spec,
                completion_id,
                created,
                model_id,
                body.include_usage,
                req_include_sources,
                hashes=hashes,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await manager().run_turn(
            key,
            prepared,
            deepthink=spec.deepthink,
            model_type=spec.model_type,
            hashes=hashes,
        )
    except DeepSeekError as exc:
        return _http_error(exc)

    final_content = result.content
    if req_include_sources and result.sources:
        appendix_query = (
            result.search_queries[0] if result.search_queries else prepared.prompt
        )
        appendix = source_appendix(result.sources, appendix_query)
        final_content = final_content + appendix

    message: dict[str, Any] = {"role": "assistant", "content": final_content}
    usage = {
        "prompt_tokens": _estimate(prepared.prompt),
        "completion_tokens": _estimate(final_content),
        "total_tokens": _estimate(prepared.prompt) + _estimate(final_content),
    }
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    )


async def _chat_stream(
    key: str,
    prepared,
    spec,
    completion_id: str,
    created: int,
    model_id: str,
    include_usage: bool,
    include_sources: bool = False,
    hashes: list[str] | None = None,
) -> AsyncIterator[str]:
    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "logprobs": None,
                        "finish_reason": finish,
                    }
                ],
            }
        )

    yield chunk({"role": "assistant", "content": ""})
    completion_est = 0
    search_queries: list[str] = []
    sources: list[dict[str, Any]] = []
    try:
        async for ev in manager().stream_turn(
            key,
            prepared,
            deepthink=spec.deepthink,
            model_type=spec.model_type,
            hashes=hashes,
        ):
            if ev.kind == "reasoning":
                yield chunk({"reasoning_content": str(ev.value)})
            elif ev.kind == "content":
                text = str(ev.value)
                completion_est += _estimate(text)
                yield chunk({"content": text})
            elif ev.kind == "search":
                if isinstance(ev.value, list):
                    search_queries.extend(str(q) for q in ev.value if q)
            elif ev.kind == "sources":
                if isinstance(ev.value, list):
                    sources = list(ev.value)
        if include_sources and sources:
            appendix_query = search_queries[0] if search_queries else prepared.prompt
            appendix = source_appendix(sources, appendix_query)
            if appendix:
                completion_est += _estimate(appendix)
                yield chunk({"content": appendix})
        prompt_est = _estimate(prepared.prompt)
        yield chunk({}, finish="stop")
        if include_usage:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_est,
                        "completion_tokens": completion_est,
                        "total_tokens": prompt_est + completion_est,
                    },
                }
            )
        yield "data: [DONE]\n\n"
    except DeepSeekError as exc:
        # OpenAI SDKs raise when an SSE payload carries a top-level "error";
        # emit that (no finish_reason, no [DONE]) so truncation is visible.
        yield _sse(
            {
                "error": {
                    "message": str(exc),
                    "type": "upstream_error",
                    "code": exc.biz_code,
                }
            }
        )


def _estimate(text: str) -> int:
    # rough char/4 estimate; DeepSeek's accumulated_token_usage isn't per-request
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------


@app.post("/v1/responses", response_model=None, dependencies=[Depends(require_api_key)])
async def responses_api(request: Request) -> StreamingResponse | JSONResponse:
    try:
        body = ResponsesRequest(**(await request.json()))
    except Exception as exc:
        return _error(f"invalid request body: {exc}", 400, "invalid_request_error")

    spec = parse_model(body.model)
    key = _conversation_key(request, None)
    if body.previous_response_id:
        stored = _response_links.get(body.previous_response_id)
        if stored is None:
            return _error(
                f"previous_response_id {body.previous_response_id} not found",
                404,
                "invalid_request_error",
            )
        key = stored["conversation"]

    items = _normalize_responses_input(body.input)
    if body.instructions:
        items.insert(0, {"role": "system", "content": body.instructions})

    hashes, system_text = compute_history_hashes(items, instructions=body.instructions)
    has_custom_session_header = bool(
        request.headers.get("x-session-id") or request.headers.get("x-conversation-id")
    )
    if body.previous_response_id or has_custom_session_header:
        conv = await manager().get_or_create(key)
        is_first = conv.deepseek_session_id is None
    else:
        # Match prefix
        match = _storage.find_prefix(hashes) if _storage is not None else None
        matched_len = match[0] if match else 0
        ref: ConvRef | None = match[1] if match else None

        if ref is not None and matched_len >= len(hashes):
            matched_len = len(hashes) - 1
            rematch = (
                _storage.find_prefix(hashes[:matched_len])
                if (matched_len > 0 and _storage is not None)
                else None
            )
            ref = rematch[1] if rematch else None

        if ref is not None:
            is_immediate_parent = matched_len == len(hashes) - 1
            key = f"auto:{uuid.uuid4().hex}"
            conv = await manager().get_or_create(key)
            conv.account_index = ref.account_index
            conv.account_token = ref.account_token
            conv.deepseek_session_id = ref.deepseek_session_id
            conv.parent_message_id = ref.parent_message_id
            is_first = not is_immediate_parent
        else:
            key = f"auto:{uuid.uuid4().hex}"
            conv = await manager().get_or_create(key)
            is_first = True

    prepared = prepare_turn(items, is_first_turn=is_first)
    if not prepared.prompt:
        return _error("no usable input", 400, "invalid_request_error")

    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    req_include_sources = _include_sources(body.include_sources)

    def base_response(
        status: str = "in_progress", output: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
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

    msg_item_id = f"msg_{uuid.uuid4().hex}"

    if body.stream:
        return StreamingResponse(
            _responses_stream(
                key,
                prepared,
                spec,
                response_id,
                msg_item_id,
                created,
                base_response,
                body.previous_response_id,
                req_include_sources,
                hashes=hashes,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await manager().run_turn(
            key,
            prepared,
            deepthink=spec.deepthink,
            model_type=spec.model_type,
            hashes=hashes,
        )
    except DeepSeekError as exc:
        return _http_error(exc)

    final_content = result.content
    if req_include_sources and result.sources:
        appendix_query = (
            result.search_queries[0] if result.search_queries else prepared.prompt
        )
        appendix = source_appendix(result.sources, appendix_query)
        final_content = final_content + appendix

    output_items: list[dict[str, Any]] = []
    if result.reasoning:
        output_items.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex}",
                "summary": [{"type": "summary_text", "text": result.reasoning}],
            }
        )
    output_items.append(
        {
            "type": "message",
            "id": msg_item_id,
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": final_content, "annotations": []}
            ],
        }
    )
    final = base_response("completed", output_items)
    final["completed_at"] = int(time.time())
    final["usage"] = {
        "input_tokens": _estimate(prepared.prompt),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": _estimate(final_content),
        "output_tokens_details": {
            "reasoning_tokens": _estimate(result.reasoning or "")
        },
        "total_tokens": _estimate(prepared.prompt) + _estimate(final_content),
    }
    _store_response_link(response_id, key, spec.wire_id)
    return JSONResponse(final)


async def _responses_stream(
    key: str,
    prepared,
    spec,
    response_id: str,
    msg_item_id: str,
    created: int,
    base_response,
    previous_response_id: str | None,
    include_sources: bool = False,
    hashes: list[str] | None = None,
) -> AsyncIterator[str]:
    def event(name: str, payload: dict[str, Any]) -> str:
        payload = {"type": name, **payload}
        return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    snapshot = base_response()
    yield event("response.created", {"response": snapshot})
    yield event("response.in_progress", {"response": snapshot})
    yield event(
        "response.output_item.added",
        {
            "output_index": 0,
            "item": {
                "id": msg_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield event(
        "response.content_part.added",
        {
            "item_id": msg_item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    full_text: list[str] = []
    reasoning_text: list[str] = []
    search_queries: list[str] = []
    sources: list[dict[str, Any]] = []
    try:
        async for ev in manager().stream_turn(
            key,
            prepared,
            deepthink=spec.deepthink,
            model_type=spec.model_type,
            hashes=hashes,
        ):
            if ev.kind == "reasoning":
                reasoning_text.append(str(ev.value))
                yield event(
                    "response.reasoning_text.delta",
                    {"item_id": msg_item_id, "output_index": 0, "delta": str(ev.value)},
                )
            elif ev.kind == "content":
                full_text.append(str(ev.value))
                yield event(
                    "response.output_text.delta",
                    {
                        "item_id": msg_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": str(ev.value),
                    },
                )
            elif ev.kind == "search":
                if isinstance(ev.value, list):
                    search_queries.extend(str(q) for q in ev.value if q)
            elif ev.kind == "sources":
                if isinstance(ev.value, list):
                    sources = list(ev.value)
        if include_sources and sources:
            appendix_query = search_queries[0] if search_queries else prepared.prompt
            appendix = source_appendix(sources, appendix_query)
            if appendix:
                full_text.append(appendix)
                yield event(
                    "response.output_text.delta",
                    {
                        "item_id": msg_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": appendix,
                    },
                )
        text = "".join(full_text)
        yield event(
            "response.output_text.done",
            {
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            },
        )
        yield event(
            "response.content_part.done",
            {
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        )
        yield event(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {
                    "id": msg_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                },
            },
        )
        final = base_response("completed")
        final["output"] = [
            item
            for item in [
                (
                    {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex}",
                        "summary": [
                            {"type": "summary_text", "text": "".join(reasoning_text)}
                        ],
                    }
                    if reasoning_text
                    else None
                ),
                {
                    "id": msg_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                },
            ]
            if item is not None
        ]
        final["completed_at"] = int(time.time())
        final["usage"] = {
            "input_tokens": _estimate(prepared.prompt),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": _estimate(text),
            "output_tokens_details": {
                "reasoning_tokens": _estimate("".join(reasoning_text))
            },
            "total_tokens": _estimate(prepared.prompt) + _estimate(text),
        }
        _store_response_link(response_id, key, spec.wire_id)
        yield event("response.completed", {"response": final})
    except DeepSeekError as exc:
        failed = base_response("failed")
        failed["error"] = {"code": "upstream_error", "message": str(exc)}
        yield event("response.failed", {"response": failed})


def _normalize_responses_input(value: Any) -> list[dict[str, Any]]:
    """Accept string input or the documented item shapes; flatten to messages."""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    out: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, str):
            out.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            itype = item.get("type")
            role = item.get("role")
            if role in ("user", "assistant", "system", "developer"):
                out.append(item)
            elif itype == "message":
                out.append({**item, "role": item.get("role", "assistant")})
            elif itype == "function_call_output":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": item.get("output"),
                    }
                )
            elif itype == "reasoning":
                summary = "".join(
                    p.get("text", "")
                    for p in (item.get("summary") or [])
                    if isinstance(p, dict)
                )
                if summary:
                    out.append(
                        {"role": "assistant", "content": f"[reasoning] {summary}"}
                    )
            # function_call / tool echoes without output are skipped
    return out


_RESPONSE_LINK_MAX = 10_000
_response_links: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _store_response_link(response_id: str, conversation_key: str, model: str) -> None:
    """LRU-bounded map so old ids age out one-by-one (never a mass wipe)."""
    _response_links[response_id] = {"conversation": conversation_key, "model": model}
    _response_links.move_to_end(response_id)
    while len(_response_links) > _RESPONSE_LINK_MAX:
        _response_links.popitem(last=False)
    _storage.store_response_link(
        response_id, conversation_key, model, limit=_RESPONSE_LINK_MAX
    )


@app.get("/v1/responses/{response_id}", dependencies=[Depends(require_api_key)])
async def get_response(response_id: str):
    stored = _response_links.get(response_id)
    if not stored:
        stored = _storage.get_response_link(response_id)
        if stored:
            _response_links[response_id] = stored
    if not stored:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"no response found with id '{response_id}'",
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
        )
    transcript = manager().transcript(stored["conversation"])
    last_answer = next(
        (m["content"] for m in reversed(transcript) if m["role"] == "assistant"), ""
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
                    {"type": "output_text", "text": last_answer, "annotations": []}
                ],
            }
        ],
    }


@app.delete("/v1/sessions/{key}", dependencies=[Depends(require_api_key)])
async def delete_session(key: str) -> dict[str, Any]:
    await manager().reset(key)
    return {"status": "deleted", "session": key}

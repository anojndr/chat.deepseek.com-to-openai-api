"""Conversation state and request orchestration with account failover.

A "conversation" is keyed by an opaque proxy id (or client-supplied session
id). Each conversation pins a DeepSeek chat_session + the account that owns
it, so multi-turn context lives natively inside DeepSeek. If the owning
account turns unhealthy or DeepSeek loses the session, the next turn replays
the accumulated transcript into a fresh session on another account.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .accounts import AccountPool
from .aggregator import FragmentAggregator
from .deepseek import DeepSeekClient, DeepSeekError
from .pow_solver import PowSolver
from .turn import PreparedTurn


@dataclass
class Conversation:
    id: str
    account_index: int | None = None
    deepseek_session_id: str | None = None
    parent_message_id: int | None = None
    # full transcript for replay after failover: list of {"role","content"}
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)


@dataclass
class TurnResult:
    content: str
    reasoning: str | None
    title: str | None


@dataclass
class StreamEvent:
    kind: str  # 'reasoning' | 'content' | 'meta'
    value: str | dict[str, Any]


MAX_HISTORY_CHARS = 400_000
SESSION_TTL = 6 * 3600.0
SWEEP_INTERVAL = 900.0


class ConversationManager:
    def __init__(self, pool: AccountPool, pow_solver: PowSolver) -> None:
        self._pool = pool
        self._solver = pow_solver
        self._conversations: dict[str, Conversation] = {}
        self._clients: dict[int, DeepSeekClient] = {}
        self._lock = asyncio.Lock()
        self._sweeper = asyncio.create_task(self._sweep_loop())

    def client_for(self, account_index: int, token: str) -> DeepSeekClient:
        client = self._clients.get(account_index)
        if client is None:
            client = DeepSeekClient(token, self._solver)
            self._clients[account_index] = client
        return client

    async def aclose(self) -> None:
        self._sweeper.cancel()
        for client in self._clients.values():
            await client.aclose()

    async def get_or_create(self, key: str) -> Conversation:
        async with self._lock:
            conv = self._conversations.get(key)
            if conv is None:
                conv = Conversation(id=key)
                self._conversations[key] = conv
            conv.last_used_at = time.monotonic()
            return conv

    def _is_first_turn(self, conv: Conversation) -> bool:
        return conv.deepseek_session_id is None

    async def _ensure_session(self, conv: Conversation) -> tuple[DeepSeekClient, int]:
        """Return (client, account_index); (re)create session when needed."""
        if conv.deepseek_session_id and conv.account_index is not None:
            client = self._clients.get(conv.account_index)
            if client is not None:
                return client, conv.account_index
            # lost client (pool reloaded): force new session
            conv.deepseek_session_id = None
            conv.parent_message_id = None

        account = self._pool.acquire()
        client = self.client_for(account.index, account.token)
        try:
            session_id = await client.create_session()
        except Exception as exc:
            self._pool.mark_failure(account)
            raise DeepSeekError(f"could not create session on account {account.index}: {exc}") from exc
        conv.account_index = account.index
        conv.deepseek_session_id = session_id
        conv.parent_message_id = None
        return client, account.index

    async def run_turn(
        self,
        key: str,
        prepared: PreparedTurn,
        *,
        deepthink: bool,
        model_type: str | None,
        max_retries: int | None = None,
    ) -> TurnResult:
        conv = await self.get_or_create(key)
        attempts = max_retries or len(self._pool.snapshot()) or 1
        last_error: Exception | None = None

        for attempt in range(max(attempts, 1)):
            first_turn = self._is_first_turn(conv)
            replay = first_turn and attempt > 0
            try:
                client, account_index = await self._ensure_session(conv)
                file_ids = await self._upload_files(client, conv, prepared, replay=replay)
                effective_model = "vision" if file_ids else model_type
                result = await self._collect(
                    client,
                    prompt=prepared.prompt if not replay else prepared.prompt,
                    conv=conv,
                    ref_file_ids=file_ids,
                    thinking_enabled=deepthink,
                    search_enabled=True,
                    model_type=effective_model,
                )
                if not result.content:
                    # silent empty stream (e.g. muted/rate-limited account) ->
                    # treat as failure and rotate to the next account
                    raise RuntimeError("empty completion from upstream")
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # upstream refusal/network -> rotate
                last_error = exc
                self._pool.mark_failure_by_index(conv.account_index)
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                continue
        raise DeepSeekError(f"all accounts failed for this turn: {last_error}")

    async def stream_turn(
        self,
        key: str,
        prepared: PreparedTurn,
        *,
        deepthink: bool,
        model_type: str | None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming variant; yields deltas as they arrive from DeepSeek."""
        conv = await self.get_or_create(key)
        attempts = max(len(self._pool.snapshot()), 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            first_turn = self._is_first_turn(conv)
            try:
                client, account_index = await self._ensure_session(conv)
                file_ids = await self._upload_files(client, conv, prepared)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # muted/rate-limited/network -> rotate
                last_error = exc
                self._pool.mark_failure_by_index(conv.account_index)
                conv.deepseek_session_id = None
                continue

            buffer_content: list[str] = []
            buffer_reasoning: list[str] = []
            emitted = False
            effective_model = "vision" if file_ids else model_type
            try:
                async for ev in self._stream_events(
                    client,
                    prompt=prepared.prompt,
                    conv=conv,
                    ref_file_ids=file_ids,
                    thinking_enabled=deepthink,
                    model_type=effective_model,
                ):
                    if ev.kind == "content":
                        buffer_content.append(str(ev.value))
                    elif ev.kind == "reasoning":
                        buffer_reasoning.append(str(ev.value))
                    elif ev.kind == "search":
                        yield StreamEvent("search", ev.value)
                        continue
                    emitted = True
                    yield ev
                if not emitted:
                    yield StreamEvent("meta", {"empty": True})
                text = "".join(buffer_content)
                reasoning = "".join(buffer_reasoning)
                if not text:
                    # No RESPONSE fragment (answer stuck in THINK or silent
                    # stream): retry on another attempt/account.
                    raise RuntimeError("completion returned no content")
                self._record_history(conv, prepared.prompt, text)
                self._pool.mark_success_by_index(account_index)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                self._pool.mark_failure_by_index(account_index)
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                if emitted:
                    # mid-stream failure after output started: surface error
                    raise DeepSeekError(f"stream interrupted: {exc}") from exc
                continue
        raise DeepSeekError(f"all accounts failed for this turn: {last_error}")

    # -- internals -----------------------------------------------------------

    async def _upload_files(
        self,
        client: DeepSeekClient,
        conv: Conversation,
        prepared: PreparedTurn,
        replay: bool = False,
    ) -> list[str]:
        if not prepared.files:
            return []
        ids: list[str] = []
        for filename, raw, mime in prepared.files:
            is_image = (mime or "").startswith("image/") or any(
                filename.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
            )
            ids.append(
                await client.upload_file(filename, raw, mime, vision=is_image)
            )
        return ids

    @staticmethod
    def _body(
        conv: Conversation,
        prompt: str,
        ref_file_ids: list[str],
        thinking: bool,
        model_type: str | None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "parent_message_id": conv.parent_message_id,
            "ref_file_ids": ref_file_ids,
            "thinking_enabled": thinking,
            "search_enabled": True,  # always-on per requirements
            "model_type": model_type,
        }

    async def _collect(
        self,
        client: DeepSeekClient,
        *,
        prompt: str,
        conv: Conversation,
        ref_file_ids: list[str],
        thinking_enabled: bool,
        search_enabled: bool = True,
        model_type: str | None,
    ) -> TurnResult:
        agg = FragmentAggregator()
        title: str | None = None
        async for event in client.stream_completion(
            prompt=prompt,
            chat_session_id=conv.deepseek_session_id or "",
            parent_message_id=conv.parent_message_id,
            ref_file_ids=ref_file_ids,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
        ):
            if event.get("event") == "ready":
                rid = (event.get("data") or {}).get("response_message_id")
                if isinstance(rid, int):
                    conv.parent_message_id = rid
            for kind, value in agg.apply(event.get("event"), event.get("data")):
                if kind == "meta" and isinstance(value, dict) and value.get("title"):
                    title = str(value["title"])
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for frag in agg.fragments:
            ftype = frag.get("type")
            if ftype == "THINK":
                reasoning_parts.append(frag.get("content") or "")
            elif ftype in ("", "RESPONSE"):
                content_parts.append(frag.get("content") or "")
        return TurnResult(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts) or None,
            title=title,
        )

    async def _stream_events(
        self,
        client: DeepSeekClient,
        *,
        prompt: str,
        conv: Conversation,
        ref_file_ids: list[str],
        thinking_enabled: bool,
        model_type: str | None,
    ) -> AsyncIterator[StreamEvent]:
        agg = FragmentAggregator()
        response_id: int | None = None
        async for event in client.stream_completion(
            prompt=prompt,
            chat_session_id=conv.deepseek_session_id or "",
            parent_message_id=conv.parent_message_id,
            ref_file_ids=ref_file_ids,
            thinking_enabled=thinking_enabled,
            search_enabled=True,
            model_type=model_type,
        ):
            if event.get("event") == "ready":
                data = event.get("data") or {}
                rid = data.get("response_message_id")
                if isinstance(rid, int):
                    response_id = rid
                continue
            for kind, value in agg.apply(event.get("event"), event.get("data")):
                if kind in ("content", "reasoning"):
                    if value:
                        yield StreamEvent(kind, value)
                elif kind == "search":
                    yield StreamEvent("search", value)
        if response_id is not None:
            conv.parent_message_id = response_id

    def _record_history(self, conv: Conversation, prompt: str, answer: str) -> None:
        conv.history.append({"role": "user", "content": prompt})
        conv.history.append({"role": "assistant", "content": answer})
        total = sum(len(m["content"]) for m in conv.history)
        while total > MAX_HISTORY_CHARS and len(conv.history) > 2:
            removed = conv.history.pop(0)
            total -= len(removed["content"])

    def transcript(self, key: str) -> list[dict[str, str]]:
        conv = self._conversations.get(key)
        return list(conv.history) if conv else []

    async def reset(self, key: str) -> None:
        async with self._lock:
            conv = self._conversations.pop(key, None)
        if conv and conv.account_index is not None and conv.deepseek_session_id:
            client = self._clients.get(conv.account_index)
            if client:
                await client.delete_session(conv.deepseek_session_id)

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL)
            now = time.monotonic()
            stale = [k for k, v in self._conversations.items() if now - v.last_used_at > SESSION_TTL]
            for k in stale:
                await self.reset(k)


def new_conversation_key() -> str:
    return uuid.uuid4().hex

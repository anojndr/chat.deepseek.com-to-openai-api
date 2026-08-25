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
from .citations import CitationRewriter, rewrite_citations
from .deepseek import DeepSeekClient, DeepSeekError
from .pow_solver import PowSolver
from .turn import PreparedTurn


@dataclass
class Conversation:
    id: str
    account_index: int | None = None
    account_token: str | None = None
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


class EmptyCompletion(Exception):
    """Upstream answered with no content (refusal, mute, empty stream).

    Rotates to another account but never poisons the pool's health state:
    an empty answer is not proof an account is broken.
    """


class ConversationManager:
    def __init__(self, pool: AccountPool, pow_solver: PowSolver) -> None:
        self._pool = pool
        self._solver = pow_solver
        self._conversations: dict[str, Conversation] = {}
        self._clients: dict[str, DeepSeekClient] = {}  # keyed by account token
        self._lock = asyncio.Lock()  # guards _conversations / _clients maps
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._sweeper = asyncio.create_task(self._sweep_loop())

    def client_for(self, token: str) -> DeepSeekClient:
        client = self._clients.get(token)
        if client is None or client.token != token:
            client = DeepSeekClient(token, self._solver)
            self._clients[token] = client
        return client

    async def invalidate_clients(self, keep_tokens: set[str]) -> int:
        """Drop clients for tokens no longer present in the pool."""
        stale = [t for t in self._clients if t not in keep_tokens]
        for t in stale:
            client = self._clients.pop(t, None)
            if client:
                await client.aclose()
        return len(stale)

    def key_lock(self, key: str) -> asyncio.Lock:
        lock = self._key_locks.get(key)
        if lock is None:
            lock = self._key_locks[key] = asyncio.Lock()
        return lock

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

    async def _ensure_session(
        self, conv: Conversation
    ) -> tuple[DeepSeekClient, Account, bool]:
        """Return (client, account, fresh_session).

        fresh_session is True when a new DeepSeek session was created for an
        existing conversation — the caller must replay prior context.
        """
        if conv.deepseek_session_id and conv.account_token:
            client = self._clients.get(conv.account_token)
            if client is not None:
                return client, self._pool.by_token(conv.account_token), False
            # lost client (pool reloaded): force new session
            conv.deepseek_session_id = None
            conv.parent_message_id = None

        account = self._pool.acquire()
        client = self.client_for(account.token)
        try:
            session_id = await client.create_session()
        except Exception as exc:
            self._pool.mark_failure(account.token)
            raise DeepSeekError(
                f"could not create session on account {account.index}: {exc}"
            ) from exc
        fresh = conv.history and conv.deepseek_session_id != session_id
        conv.account_index = account.index
        conv.account_token = account.token
        conv.deepseek_session_id = session_id
        conv.parent_message_id = None
        return client, account, bool(fresh)

    @staticmethod
    def _replay_prompt(conv: Conversation, prepared: PreparedTurn) -> PreparedTurn:
        """Rebuild a first-turn-style prompt from stored history after failover."""
        lines: list[str] = []
        for msg in conv.history:
            lines.append(f"[{msg['role']}] {msg['content']}")
        lines.append(prepared.prompt)
        return PreparedTurn(prompt="\n\n".join(lines).strip(), files=prepared.files)

    async def run_turn(
        self,
        key: str,
        prepared: PreparedTurn,
        *,
        deepthink: bool,
        model_type: str | None,
        max_retries: int | None = None,
    ) -> TurnResult:
        async with self.key_lock(key):
            return await self._run_turn_locked(key, prepared, deepthink=deepthink, model_type=model_type, max_retries=max_retries)

    async def _run_turn_locked(
        self,
        key: str,
        prepared: PreparedTurn,
        *,
        deepthink: bool,
        model_type: str | None,
        max_retries: int | None = None,
    ) -> TurnResult:
        conv = await self.get_or_create(key)
        attempts = max_retries or self._pool.size or 1
        last_error: Exception | None = None
        last_ds_error: DeepSeekError | None = None

        for attempt in range(max(attempts, 1)):
            prev_session = conv.deepseek_session_id
            account_token: str | None = None
            try:
                client, account, fresh_session = await self._ensure_session(conv)
                account_token = account.token
                turn_prepared = prepared
                if fresh_session and conv.history:
                    # session lost mid-conversation (failover): replay context
                    turn_prepared = self._replay_prompt(conv, prepared)
                elif not fresh_session and conv.history and prev_session is None:
                    turn_prepared = self._replay_prompt(conv, prepared)
                file_ids = await self._upload_files(client, conv, turn_prepared)
                # explicit caller model wins; auto-select vision only when the
                # caller left it unspecified but files need vision parsing
                effective_model = model_type or ("vision" if file_ids else None)
                result = await self._collect(
                    client,
                    prompt=turn_prepared.prompt,
                    conv=conv,
                    ref_file_ids=file_ids,
                    thinking_enabled=deepthink,
                    search_enabled=True,
                    model_type=effective_model,
                )
                if not result.content:
                    # refusal / muted / empty stream. Rotate attempts WITHOUT
                    # poisoning pool health: an empty answer is not proof the
                    # account is broken.
                    raise EmptyCompletion("empty completion from upstream")
                self._record_history(conv, turn_prepared.prompt, result.content)
                self._pool.mark_success(account.token)
                return result
            except asyncio.CancelledError:
                raise
            except EmptyCompletion as exc:
                last_error = exc
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                continue
            except Exception as exc:
                last_error = exc
                if isinstance(exc, DeepSeekError):
                    last_ds_error = exc
                if account_token:
                    self._pool.mark_failure(account_token)
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                continue
        final = last_ds_error or DeepSeekError(f"all accounts failed for this turn: {last_error}")
        raise final

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
        async with self.key_lock(key):
            async for ev in self._stream_turn_locked(
                key, conv, prepared, deepthink=deepthink, model_type=model_type
            ):
                yield ev

    async def _stream_turn_locked(
        self,
        key: str,
        conv: Conversation,
        prepared: PreparedTurn,
        *,
        deepthink: bool,
        model_type: str | None,
    ) -> AsyncIterator[StreamEvent]:
        attempts = max(self._pool.size, 1)
        last_error: Exception | None = None
        last_ds_error: DeepSeekError | None = None

        for attempt in range(attempts):
            prev_session = conv.deepseek_session_id
            account_token: str | None = None
            try:
                client, account, fresh_session = await self._ensure_session(conv)
                account_token = account.token
                turn_prepared = prepared
                if fresh_session and conv.history:
                    turn_prepared = self._replay_prompt(conv, prepared)
                elif not fresh_session and conv.history and prev_session is None:
                    turn_prepared = self._replay_prompt(conv, prepared)
                file_ids = await self._upload_files(client, conv, turn_prepared)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # muted/rate-limited/network -> rotate
                last_error = exc
                if isinstance(exc, DeepSeekError):
                    last_ds_error = exc
                if account_token:
                    self._pool.mark_failure(account_token)
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                continue

            buffer_content: list[str] = []
            buffer_reasoning: list[str] = []
            searches: list[Any] = []
            reference_urls: list[str | None] = []
            rewriter = CitationRewriter(reference_urls)
            emitted = False
            effective_model = model_type or ("vision" if file_ids else None)
            try:
                async for ev in self._stream_events(
                    client,
                    prompt=turn_prepared.prompt,
                    conv=conv,
                    ref_file_ids=file_ids,
                    thinking_enabled=deepthink,
                    model_type=effective_model,
                ):
                    if ev.kind == "references":
                        reference_urls.extend(ev.value if isinstance(ev.value, list) else [])
                    elif ev.kind == "content":
                        chunk = rewriter.feed(str(ev.value))
                        if chunk:
                            buffer_content.append(chunk)
                            emitted = True
                            yield StreamEvent("content", chunk)
                    elif ev.kind == "reasoning":
                        buffer_reasoning.append(str(ev.value))
                        emitted = True
                        yield ev
                    else:
                        # search/meta events are buffered so retries never
                        # duplicate output the client already received
                        if isinstance(ev.value, list):
                            searches.extend(ev.value)
                        emitted = True
                final_chunk = rewriter.finish()
                if final_chunk:
                    buffer_content.append(final_chunk)
                    emitted = True
                    yield StreamEvent("content", final_chunk)
                text = "".join(buffer_content)
                reasoning = "".join(buffer_reasoning)
                if not text:
                    # No RESPONSE fragment (refusal / answer stuck in THINK /
                    # silent stream): rotate WITHOUT poisoning pool health and
                    # without having emitted any content to the client.
                    raise EmptyCompletion("completion returned no content")
                if searches:
                    yield StreamEvent("search", searches)
                self._record_history(conv, turn_prepared.prompt, text)
                self._pool.mark_success(account_token or "")
                return
            except asyncio.CancelledError:
                raise
            except EmptyCompletion as exc:
                last_error = exc
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                continue
            except Exception as exc:
                last_error = exc
                if isinstance(exc, DeepSeekError):
                    last_ds_error = exc
                if account_token:
                    self._pool.mark_failure(account_token)
                conv.deepseek_session_id = None
                conv.parent_message_id = None
                if emitted:
                    # mid-stream failure after content was streamed: surface
                    raise DeepSeekError(
                        f"stream interrupted: {exc}",
                        status=getattr(exc, "status", None),
                        biz_code=getattr(exc, "biz_code", None),
                    ) from exc
                continue
        final = last_ds_error or DeepSeekError(f"all accounts failed for this turn: {last_error}")
        raise final

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
        content = rewrite_citations("".join(content_parts), agg.reference_urls)
        return TurnResult(
            content=content,
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
        announced_refs = 0
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
            emitted_from_patch = False
            for kind, value in agg.apply(event.get("event"), event.get("data")):
                emitted_from_patch = True
                if kind in ("content", "reasoning"):
                    if value:
                        yield StreamEvent(kind, value)
                elif kind == "search":
                    yield StreamEvent("search", value)
            if len(agg.reference_urls) > announced_refs:
                new_refs = agg.reference_urls[announced_refs:]
                announced_refs = len(agg.reference_urls)
                yield StreamEvent("references", new_refs)
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
        async with self.key_lock(key):
            async with self._lock:
                conv = self._conversations.pop(key, None)
        if conv and conv.account_token and conv.deepseek_session_id:
            client = self._clients.get(conv.account_token)
            if client:
                await client.delete_session(conv.deepseek_session_id)

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(SWEEP_INTERVAL)
                now = time.monotonic()
                async with self._lock:
                    stale = [
                        k
                        for k, v in self._conversations.items()
                        if now - v.last_used_at > SESSION_TTL
                    ]
                for k in stale:
                    await self.reset(k)
            except asyncio.CancelledError:
                return
            except Exception:
                # sweeper must never die; retry next interval
                continue


def new_conversation_key() -> str:
    return uuid.uuid4().hex

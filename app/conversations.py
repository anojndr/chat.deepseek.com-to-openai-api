# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Conversation state and request orchestration with account failover.

A "conversation" is keyed by an opaque proxy id (or client-supplied session
id). Each conversation pins a DeepSeek chat_session + the account that owns
it, so multi-turn context lives natively inside DeepSeek. If the owning
account turns unhealthy or DeepSeek loses the session, the next turn replays
the accumulated transcript into a fresh session on another account.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack

import httpx

from .aggregator import FragmentAggregator
from .citations import CitationRewriter, rewrite_citations
from .deepseek import (
    AccountMutedError,
    DeepSeekClient,
    DeepSeekError,
    ModerationError,
    NonRetryableCompletionError,
    UploadModerationError,
)
from .storage import ConvRef, Storage
from .turn import PreparedTurn, item_hash

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .accounts import Account, AccountPool
    from .pow_solver import PowSolver

MAX_HISTORY_CHARS = 400_000
SESSION_TTL = 6 * 3600.0
SWEEP_INTERVAL = 900.0
logger = logging.getLogger(__name__)
_MIN_ATTEMPTS = 2
# Minimum retained transcript length when trimming history.
_MIN_HISTORY_KEEP = 2
# Image suffixes that force vision upload handling.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})
# Content fragment types collected as reasoning versus answer text.
_THINK_FRAGMENT = "THINK"
_RESPONSE_FRAGMENTS = frozenset({"", "RESPONSE"})
# Stream event kinds emitted as content deltas.
_CONTENT_KINDS = frozenset({"content", "reasoning"})
# Turn failures the web client treats as terminal: moderation mutes and
# deterministic request rejections. Retrying them on the next account
# re-sends the same flagged prompt, burning every account in turn.
_TERMINAL_TURN_ERRORS = (
    ModerationError,
    AccountMutedError,
    NonRetryableCompletionError,
    UploadModerationError,
)


def _is_terminal_turn_error(exc: BaseException) -> bool:
    """Check whether a turn failure must not rotate accounts.

    Args:
        exc: Failure from one turn attempt.

    Returns:
        True for moderation mutes and deterministic rejections.

    """
    return isinstance(exc, _TERMINAL_TURN_ERRORS)


class RunTurnOptions(TypedDict):
    """Options for a non-streaming turn."""

    deepthink: bool
    model_type: str | None
    max_retries: NotRequired[int | None]
    hashes: NotRequired[list[str] | None]


class StreamTurnOptions(TypedDict):
    """Options for a streaming turn."""

    deepthink: bool
    model_type: str | None
    hashes: NotRequired[list[str] | None]


class CollectOptions(TypedDict):
    """Options for collecting a completed turn."""

    ref_file_ids: NotRequired[list[str]]
    thinking_enabled: NotRequired[bool]
    search_enabled: NotRequired[bool]
    model_type: NotRequired[str | None]


class StreamEventsOptions(TypedDict):
    """Options for streaming raw turn events."""

    ref_file_ids: NotRequired[list[str]]
    thinking_enabled: NotRequired[bool]
    model_type: NotRequired[str | None]


@dataclass
class Conversation:
    """Store pinned session and transcript for one proxy conversation."""

    id: str
    account_index: int | None = None
    account_token: str | None = None
    deepseek_session_id: str | None = None
    parent_message_id: int | None = None
    # full transcript for replay after failover: list of {"role","content"}
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)


@dataclass
class TurnResult:
    """Hold the aggregated result of one completed turn."""

    content: str
    reasoning: str | None
    title: str | None
    sources: list[dict[str, object]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


@dataclass
class StreamEvent:
    """Carry one streaming delta to the caller."""

    kind: str  # 'reasoning' | 'content' | 'search' | 'sources' | 'references' | 'meta'
    value: str | list[object] | dict[str, object]


@dataclass
class _StreamBuffers:
    """Accumulate streaming deltas across one attempt."""

    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    searches: list[object] = field(default_factory=list)
    reference_urls: list[str | None] = field(default_factory=list)
    sources: list[dict[str, object]] = field(default_factory=list)
    emitted: bool = False


@dataclass
class _DeltaRequest:
    """Carry streaming inputs for one attempt."""

    client: DeepSeekClient
    conv: Conversation
    prompt: str
    file_ids: list[str]
    buffers: _StreamBuffers
    deepthink: bool
    model_type: str | None


@dataclass
class _StreamErrorContext:
    """Carry stream failure details for handling."""

    attempt: int
    attempts: int
    account_token: str | None
    exc: Exception
    emitted: bool
    last_ds: DeepSeekError | None


class EmptyCompletionError(Exception):
    """Signal an upstream answer with no content."""

    def __init__(self, message: str = "empty completion from upstream") -> None:
        """Build fixed message.

        Args:
            message: Detail message for the error.

        """
        super().__init__(message)


class EmptyStreamError(EmptyCompletionError):
    """Signal a streaming turn that yielded no content."""

    def __init__(self, message: str = "completion returned no content") -> None:
        """Build fixed message.

        Args:
            message: Detail message for the error.

        """
        super().__init__(message)


class SessionCreateError(DeepSeekError):
    """Raised when a DeepSeek session cannot be created."""

    def __init__(self, index: int, cause: Exception) -> None:
        """Build message from account index and cause."""
        super().__init__(f"could not create session on account {index}: {cause}")
        self.account_index = index
        self.cause = cause


class TurnExhaustedError(DeepSeekError):
    """Raised when all accounts fail for one turn."""

    def __init__(self, cause: Exception | None) -> None:
        """Build message from the last error."""
        super().__init__(f"all accounts failed for this turn: {cause}")
        self.cause = cause


class StreamInterruptedError(DeepSeekError):
    """Raised when a stream fails after emitting content."""

    def __init__(self, cause: Exception) -> None:
        """Build message from cause, preserving upstream codes."""
        status = cause.status if isinstance(cause, DeepSeekError) else None
        biz_code = cause.biz_code if isinstance(cause, DeepSeekError) else None
        super().__init__(
            f"stream interrupted: {cause}",
            status=status,
            biz_code=biz_code,
        )
        self.cause = cause


def _ensure_non_empty(result: TurnResult) -> None:
    """Reject empty completions without inline raise.

    Raises:
        EmptyCompletionError: If content is empty.

    """
    if not result.content:
        raise EmptyCompletionError


def _ensure_stream_text(text: str) -> None:
    """Reject empty streams without inline raise.

    Raises:
        EmptyStreamError: If text is empty.

    """
    if not text:
        raise EmptyStreamError


class ConversationManager:
    """Coordinate conversations with account failover and retries."""

    def __init__(
        self,
        pool: AccountPool,
        pow_solver: PowSolver,
        storage: Storage | None = None,
    ) -> None:
        """Store pool, solver, storage, and start the sweeper."""
        self._pool = pool
        self._solver = pow_solver
        self._storage = storage
        self._conversations: dict[str, Conversation] = {}
        self._clients: dict[str, DeepSeekClient] = {}  # keyed by account token
        self._lock = asyncio.Lock()  # guards _conversations / _clients maps
        self._key_locks: dict[str, asyncio.Lock] = {}
        if self._storage is not None:
            self._load_from_storage()
        self._sweeper: asyncio.Task[None] | None = None
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self._sweeper = loop.create_task(self._sweep_loop())
        except RuntimeError:
            # No running loop yet (e.g. before startup
            # or in sync tests).
            self._sweeper = None

    def ensure_sweeper(self) -> None:
        """Start the background sweep task if not already running."""
        if self._sweeper is None or self._sweeper.done():
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    self._sweeper = loop.create_task(self._sweep_loop())
            except RuntimeError:
                pass

    def _load_from_storage(self) -> None:
        """Load persisted conversations into memory."""
        if self._storage is None:
            return
        stored = self._storage.get_all_conversations()
        for key, data in stored.items():
            self._conversations[key] = Conversation(
                id=data["id"],
                account_index=data["account_index"],
                account_token=data["account_token"],
                deepseek_session_id=data["deepseek_session_id"],
                parent_message_id=data["parent_message_id"],
                history=data["history"],
                created_at=data["created_at"],
                last_used_at=data["last_used_at"],
            )

    def _persist_conversation(self, conv: Conversation) -> None:
        """Persist one conversation to storage."""
        if self._storage is not None:
            self._storage.save_conversation(
                conv.id,
                account_index=conv.account_index,
                account_token=conv.account_token,
                deepseek_session_id=conv.deepseek_session_id,
                parent_message_id=conv.parent_message_id,
                history=conv.history,
                created_at=conv.created_at,
                last_used_at=conv.last_used_at,
            )

    def persist_conversation(self, conv: Conversation) -> None:
        """Persist one conversation to storage."""
        self._persist_conversation(conv)

    def inherit_from_ref(
        self,
        conv: Conversation,
        ref: ConvRef,
        *,
        history: bool,
    ) -> None:
        """Pin a forked conversation to a parent checkpoint.

        Args:
            conv: Forked conversation receiving the checkpoint.
            ref: Stored parent checkpoint with session and parent ids.
            history: Whether to copy the parent transcript for failover replay.

        """
        conv.account_index = ref.account_index
        conv.account_token = ref.account_token
        conv.deepseek_session_id = ref.deepseek_session_id
        conv.parent_message_id = ref.parent_message_id
        if history:
            parent = self.transcript(ref.conversation_key)
            if parent:
                conv.history = list(parent)
        self._persist_conversation(conv)

    def test_hook_record_history(
        self,
        conv: Conversation,
        prompt: str,
        answer: str,
    ) -> None:
        """Record history via the public test hook."""
        self._record_history(conv, prompt, answer)

    def test_hook_inject_client(self, token: str, client: DeepSeekClient) -> None:
        """Inject a client for tests via the public hook."""
        self._clients[token] = client

    def test_hook_drop_memory(self) -> None:
        """Drop in-memory conversations, keeping SQLite rows for reload."""
        self._conversations.clear()

    async def test_hook_stream_events(
        self,
        client: DeepSeekClient,
        conv: Conversation,
        prompt: str,
        **options: Unpack[StreamEventsOptions],
    ) -> AsyncIterator[StreamEvent]:
        """Yield stream events via the public test hook.

        Yields:
            Stream deltas.

        """
        async for event in self._stream_events(
            client,
            conv,
            prompt,
            **options,
        ):
            yield event

    def _clear_session(self, conv: Conversation, *, reason: str) -> None:
        """Drop the pinned session and invalidate prefix refs to it.

        Follow-ups inherit sessions via longest-prefix match, so a dead id
        left in `prefixes` would be re-inherited and fail deterministically.
        """
        dead = conv.deepseek_session_id
        conv.deepseek_session_id = None
        conv.parent_message_id = None
        try:
            self._persist_conversation(conv)
        except Exception:
            logger.exception("failed to persist cleared session for %s", conv.id)
        if dead and self._storage is not None:
            try:
                self._storage.delete_session_refs(dead)
            except Exception:
                logger.exception("failed to invalidate prefix refs for dead session")
        logger.debug("cleared session for %s: %s", conv.id, reason)

    async def _drop_if_empty(self, conv: Conversation) -> None:
        """Remove junk rows for keys that never recorded a turn.

        `get_or_create` persists `[]` upfront; a turn that fails before the
        first `_record_history` must not leave an empty row behind.
        """
        if conv.history:
            return
        async with self._lock:
            self._conversations.pop(conv.id, None)
        if self._storage is not None:
            try:
                self._storage.delete_conversation(conv.id)
            except Exception:
                logger.exception("failed to delete empty conversation %s", conv.id)

    def client_for(self, token: str) -> DeepSeekClient:
        """Return the cached client for a token, creating it.

        Returns:
            The client bound to the token.

        """
        client = self._clients.get(token)
        if client is None or client.token != token:
            client = DeepSeekClient(token, self._solver)
            self._clients[token] = client
        return client

    async def invalidate_clients(self, keep_tokens: set[str]) -> int:
        """Drop clients for tokens no longer present in the pool.

        Returns:
            The number of dropped clients.

        """
        stale = [t for t in self._clients if t not in keep_tokens]
        for token in stale:
            client = self._clients.pop(token, None)
            if client:
                await client.aclose()
        return len(stale)

    def key_lock(self, key: str) -> asyncio.Lock:
        """Return the per-key lock, creating it.

        Returns:
            The lock guarding the key.

        """
        lock = self._key_locks.get(key)
        if lock is None:
            lock = self._key_locks[key] = asyncio.Lock()
        return lock

    async def aclose(self) -> None:
        """Close the sweeper and all cached clients."""
        if self._sweeper is not None:
            self._sweeper.cancel()
        for client in self._clients.values():
            await client.aclose()

    async def get_or_create(self, key: str) -> Conversation:
        """Load or create the conversation for a key.

        Returns:
            The conversation for the key.

        """
        async with self._lock:
            conv = self._conversations.get(key)
            if conv is None:
                conv = await self._load_or_init(key)
            conv.last_used_at = time.time()
            self._persist_conversation(conv)
            return conv

    async def _load_or_init(self, key: str) -> Conversation:
        """Load a stored conversation or initialize a fresh one.

        Returns:
            The loaded or new conversation.

        """
        if self._storage is not None:
            data = self._storage.get_conversation(key)
            if data is not None:
                conv = Conversation(
                    id=data["id"],
                    account_index=data["account_index"],
                    account_token=data["account_token"],
                    deepseek_session_id=data["deepseek_session_id"],
                    parent_message_id=data["parent_message_id"],
                    history=data["history"],
                    created_at=data["created_at"],
                    last_used_at=data["last_used_at"],
                )
                self._conversations[key] = conv
                return conv
        conv = Conversation(id=key)
        self._conversations[key] = conv
        return conv

    @staticmethod
    def _is_first_turn(conv: Conversation) -> bool:
        """Check whether a conversation needs its first session.

        Returns:
            True when no session is pinned.

        """
        return conv.deepseek_session_id is None

    async def _ensure_session(
        self,
        conv: Conversation,
    ) -> tuple[DeepSeekClient, Account, bool]:
        """Return client, account, and freshness for a conversation.

        Freshness is True when a new DeepSeek session was created for an
        existing conversation — the caller must replay prior context.

        Returns:
            Tuple of client, account, and freshness flag.

        Raises:
            SessionCreateError: If session creation fails.

        """
        if conv.deepseek_session_id and conv.account_token:
            account = self._pool.by_token(conv.account_token)
            if account is not None:
                client = self.client_for(conv.account_token)
                return client, account, False
            # Lost account (token no longer in pool): force new session.
            conv.deepseek_session_id = None
            conv.parent_message_id = None
        account = self._pool.acquire()
        client = self.client_for(account.token)
        try:
            session_id = await client.create_session()
        except (
            DeepSeekError,
            httpx.HTTPError,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            self._pool.mark_failure(account.token)
            raise SessionCreateError(account.index, exc) from exc
        else:
            fresh = conv.history and conv.deepseek_session_id != session_id
            conv.account_index = account.index
            conv.account_token = account.token
            conv.deepseek_session_id = session_id
            conv.parent_message_id = None
            self._persist_conversation(conv)
            return client, account, bool(fresh)

    @staticmethod
    def _replay_prompt(conv: Conversation, prepared: PreparedTurn) -> PreparedTurn:
        """Rebuild a first-turn prompt from history after failover.

        History entries recorded from a failed full first-turn prompt may
        already carry role labels, so keep pre-labeled lines verbatim.

        Returns:
            The replayed turn.

        """
        lines: list[str] = []
        for msg in conv.history:
            content = msg["content"]
            if content.lstrip().startswith(
                ("[user]", "[assistant]", "[tool result]", "[system"),
            ):
                lines.append(content)
            else:
                lines.append(f"[{msg['role']}] {content}")
        lines.append(prepared.prompt)
        return PreparedTurn(prompt="\n\n".join(lines).strip(), files=prepared.files)

    @staticmethod
    def _resolve_prepared(
        conv: Conversation,
        prepared: PreparedTurn,
        prev_session: str | None,
        *,
        fresh_session: bool,
    ) -> PreparedTurn:
        """Resolve replay versus direct prompt for one attempt.

        Returns:
            The prepared turn to send.

        """
        needs_replay = bool(conv.history) and (fresh_session or prev_session is None)
        if needs_replay:
            return ConversationManager._replay_prompt(conv, prepared)
        return prepared

    def _attempt_count(self, max_retries: int | None) -> int:
        """Compute attempts from pool size and retry override.

        Returns:
            The attempt count.

        """
        return max(max_retries or 0, self._pool.size, _MIN_ATTEMPTS)

    async def _upload_and_collect(
        self,
        client: DeepSeekClient,
        conv: Conversation,
        turn_prepared: PreparedTurn,
        *,
        deepthink: bool,
        model_type: str | None,
    ) -> TurnResult:
        """Upload files and collect one turn result.

        Returns:
            The turn result.

        """
        file_ids = await self._upload_files(client, turn_prepared)
        return await self._collect(
            client,
            conv,
            turn_prepared.prompt,
            ref_file_ids=file_ids,
            thinking_enabled=deepthink,
            search_enabled=True,
            model_type=model_type,
        )

    def _record_prefix_turn(
        self,
        conv: Conversation,
        content: str,
        hashes: list[str] | None,
    ) -> None:
        """Record prefix linkage for branching recovery."""
        if hashes and self._storage is not None and conv.deepseek_session_id:
            assistant_hash = item_hash(hashes[-1], "assistant", content)
            ref = ConvRef(
                conversation_key=conv.id,
                account_index=conv.account_index,
                account_token=conv.account_token or "",
                deepseek_session_id=conv.deepseek_session_id,
                parent_message_id=conv.parent_message_id,
                turns=len(hashes),
                updated_at=time.time(),
            )
            self._storage.record_prefix_turn([assistant_hash], ref)

    def _succeed_turn(
        self,
        conv: Conversation,
        account_token: str,
        turn_prepared: PreparedTurn,
        result: TurnResult,
        hashes: list[str] | None,
    ) -> TurnResult:
        """Record history and health for a successful turn.

        Returns:
            The turn result.

        """
        self._record_history(conv, turn_prepared.prompt, result.content)
        self._record_prefix_turn(conv, result.content, hashes)
        self._pool.mark_success(account_token)
        return result

    def _fail_turn(
        self,
        conv: Conversation,
        attempt: int,
        attempts: int,
        account_token: str | None,
        exc: Exception,
    ) -> None:
        """Mark failure and clear session for a failed attempt."""
        if account_token:
            self._pool.mark_failure(account_token)
        self._clear_session(conv, reason=str(exc))
        logger.warning(
            "run_turn failed key=%s attempt=%d/%d err=%s",
            conv.id,
            attempt + 1,
            max(attempts, 1),
            exc,
        )

    def _empty_turn(
        self,
        conv: Conversation,
        key: str,
        attempt: int,
        attempts: int,
        exc: Exception,
    ) -> None:
        """Clear session for an empty completion without pool poison."""
        self._clear_session(conv, reason=f"empty completion: {exc}")
        logger.warning(
            "run_turn empty completion key=%s attempt=%d/%d",
            key,
            attempt + 1,
            max(attempts, 1),
        )

    async def run_turn(
        self,
        key: str,
        prepared: PreparedTurn,
        **options: Unpack[RunTurnOptions],
    ) -> TurnResult:
        """Run one turn with failover across accounts.

        Returns:
            The turn result.

        """
        async with self.key_lock(key):
            return await self._run_turn_locked(
                key,
                prepared,
                **options,
            )

    async def _run_turn_locked(
        self,
        key: str,
        prepared: PreparedTurn,
        **options: Unpack[RunTurnOptions],
    ) -> TurnResult:
        """Run one locked turn with retries.

        Terminal moderation and rejection failures raise immediately
        without rotating accounts.

        Returns:
            The turn result.

        Raises:
            AccountMutedError: If the account is muted.
            CancelledError: If cancelled.
            ModerationError: If the prompt tripped moderation.
            NonRetryableCompletionError: If the turn is rejected terminally.
            TurnExhaustedError: If all accounts fail.
            UploadModerationError: If a file tripped moderation.

        """
        conv = await self.get_or_create(key)
        attempts = self._attempt_count(options.get("max_retries"))
        last_error: Exception | None = None
        last_ds_error: DeepSeekError | None = None
        for attempt in range(max(attempts, 1)):
            prev_session = conv.deepseek_session_id
            account_token: str | None = None
            try:
                client, account, fresh = await self._ensure_session(conv)
                account_token = account.token
                turn_prepared = self._resolve_prepared(
                    conv,
                    prepared,
                    prev_session,
                    fresh_session=fresh,
                )
                result = await self._upload_and_collect(
                    client,
                    conv,
                    turn_prepared,
                    deepthink=options["deepthink"],
                    model_type=options["model_type"],
                )
                _ensure_non_empty(result)
            except asyncio.CancelledError:
                self._clear_session(conv, reason="cancelled")
                raise
            except EmptyCompletionError as exc:
                last_error = exc
                self._empty_turn(conv, key, attempt, attempts, exc)
                continue
            except (AccountMutedError, UploadModerationError) as exc:
                self._fail_turn(conv, attempt, attempts, account_token, exc)
                await self._drop_if_empty(conv)
                raise
            except (ModerationError, NonRetryableCompletionError) as exc:
                self._fail_turn(conv, attempt, attempts, None, exc)
                await self._drop_if_empty(conv)
                raise
            except (
                DeepSeekError,
                httpx.HTTPError,
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                last_error = exc
                if isinstance(exc, DeepSeekError):
                    last_ds_error = exc
                self._fail_turn(conv, attempt, attempts, account_token, exc)
                continue
            else:
                return self._succeed_turn(
                    conv,
                    account_token or "",
                    turn_prepared,
                    result,
                    options.get("hashes"),
                )
        await self._drop_if_empty(conv)
        if last_ds_error is not None:
            logger.error("run_turn exhausted key=%s err=%s", key, last_ds_error)
            raise last_ds_error
        logger.error("run_turn exhausted key=%s err=%s", key, last_error)
        raise TurnExhaustedError(last_error)

    async def stream_turn(
        self,
        key: str,
        prepared: PreparedTurn,
        **options: Unpack[StreamTurnOptions],
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn with failover across accounts.

        Yields:
            Stream deltas.

        """
        lock = self.key_lock(key)
        await lock.acquire()
        try:
            conv = await self.get_or_create(key)
            async for event in self._stream_turn_locked(
                key,
                conv,
                prepared,
                **options,
            ):
                yield event
        finally:
            lock.release()

    def _setup_failed(
        self,
        conv: Conversation,
        progress: tuple[int, int],
        account_token: str | None,
        exc: Exception,
        last_ds: DeepSeekError | None,
    ) -> DeepSeekError | None:
        """Handle a stream setup failure with rotation bookkeeping.

        Terminal moderation and rejection failures raise immediately: the
        same prompt would be flagged again on the next account.

        Returns:
            The updated last backend error.

        """
        attempt, attempts = progress
        if _is_terminal_turn_error(exc):
            poisoned = (
                account_token
                if isinstance(exc, (AccountMutedError, UploadModerationError))
                else None
            )
            self._fail_turn(conv, attempt, attempts, poisoned, exc)
            raise exc
        if isinstance(exc, DeepSeekError):
            last_ds = exc
        if account_token:
            self._pool.mark_failure(account_token)
        self._clear_session(conv, reason=str(exc))
        logger.warning(
            "stream_turn setup failed key=%s attempt=%d/%d err=%s",
            conv.id,
            attempt + 1,
            max(attempts, 1),
            exc,
        )
        return last_ds

    async def _setup_stream_attempt(
        self,
        conv: Conversation,
        prepared: PreparedTurn,
        prev_session: str | None,
    ) -> tuple[DeepSeekClient, str, PreparedTurn, list[str]]:
        """Ensure session and upload files for one stream attempt.

        Marks the account when the upload itself fails so repeated
        upload failures cool the broken account down.

        Returns:
            Tuple of client, account token, prepared turn, and file ids.

        Raises:
            DeepSeekError: If session setup or upload fails upstream.
            httpx.HTTPError: If network transport fails.
            OSError: If local I/O fails.
            ValueError: If payload validation fails.
            RuntimeError: If runtime state is invalid.

        """
        client, account, fresh = await self._ensure_session(conv)
        try:
            turn_prepared = self._resolve_prepared(
                conv,
                prepared,
                prev_session,
                fresh_session=fresh,
            )
            file_ids = await self._upload_files(client, turn_prepared)
        except (
            DeepSeekError,
            httpx.HTTPError,
            OSError,
            ValueError,
            RuntimeError,
        ):
            self._pool.mark_failure(account.token)
            raise
        return client, account.token, turn_prepared, file_ids

    @staticmethod
    def _apply_stream_event(
        event: StreamEvent,
        buffers: _StreamBuffers,
        rewriter: CitationRewriter,
    ) -> StreamEvent | None:
        """Apply one raw event to buffers and map to output.

        Returns:
            The event to yield or None to suppress.

        """
        if event.kind == "references":
            if isinstance(event.value, list):
                buffers.reference_urls.extend(
                    item
                    for item in event.value
                    if isinstance(item, str) or item is None
                )
            return None
        if event.kind == "content":
            chunk = rewriter.feed(str(event.value))
            if chunk:
                buffers.content.append(chunk)
                buffers.emitted = True
                return StreamEvent("content", chunk)
            return None
        if event.kind == "reasoning":
            buffers.reasoning.append(str(event.value))
            buffers.emitted = True
            return event
        if event.kind == "sources":
            if isinstance(event.value, list):
                buffers.sources = [
                    {key: val for key, val in item.items() if isinstance(key, str)}
                    for item in event.value
                    if isinstance(item, dict)
                ]
            return None
        if isinstance(event.value, list):
            buffers.searches.extend(event.value)
        buffers.emitted = True
        return None

    @staticmethod
    def _new_delta_req(
        client: DeepSeekClient,
        conv: Conversation,
        prompt: str,
        file_ids: list[str],
        options: StreamTurnOptions,
    ) -> _DeltaRequest:
        """Build a delta request from attempt parts.

        Returns:
            The request object.

        """
        return _DeltaRequest(
            client=client,
            conv=conv,
            prompt=prompt,
            file_ids=file_ids,
            buffers=_StreamBuffers(),
            deepthink=options["deepthink"],
            model_type=options["model_type"],
        )

    def _persist_stream_text(
        self,
        conv: Conversation,
        prompt: str,
        buffers: _StreamBuffers,
        options: StreamTurnOptions,
        account_token: str | None,
    ) -> None:
        """Record history and health for streamed text."""
        text = "".join(buffers.content)
        self._record_history(conv, prompt, text)
        self._record_prefix_turn(conv, text, options.get("hashes"))
        self._pool.mark_success(account_token or "")

    async def _iter_deltas(
        self,
        req: _DeltaRequest,
    ) -> AsyncIterator[StreamEvent]:
        """Yield deltas for one attempt.

        Yields:
            Stream deltas.

        """
        rewriter = CitationRewriter(req.buffers.reference_urls)
        async for event in self._stream_events(
            req.client,
            req.conv,
            req.prompt,
            ref_file_ids=req.file_ids,
            thinking_enabled=req.deepthink,
            model_type=req.model_type,
        ):
            mapped = self._apply_stream_event(event, req.buffers, rewriter)
            if mapped is not None:
                yield mapped
        final_chunk = rewriter.finish()
        if final_chunk:
            req.buffers.content.append(final_chunk)
            req.buffers.emitted = True
            yield StreamEvent("content", final_chunk)
        text = "".join(req.buffers.content)
        _ensure_stream_text(text)
        if req.buffers.searches:
            yield StreamEvent("search", list(req.buffers.searches))
        if req.buffers.sources:
            yield StreamEvent("sources", list(req.buffers.sources))

    def _handle_stream_error(
        self,
        conv: Conversation,
        ctx: _StreamErrorContext,
    ) -> DeepSeekError | None:
        """Handle a stream failure, raising on interruption.

        Terminal moderation and rejection failures raise immediately when
        nothing was emitted yet: retrying re-sends the flagged prompt.

        Returns:
            The updated last backend error.

        Raises:
            StreamInterruptedError: If content was already emitted.

        """
        if _is_terminal_turn_error(ctx.exc) and not ctx.emitted:
            poisoned = (
                ctx.account_token
                if isinstance(ctx.exc, (AccountMutedError, UploadModerationError))
                else None
            )
            self._fail_turn(conv, ctx.attempt, ctx.attempts, poisoned, ctx.exc)
            raise ctx.exc
        if isinstance(ctx.exc, DeepSeekError):
            ctx.last_ds = ctx.exc
        if ctx.account_token:
            self._pool.mark_failure(ctx.account_token)
        self._clear_session(conv, reason=str(ctx.exc))
        if ctx.emitted:
            raise StreamInterruptedError(ctx.exc) from ctx.exc
        logger.warning(
            "stream_turn failed key=%s attempt=%d/%d err=%s",
            conv.id,
            ctx.attempt + 1,
            max(ctx.attempts, 1),
            ctx.exc,
        )
        return ctx.last_ds

    async def _stream_turn_locked(
        self,
        key: str,
        conv: Conversation,
        prepared: PreparedTurn,
        **options: Unpack[StreamTurnOptions],
    ) -> AsyncIterator[StreamEvent]:
        """Stream one locked turn with retries.

        Yields:
            Stream deltas.

        Raises:
            CancelledError: If cancelled.
            TurnExhaustedError: If all accounts fail.

        """
        attempts = self._attempt_count(None)
        last_error: Exception | None = None
        last_ds_error: DeepSeekError | None = None
        for attempt in range(max(attempts, 1)):
            prev_session = conv.deepseek_session_id
            account_token: str | None = None
            try:
                setup = await self._setup_stream_attempt(
                    conv,
                    prepared,
                    prev_session,
                )
                client, token, turn_prepared, file_ids = setup
                account_token = token
            except asyncio.CancelledError:
                self._clear_session(conv, reason="cancelled during setup")
                raise
            except (
                DeepSeekError,
                httpx.HTTPError,
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                last_error = exc
                last_ds_error = self._setup_failed(
                    conv,
                    (attempt, attempts),
                    account_token,
                    exc,
                    last_ds_error,
                )
                continue
            req = self._new_delta_req(
                client,
                conv,
                turn_prepared.prompt,
                file_ids,
                options,
            )
            buffers = req.buffers
            try:
                async for event in self._iter_deltas(req):
                    yield event
            except asyncio.CancelledError:
                self._clear_session(conv, reason="cancelled mid-stream")
                raise
            except EmptyCompletionError as exc:
                last_error = exc
                self._clear_session(conv, reason=f"empty completion: {exc}")
                logger.warning(
                    "stream_turn empty completion key=%s attempt=%d/%d",
                    key,
                    attempt + 1,
                    max(attempts, 1),
                )
                continue
            except (
                DeepSeekError,
                httpx.HTTPError,
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                last_error = exc
                ctx = _StreamErrorContext(
                    attempt=attempt,
                    attempts=attempts,
                    account_token=account_token,
                    exc=exc,
                    emitted=buffers.emitted,
                    last_ds=last_ds_error,
                )
                last_ds_error = self._handle_stream_error(conv, ctx)
                continue
            else:
                self._persist_stream_text(
                    conv,
                    turn_prepared.prompt,
                    buffers,
                    options,
                    account_token,
                )
                return
        await self._drop_if_empty(conv)
        if last_ds_error is not None:
            logger.error("stream_turn exhausted key=%s err=%s", key, last_ds_error)
            raise last_ds_error
        logger.error("stream_turn exhausted key=%s err=%s", key, last_error)
        raise TurnExhaustedError(last_error)

    @staticmethod
    async def _upload_files(
        client: DeepSeekClient,
        prepared: PreparedTurn,
    ) -> list[str]:
        """Upload turn files and return ids.

        Returns:
            The uploaded file identifiers.

        """
        if not prepared.files:
            return []
        ids: list[str] = []
        for filename, raw, mime in prepared.files:
            is_image = (mime or "").startswith("image/") or any(
                filename.lower().endswith(ext) for ext in _IMAGE_SUFFIXES
            )
            ids.append(await client.upload_file(filename, raw, mime, vision=is_image))
        return ids

    def _record_ready_id(
        self,
        conv: Conversation,
        event: dict[str, object],
    ) -> None:
        """Persist response_message_id the moment it commits."""
        if event.get("event") == "ready":
            data = event.get("data") or {}
            rid = data.get("response_message_id") if isinstance(data, dict) else None
            if isinstance(rid, int) and rid != conv.parent_message_id:
                conv.parent_message_id = rid
                self._persist_conversation(conv)

    @staticmethod
    def _split_fragments(
        fragments: list[dict[str, object]],
    ) -> tuple[list[str], list[str]]:
        """Split fragments into reasoning and response parts.

        Returns:
            Tuple of reasoning and content part lists.

        """
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for frag in fragments:
            ftype = frag.get("type")
            content = frag.get("content")
            text = str(content) if isinstance(content, str) else ""
            if ftype == _THINK_FRAGMENT:
                reasoning_parts.append(text)
            elif ftype in _RESPONSE_FRAGMENTS:
                content_parts.append(text)
        return reasoning_parts, content_parts

    async def _collect(
        self,
        client: DeepSeekClient,
        conv: Conversation,
        prompt: str,
        **options: Unpack[CollectOptions],
    ) -> TurnResult:
        """Collect a full turn from streaming completion.

        Returns:
            The aggregated turn result.

        """
        ref_file_ids = options.get("ref_file_ids") or []
        thinking_enabled = options.get("thinking_enabled", False)
        search_enabled = options.get("search_enabled", True)
        model_type = options.get("model_type")
        agg = FragmentAggregator()
        title: str | None = None
        search_queries: list[str] = []
        async for event in client.stream_completion(
            prompt=prompt,
            chat_session_id=conv.deepseek_session_id or "",
            parent_message_id=conv.parent_message_id,
            ref_file_ids=list(ref_file_ids),
            thinking_enabled=bool(thinking_enabled),
            search_enabled=bool(search_enabled),
            model_type=model_type,
        ):
            self._record_ready_id(conv, event)
            for kind, value in agg.apply(event.get("event"), event.get("data")):
                if kind == "meta" and isinstance(value, dict) and value.get("title"):
                    title = str(value["title"])
                elif kind == "search" and isinstance(value, list):
                    search_queries.extend(str(item) for item in value if item)
        reasoning_parts, content_parts = self._split_fragments(
            list(agg.fragments),
        )
        content = rewrite_citations("".join(content_parts), agg.reference_urls)
        return TurnResult(
            content=content,
            reasoning="".join(reasoning_parts) or None,
            title=title,
            sources=list(agg.search_results),
            search_queries=search_queries,
        )

    def _record_stream_ready(
        self,
        conv: Conversation,
        event: dict[str, object],
    ) -> bool:
        """Persist ready ids for streams, reporting readiness.

        Returns:
            True when the event was ready.

        """
        if event.get("event") == "ready":
            data = event.get("data") or {}
            rid = data.get("response_message_id") if isinstance(data, dict) else None
            if isinstance(rid, int) and rid != conv.parent_message_id:
                conv.parent_message_id = rid
                self._persist_conversation(conv)
            return True
        return False

    async def _stream_events(
        self,
        client: DeepSeekClient,
        conv: Conversation,
        prompt: str,
        **options: Unpack[StreamEventsOptions],
    ) -> AsyncIterator[StreamEvent]:
        """Yield live deltas from streaming completion.

        Yields:
            Stream deltas.

        """
        ref_file_ids = options.get("ref_file_ids") or []
        thinking_enabled = options.get("thinking_enabled", False)
        model_type = options.get("model_type")
        agg = FragmentAggregator()
        announced_refs = 0
        announced_sources = 0
        async for event in client.stream_completion(
            prompt=prompt,
            chat_session_id=conv.deepseek_session_id or "",
            parent_message_id=conv.parent_message_id,
            ref_file_ids=list(ref_file_ids),
            thinking_enabled=bool(thinking_enabled),
            search_enabled=True,
            model_type=model_type,
        ):
            if self._record_stream_ready(conv, event):
                continue
            for kind, value in agg.apply(event.get("event"), event.get("data")):
                if kind in _CONTENT_KINDS:
                    if value:
                        yield StreamEvent(kind, value)
                elif kind == "search":
                    yield StreamEvent("search", value)
            if len(agg.reference_urls) > announced_refs:
                new_refs = agg.reference_urls[announced_refs:]
                announced_refs = len(agg.reference_urls)
                yield StreamEvent("references", list[object](new_refs))
            if len(agg.search_results) > announced_sources:
                announced_sources = len(agg.search_results)
                yield StreamEvent("sources", list(agg.search_results))

    def _record_history(self, conv: Conversation, prompt: str, answer: str) -> None:
        """Append a turn and trim history to the char budget."""
        conv.history.append({"role": "user", "content": prompt})
        conv.history.append({"role": "assistant", "content": answer})
        total = sum(len(item["content"]) for item in conv.history)
        while total > MAX_HISTORY_CHARS and len(conv.history) > _MIN_HISTORY_KEEP:
            removed = conv.history.pop(0)
            total -= len(removed["content"])
        conv.last_used_at = time.time()
        self._persist_conversation(conv)

    def transcript(self, key: str) -> list[dict[str, str]]:
        """Return the transcript for a key.

        Returns:
            The stored history.

        """
        conv = self._conversations.get(key)
        if conv:
            return list(conv.history)
        if self._storage is not None:
            data = self._storage.get_conversation(key)
            if data and data.get("history"):
                return list(data["history"])
        return []

    async def reset(self, key: str) -> None:
        """Reset one conversation and delete its remote session."""
        async with self.key_lock(key):
            async with self._lock:
                conv = self._conversations.pop(key, None)
            if self._storage is not None:
                self._storage.delete_conversation(key)
        if conv and conv.account_token and conv.deepseek_session_id:
            client = self._clients.get(conv.account_token)
            if client:
                await client.delete_session(conv.deepseek_session_id)

    def _stale_keys(self, now: float) -> list[str]:
        """Collect keys idle beyond the session TTL.

        Returns:
            The stale conversation keys.

        """
        return [
            key
            for key, value in self._conversations.items()
            if now - value.last_used_at > SESSION_TTL
        ]

    async def _sweep_once(self) -> None:
        """Sweep one batch of stale conversations."""
        await asyncio.sleep(SWEEP_INTERVAL)
        now = time.time()
        async with self._lock:
            stale = self._stale_keys(now)
        for key in stale:
            await self.reset(key)

    async def _sweep_loop(self) -> None:
        """Run the sweeper until cancelled."""
        while True:
            try:
                await self._sweep_once()
            except asyncio.CancelledError:
                return
            except (OSError, ValueError, RuntimeError, httpx.HTTPError):
                logger.exception("sweeper failed")
                continue


def new_conversation_key() -> str:
    """Create a fresh conversation key.

    Returns:
        The new key.

    """
    return uuid.uuid4().hex

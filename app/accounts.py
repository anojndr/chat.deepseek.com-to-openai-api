# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Account store: parse accounts.txt (browser localStorage dumps) and rotate tokens.

Format: one or more blocks separated by `account N` headers, each containing a
JSON object that mirrors chat.deepseek.com localStorage. Only `userToken.value`
is required; every other key is ignored so new exports keep working.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

# Upper bound for exponential backoff cooldowns.
_MAX_COOLDOWN_S = 900.0
# Cap for the backoff exponent shift.
_MAX_SHIFT = 5
# Default cooldown applied per failure.
_DEFAULT_FAILURE_COOLDOWN_S = 30.0
# Base for exponential backoff multiplication.
_BACKOFF_BASE = 2


class AccountsParseError(ValueError):
    """Raised when an accounts.txt block lacks a usable token."""

    def __init__(self, block: int) -> None:
        """Create an error for a 1-based block number."""
        super().__init__(
            f"accounts.txt block {block}: no usable userToken.value",
        )
        self.block = block


class AccountsEmptyError(ValueError):
    """Raised when no account blocks are found."""

    def __init__(self, path: Path) -> None:
        """Create an error for the accounts file path."""
        super().__init__(
            f"accounts.txt at {path} contains no account blocks",
        )
        self.path = path


class AccountSnapshot(TypedDict):
    """Snapshot of account health for status endpoints."""

    account: int
    healthy: bool
    cooldown_remaining_s: float
    consecutive_failures: int


class Account:
    """Represent a single DeepSeek account token with backoff state."""

    __slots__ = ("disabled_until", "failures", "index", "token")

    def __init__(self, index: int, token: str) -> None:
        """Create an account entry."""
        self.index = index
        self.token = token
        self.disabled_until = 0.0
        self.failures = 0

    @property
    def available(self) -> bool:
        """Check whether the account is outside its cooldown window.

        Returns:
            True when the account may be used immediately.

        """
        return time.monotonic() >= self.disabled_until

    def mark_failure(self, cooldown: float) -> None:
        """Record a failure and apply exponential backoff."""
        self.failures += 1
        # Cap reached at failures == 6; clamp the exponent so 2**(n-1) can
        # never overflow float conversion during prolonged outages.
        shift = min(self.failures - 1, _MAX_SHIFT)
        self.disabled_until = time.monotonic() + min(
            cooldown * (_BACKOFF_BASE**shift),
            _MAX_COOLDOWN_S,
        )

    def mark_success(self) -> None:
        """Reset failure counters after a successful request."""
        self.failures = 0
        self.disabled_until = 0.0


def _extract_token(raw_token: object) -> str | None:
    """Extract the token string from a userToken value.

    Returns:
        Token string, or None when the value is unusable.

    """
    if isinstance(raw_token, str):
        try:
            parsed: object = json.loads(raw_token)
        except json.JSONDecodeError:
            return raw_token or None
        if isinstance(parsed, dict):
            candidate = parsed.get("value")
            if isinstance(candidate, str):
                return candidate or None
        return raw_token or None
    if isinstance(raw_token, dict):
        value = raw_token.get("value")
        if isinstance(value, str):
            return value or None
        return None
    return None


def parse_accounts(path: Path) -> list[Account]:
    """Parse account tokens from an accounts.txt dump.

    Returns:
        List of accounts in file order.

    Raises:
        AccountsParseError: If a block lacks a usable token.
        AccountsEmptyError: If no account blocks are found.

    """
    text = path.read_text(encoding="utf-8")
    header = re.compile(r"(?im)^account\s*\d+\s*:?\s*$")
    starts = [m.start() for m in header.finditer(text)]
    if not starts and text.lstrip().startswith("{"):
        starts = [0]
    accounts: list[Account] = []
    decoder = json.JSONDecoder()
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(text)
        body = text[start:end]
        brace = body.find("{")
        if brace == -1:
            continue
        decoded: object = decoder.raw_decode(body[brace:])[0]
        if not isinstance(decoded, dict):
            raise AccountsParseError(n + 1)
        token = _extract_token(decoded.get("userToken"))
        if token is None:
            raise AccountsParseError(n + 1)
        accounts.append(Account(index=len(accounts), token=token))
    if not accounts:
        raise AccountsEmptyError(path)
    return accounts


class AccountPool:
    """Round-robin over any number of accounts; unhealthy ones are skipped."""

    def __init__(
        self,
        path: Path,
        failure_cooldown: float = _DEFAULT_FAILURE_COOLDOWN_S,
    ) -> None:
        """Create a pool backed by the given accounts file."""
        self._path = path
        self._failure_cooldown = failure_cooldown
        self._lock = threading.Lock()
        self._accounts: list[Account] = []
        self._cursor = 0
        self.reload()

    def reload(self) -> int:
        """Reload accounts from disk while preserving health state.

        Returns:
            Number of accounts loaded.

        """
        fresh = parse_accounts(self._path)
        with self._lock:
            old = {a.token: a for a in self._accounts}
            for account in fresh:
                prev = old.get(account.token)
                if prev:
                    account.disabled_until = prev.disabled_until
                    account.failures = prev.failures
            self._accounts = fresh
            self._cursor %= max(len(fresh), 1)
        return len(fresh)

    def acquire(self) -> Account:
        """Acquire the next available account in round-robin order.

        Returns:
            Next healthy account, or least-recently-failed one.

        """
        with self._lock:
            pool = self._accounts
            count = len(pool)
            for i in range(count):
                candidate = pool[(self._cursor + i) % count]
                if candidate.available:
                    self._cursor = (self._cursor + i + 1) % count
                    return candidate
            # All cooling down: return least-recently-failed rather
            # than fail hard.
            return min(pool, key=lambda item: item.disabled_until)

    def by_token(self, token: str | None) -> Account | None:
        """Resolve an account by its token (stable across reloads).

        Returns:
            Matching account, or None when absent.

        """
        if token is None:
            return None
        with self._lock:
            for account in self._accounts:
                if account.token == token:
                    return account
        return None

    def mark_success(self, token: str) -> None:
        """Mark the account matching the token as successful."""
        with self._lock:
            for account in self._accounts:
                if account.token == token:
                    account.mark_success()
                    return

    def mark_failure(self, token: str) -> None:
        """Mark the account matching the token as failed."""
        with self._lock:
            for account in self._accounts:
                if account.token == token:
                    account.mark_failure(self._failure_cooldown)
                    return

    @property
    def size(self) -> int:
        """Number of accounts in the pool.

        Returns:
            Count of loaded accounts.

        """
        with self._lock:
            return len(self._accounts)

    def tokens(self) -> list[str]:
        """List all account tokens in pool order.

        Returns:
            Token strings for every loaded account.

        """
        with self._lock:
            return [account.token for account in self._accounts]

    def snapshot(self) -> list[AccountSnapshot]:
        """Capture health state for every account.

        Returns:
            Per-account health dictionaries.

        """
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "account": account.index,
                    "healthy": account.available,
                    "cooldown_remaining_s": round(
                        max(0.0, account.disabled_until - now),
                        1,
                    ),
                    "consecutive_failures": account.failures,
                }
                for account in self._accounts
            ]

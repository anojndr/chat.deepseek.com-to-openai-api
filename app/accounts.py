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
from pathlib import Path


class Account:
    __slots__ = ("index", "token", "disabled_until", "failures")

    def __init__(self, index: int, token: str) -> None:
        self.index = index
        self.token = token
        self.disabled_until = 0.0
        self.failures = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.disabled_until

    def mark_failure(self, cooldown: float) -> None:
        self.failures += 1
        # cap reached at failures == 6; clamp the exponent so 2**(n-1) can
        # never overflow float conversion during prolonged outages
        shift = min(self.failures - 1, 5)
        self.disabled_until = time.monotonic() + min(cooldown * (2**shift), 900.0)

    def mark_success(self) -> None:
        self.failures = 0
        self.disabled_until = 0.0


def parse_accounts(path: Path) -> list[Account]:
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
        obj, _ = decoder.raw_decode(body[brace:])
        token = None
        raw_token = obj.get("userToken")
        if isinstance(raw_token, str):
            try:
                token = json.loads(raw_token).get("value")
            except json.JSONDecodeError:
                token = None
            if not isinstance(token, str):
                token = raw_token if raw_token else None
        elif isinstance(raw_token, dict):
            token = raw_token.get("value")
        if not token or not isinstance(token, str):
            raise ValueError(f"accounts.txt block {n + 1}: no usable userToken.value")
        accounts.append(Account(index=len(accounts), token=token))
    if not accounts:
        raise ValueError(f"accounts.txt at {path} contains no account blocks")
    return accounts


class AccountPool:
    """Round-robin over any number of accounts; unhealthy ones are skipped."""

    def __init__(self, path: Path, failure_cooldown: float = 30.0) -> None:
        self._path = path
        self._failure_cooldown = failure_cooldown
        self._lock = threading.Lock()
        self._accounts: list[Account] = []
        self._cursor = 0
        self.reload()

    def reload(self) -> int:
        fresh = parse_accounts(self._path)
        with self._lock:
            old = {a.token: a for a in self._accounts}
            for a in fresh:
                prev = old.get(a.token)
                if prev:
                    a.disabled_until = prev.disabled_until
                    a.failures = prev.failures
            self._accounts = fresh
            self._cursor %= max(len(fresh), 1)
        return len(fresh)

    def acquire(self) -> Account:
        with self._lock:
            pool = self._accounts
            n = len(pool)
            for i in range(n):
                candidate = pool[(self._cursor + i) % n]
                if candidate.available:
                    self._cursor = (self._cursor + i + 1) % n
                    return candidate
            # all cooling down: return least-recently-failed rather than fail hard
            best = min(pool, key=lambda a: a.disabled_until)
            return best

    def by_token(self, token: str | None) -> Account | None:
        """Resolve an account by its token (stable across reloads)."""
        if token is None:
            return None
        with self._lock:
            for a in self._accounts:
                if a.token == token:
                    return a
        return None

    def mark_success(self, token: str) -> None:
        with self._lock:
            for a in self._accounts:
                if a.token == token:
                    a.mark_success()
                    return

    def mark_failure(self, token: str) -> None:
        with self._lock:
            for a in self._accounts:
                if a.token == token:
                    a.mark_failure(self._failure_cooldown)
                    return

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._accounts)

    def tokens(self) -> list[str]:
        with self._lock:
            return [a.token for a in self._accounts]

    def snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "account": a.index,
                    "healthy": a.available,
                    "cooldown_remaining_s": round(max(0.0, a.disabled_until - now), 1),
                    "consecutive_failures": a.failures,
                }
                for a in self._accounts
            ]

# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""DeepSeekHashV1 proof-of-work solver.

Runs the official sha3_wasm_bg.wasm module (same bytes the web client loads)
through wasmtime. The wasm-bindgen ABI is: allocate stack, write challenge and
prefix strings via __wbindgen_export_0(len, 1), call
wasm_solve(retptr, chal_ptr, chal_len, prefix_ptr, prefix_len, difficulty),
then read i32 status at retptr+0 and f64 answer at retptr+8. Status 0 means no
solution; any other value means `answer` holds the solution.
"""

from __future__ import annotations

import struct
import threading
from pathlib import Path

import wasmtime

_WASM_PATH = Path(__file__).resolve().parent / "vendor" / "sha3_wasm_bg.wasm"
_MEMORY_EXPORT = "memory"
_MEMORY_KIND = "Memory"
_SOLVE_EXPORT = "wasm_solve"
_STACK_EXPORT = "__wbindgen_add_to_stack_pointer"
_ALLOC_EXPORT = "__wbindgen_export_0"
_FUNC_KIND = "Func"
_STACK_WHAT = "stack pointer"
_ALLOC_WHAT = "allocator"


class WasmExportError(TypeError):
    """Raised when the PoW wasm module exports an unexpected value."""

    def __init__(self, name: str, actual: str, expected: str) -> None:
        """Store the export details."""
        message = f"wasm export {name!r} is {actual}, expected {expected}"
        super().__init__(message)
        self.name = name
        self.actual = actual
        self.expected = expected


class WasmAbiError(TypeError):
    """Raised when the PoW wasm ABI returns an unexpected value."""

    def __init__(self, what: str, actual: str) -> None:
        """Store the ABI details."""
        message = f"wasm {what} returned {actual}, expected int"
        super().__init__(message)
        self.what = what
        self.actual = actual


def _expire_str(expire_at: str | float) -> str:
    """Format a PoW expiry exactly like the DeepSeek web client.

    Args:
        expire_at: Raw `expire_at` value from the challenge payload. Integer
            millisecond timestamps arrive here as `int` (assignable to `float`
            per the numeric tower) and MUST stay integer-formatted: the web
            client stringifies `1789193300957` without a trailing `.0`, and the
            wasm hash only matches with that exact prefix.

    Returns:
        String form used in the `salt_expire_` hash prefix.

    """
    if isinstance(expire_at, str):
        return expire_at
    if isinstance(expire_at, int):
        return str(expire_at)
    if expire_at.is_integer():
        return str(int(expire_at))
    return str(expire_at)


class PowSolver:
    """Solve DeepSeekHashV1 proof-of-work challenges with wasmtime."""

    _store: wasmtime.Store
    _memory: wasmtime.Memory
    _solve: wasmtime.Func
    _stack: wasmtime.Func
    _alloc: wasmtime.Func
    _lock: threading.Lock

    def __init__(self) -> None:
        """Load the wasm module and bind its exports.

        Raises:
            WasmExportError: If a wasm export has an unexpected type.

        """
        store = wasmtime.Store()
        with _WASM_PATH.open("rb") as fh:
            wasm_bytes = fh.read()
        instance = wasmtime.Instance(
            store,
            wasmtime.Module(store.engine, wasm_bytes),
            [],
        )
        exports = instance.exports(store)
        memory = exports[_MEMORY_EXPORT]
        if not isinstance(memory, wasmtime.Memory):
            raise WasmExportError(
                _MEMORY_EXPORT,
                type(memory).__name__,
                _MEMORY_KIND,
            )
        solve = exports[_SOLVE_EXPORT]
        if not isinstance(solve, wasmtime.Func):
            raise WasmExportError(_SOLVE_EXPORT, type(solve).__name__, _FUNC_KIND)
        stack = exports[_STACK_EXPORT]
        if not isinstance(stack, wasmtime.Func):
            raise WasmExportError(_STACK_EXPORT, type(stack).__name__, _FUNC_KIND)
        alloc = exports[_ALLOC_EXPORT]
        if not isinstance(alloc, wasmtime.Func):
            raise WasmExportError(_ALLOC_EXPORT, type(alloc).__name__, _FUNC_KIND)
        self._store = store
        self._memory = memory
        self._solve = solve
        self._stack = stack
        self._alloc = alloc
        # wasmtime Store/Instance/Memory are NOT thread-safe; solve() runs on
        # worker threads via asyncio.to_thread, so serialize all wasm access.
        self._lock = threading.Lock()

    def solve(
        self,
        challenge_hex: str,
        salt: str,
        expire_at: str | float,
        difficulty: float,
    ) -> int | None:
        """Return the integer answer for a DeepSeekHashV1 challenge.

        Args:
            challenge_hex: Hex challenge string from the backend.
            salt: Challenge salt from the backend.
            expire_at: Raw `expire_at` value; integer timestamps stay
                integer-formatted via `_expire_str` so the wasm prefix matches
                the web client (no trailing `.0`).
            difficulty: PoW difficulty from the backend.

        Returns:
            int | None: Solution integer, or None when wasm reports no answer.

        Raises:
            WasmAbiError: If the wasm stack pointer or allocator misbehaves.

        """
        challenge_bytes = challenge_hex.encode()
        prefix_bytes = f"{salt}_{_expire_str(expire_at)}_".encode()

        # One wasm call at a time: the Store, its linear memory and the stack
        # pointer are shared mutable state (solve runs on worker threads).
        with self._lock:
            store = self._store
            ret_ptr_raw = self._stack(store, -16)
            if not isinstance(ret_ptr_raw, int):
                raise WasmAbiError(_STACK_WHAT, type(ret_ptr_raw).__name__)
            ret_ptr = ret_ptr_raw
            try:
                c_ptr_raw = self._alloc(store, len(challenge_bytes), 1)
                if not isinstance(c_ptr_raw, int):
                    raise WasmAbiError(_ALLOC_WHAT, type(c_ptr_raw).__name__)
                c_ptr = c_ptr_raw
                self._memory.write(store, challenge_bytes, c_ptr)
                p_ptr_raw = self._alloc(store, len(prefix_bytes), 1)
                if not isinstance(p_ptr_raw, int):
                    raise WasmAbiError(_ALLOC_WHAT, type(p_ptr_raw).__name__)
                p_ptr = p_ptr_raw
                self._memory.write(store, prefix_bytes, p_ptr)
                self._solve(
                    store,
                    ret_ptr,
                    c_ptr,
                    len(challenge_bytes),
                    p_ptr,
                    len(prefix_bytes),
                    float(difficulty),
                )
                raw = bytes(self._memory.read(store, ret_ptr, ret_ptr + 16))
            finally:
                self._stack(store, 16)

            status = int.from_bytes(raw[0:4], "little")
            if status == 0:
                return None
            return int(struct.unpack("<d", raw[8:16])[0])

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

_WASM_PATH = Path(__file__).resolve().parent / "vendor" / "sha3_wasm_bg.wasm"


class PowSolver:
    def __init__(self) -> None:
        from wasmtime import Instance, Module, Store

        self._store = Store()
        with _WASM_PATH.open("rb") as fh:
            wasm_bytes = fh.read()
        instance = Instance(self._store, Module(self._store.engine, wasm_bytes), [])
        exports = instance.exports(self._store)
        self._memory = exports["memory"]
        self._solve = exports["wasm_solve"]
        self._stack = exports["__wbindgen_add_to_stack_pointer"]
        self._alloc = exports["__wbindgen_export_0"]
        # wasmtime Store/Instance/Memory are NOT thread-safe; solve() runs on
        # worker threads via asyncio.to_thread, so serialize all wasm access.
        self._lock = threading.Lock()

    def solve(
        self, challenge_hex: str, salt: str, expire_at, difficulty: float | int
    ) -> int | None:
        """Return the integer answer for a DeepSeekHashV1 challenge, or None."""
        challenge_bytes = challenge_hex.encode()
        prefix_bytes = f"{salt}_{expire_at}_".encode()

        # One wasm call at a time: the Store, its linear memory and the stack
        # pointer are shared mutable state (solve runs on worker threads).
        with self._lock:
            store = self._store
            ret_ptr = self._stack(store, -16)
            try:
                c_ptr = self._alloc(store, len(challenge_bytes), 1)
                self._memory.write(store, challenge_bytes, c_ptr)
                p_ptr = self._alloc(store, len(prefix_bytes), 1)
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

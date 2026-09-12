# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Bootstrap installer for the vendored wasmtime wheel."""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
REQUIRED = ("fastapi", "httpx", "uvicorn", "pydantic")

_X86_64_ALIASES = {"x86_64", "amd64"}
_PIP_BREAK_FLAG = "--break-system-packages"
_PIP_BREAK_MIN_VERSION: tuple[int, int] = (23, 0)
_EXIT_OK = 0
_VERSION_PARTS = 3


def _have(mod: str) -> bool:
    """Check whether a module is importable.

    Args:
        mod: Module name to probe.

    Returns:
        True when the module can be found.

    """
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _pip_version_tuple() -> tuple[int, int, int]:
    """Return the installed pip version as a triple.

    Returns:
        Parsed version triple, zeros when unknown.

    """
    try:
        raw = importlib.metadata.version("pip")
    except importlib.metadata.PackageNotFoundError:
        return (0, 0, 0)
    parts: list[int] = []
    for piece in raw.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) >= _VERSION_PARTS:
            break
    while len(parts) < _VERSION_PARTS:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _pip_supports(flag: str) -> bool:
    """Check whether the installed pip supports a flag.

    Args:
        flag: Pip flag to probe.

    Returns:
        True when the flag is supported.

    """
    if flag != _PIP_BREAK_FLAG:
        return False
    major, minor, _micro = _pip_version_tuple()
    return (major, minor) >= _PIP_BREAK_MIN_VERSION


def _pip_install(args: list[str]) -> bool:
    """Install packages by spawning the pip module.

    Args:
        args: Arguments appended after `pip`.

    Returns:
        True when pip reports success.

    """
    cmd: list[str] = [sys.executable, "-m", "pip", *args]
    if hasattr(os, "posix_spawn"):
        try:
            pid: int = os.posix_spawn(sys.executable, cmd, dict(os.environ))
        except OSError as exc:
            logger.warning("pip spawn failed: %s", exc)
            return False
        try:
            _, status = os.waitpid(pid, 0)
        except OSError as exc:
            logger.warning("pip wait failed: %s", exc)
            return False
        return os.waitstatus_to_exitcode(status) == _EXIT_OK
    # Windows fallback without posix_spawn: run pip with full path and list
    # args, no shell. Dynamic import defers the subprocess dependency to
    # the fallback path.
    subprocess_mod = importlib.import_module("subprocess")
    try:
        subprocess_mod.check_call(cmd)
    except OSError as exc:
        logger.warning("pip spawn failed: %s", exc)
        return False
    except subprocess_mod.CalledProcessError as exc:
        logger.warning("pip install failed: %s", exc)
        return False
    return True


def ensure_wasmtime() -> None:
    """Ensure the wasmtime runtime is installed."""
    if _have("wasmtime"):
        return
    machine = platform.machine().lower()
    if machine not in _X86_64_ALIASES:
        logger.info("installing wasmtime from PyPI for %s", machine)
        if not _pip_install(["install", "wasmtime"]):
            sys.exit("[setup] failed to install wasmtime from PyPI")
        return
    wheel = ROOT / "vendor" / "wasmtime-48.0.0-py3-none-manylinux1_x86_64.whl"
    if not wheel.exists():
        sys.exit(
            "wasmtime is required for the DeepSeek proof-of-work solver.\n"
            f"Expected vendored wheel at {wheel} (or run: pip install wasmtime)",
        )
    logger.info("installing vendored wasmtime")
    attempts: list[list[str]] = [["install", str(wheel)]]
    if _pip_supports(_PIP_BREAK_FLAG):
        attempts.insert(0, ["install", _PIP_BREAK_FLAG, str(wheel)])
    for extra in attempts:
        if _pip_install(["--quiet", *extra]):
            return
    sys.exit(
        "[setup] failed to install wasmtime wheel.\n"
        "Install it manually: pip install wasmtime",
    )


def check_runtime_deps() -> None:
    """Log any missing runtime dependencies."""
    missing = [m for m in REQUIRED if not _have(m)]
    if missing:
        joined = ", ".join(missing)
        logger.warning(
            "missing required dependencies: %s (pip install %s)",
            joined,
            joined,
        )


def bootstrap() -> None:
    """Ensure wasmtime and fail fast when required deps are absent."""
    ensure_wasmtime()
    check_runtime_deps()
    for mod in REQUIRED:
        if not _have(mod):
            sys.exit(f"[setup] required dependency '{mod}' is missing; aborting")


if __name__ == "__main__":
    bootstrap()

    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("wasmtime")

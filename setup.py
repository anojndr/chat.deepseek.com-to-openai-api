#!/usr/bin/env python3
"""Bootstrap: install the vendored wasmtime wheel and check runtime deps."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ("fastapi", "httpx", "uvicorn", "pydantic")


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _pip_supports(flag: str) -> bool:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return flag in (out.stdout + out.stderr)
    except Exception:
        return False


def ensure_wasmtime() -> None:
    try:
        import wasmtime  # noqa: F401

        return
    except Exception:
        pass
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        # vendored wheel is x86_64-only; PyPI publishes aarch64 wheels
        print(f"[setup] {machine}: installing wasmtime from PyPI")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wasmtime"])
        return
    wheel = ROOT / "vendor" / "wasmtime-48.0.0-py3-none-manylinux1_x86_64.whl"
    if not wheel.exists():
        sys.exit(
            "wasmtime is required for the DeepSeek proof-of-work solver.\n"
            f"Expected vendored wheel at {wheel} (or run: pip install wasmtime)"
        )
    print("[setup] installing vendored wasmtime")
    supports_flag = _pip_supports("--break-system-packages")
    attempts = [["install", str(wheel)]]
    if supports_flag:
        attempts.insert(0, ["install", "--break-system-packages", str(wheel)])
    last_error: Exception | None = None
    for extra in attempts:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--quiet", *extra])
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
    sys.exit(
        f"[setup] failed to install wasmtime wheel ({last_error}).\n"
        "Install it manually: pip install wasmtime"
    )


def check_runtime_deps() -> None:
    missing = [m for m in REQUIRED if not _have(m)]
    if missing:
        print(
            f"[setup] missing required dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


def bootstrap() -> None:
    ensure_wasmtime()
    check_runtime_deps()
    # fail fast (rather than mid-request) when a required dep is absent
    for mod in REQUIRED:
        if not _have(mod):
            sys.exit(f"[setup] required dependency '{mod}' is missing; aborting")


if __name__ == "__main__":
    bootstrap()

    try:
        importlib.metadata.version("wasmtime")
    except importlib.metadata.PackageNotFoundError:
        pass  # already handled by ensure_wasmtime above

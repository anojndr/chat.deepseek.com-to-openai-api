#!/usr/bin/env python3
"""Bootstrap: install local vendored deps (wasmtime) if missing."""
import importlib.util
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def ensure_wasmtime() -> None:
    try:
        import wasmtime  # noqa: F401
        return
    except Exception:
        pass
    wheel = ROOT / "vendor" / "wasmtime-48.0.0-py3-none-manylinux1_x86_64.whl"
    if not wheel.exists():
        sys.exit(
            "wasmtime is required for the DeepSeek proof-of-work solver.\n"
            f"Expected vendored wheel at {wheel} (or run: pip install wasmtime)"
        )
    target = sysconfig.get_paths()["purelib"]
    print("[setup] installing vendored wasmtime")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", str(wheel)]
    )


if __name__ == "__main__":
    ensure_wasmtime()
    for mod in ("fastapi", "httpx", "uvicorn", "pydantic"):
        if not have(mod):
            print(f"[setup] missing dependency: {mod}")

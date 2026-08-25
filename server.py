"""Entry point: uvicorn chat_deepseek_api.app:app --port 34868"""

import sys
from pathlib import Path

# allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup import ensure_wasmtime  # noqa: E402

ensure_wasmtime()

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=34868, log_level="info")

# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Entry point: python3 server.py — serves app.main:app.

Binds to HOST (default 127.0.0.1) on PORT (default 34868).
Set API_KEY to require `Authorization: Bearer <key>` on generation/admin routes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from setup import bootstrap

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Default bind address when HOST is unset.
_DEFAULT_HOST = "127.0.0.1"
# Default port when PORT is unset.
_DEFAULT_PORT = 34868


def _bootstrap() -> None:
    """Run repository bootstrap checks."""
    bootstrap()


_bootstrap()


def main() -> None:
    """Launch the API server."""
    host = os.environ.get("HOST", _DEFAULT_HOST)
    port = int(os.environ.get("PORT", str(_DEFAULT_PORT)))
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

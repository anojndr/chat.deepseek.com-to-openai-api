"""Entry point: python3 server.py — serves app.main:app.

Binds to HOST (default 127.0.0.1) on PORT (default 34868).
Set API_KEY to require `Authorization: Bearer <key>` on generation/admin routes.
"""

import os
import sys
from pathlib import Path

# allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup import bootstrap  # noqa: E402

bootstrap()

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "34868")),
        log_level="info",
    )

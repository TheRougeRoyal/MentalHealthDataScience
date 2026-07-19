"""Vercel serverless entrypoint for the MHRAS FastAPI backend.

Strips a leading ``/api`` so production frontend paths (``/api/health``)
map onto the same routes used by local uvicorn (``/health``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.app import app as _mhras_app


class _StripApiPrefix:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path == "/api" or path.startswith("/api/"):
                new_path = path[4:] or "/"
                scope = dict(scope)
                scope["path"] = new_path
                if "raw_path" in scope:
                    scope["raw_path"] = new_path.encode("utf-8")
        await self.app(scope, receive, send)


app = _StripApiPrefix(_mhras_app)

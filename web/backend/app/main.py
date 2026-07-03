"""FastAPI application factory.

In development the React app runs on the Vite dev server and proxies ``/api`` to
this backend (CORS is opened for localhost). In production the built frontend can
be dropped into ``web/frontend/dist`` and this app serves it as static files with
an SPA fallback — the mount is added only if that directory exists.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import api_router
from .config import REPO_ROOT

_FRONTEND_DIST = REPO_ROOT / "web" / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(
        title="CityBehavEx Web API",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    app.include_router(api_router)

    if _FRONTEND_DIST.is_dir():
        _mount_frontend(app, _FRONTEND_DIST)

    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"

    @app.exception_handler(404)
    async def spa_fallback(request: Request, exc):  # noqa: ANN001
        if request.url.path.startswith("/api") or not index.exists():
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(index)

    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


app = create_app()

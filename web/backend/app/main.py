"""FastAPI application factory.

In development the React app runs on the Vite dev server and proxies ``/api`` to
this backend (CORS is opened for localhost). In production the built frontend can
be dropped into ``web/frontend/dist`` and this app serves it as static files with
an SPA fallback — the mount is added only if that directory exists.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.routing import APIRoute, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.charts import router as charts_router
from .api.experiments import router as experiments_router
from .api.timeline import router as timeline_router
from .config import REPO_ROOT
from .executor import init_executor, shutdown_executor

_FRONTEND_DIST = REPO_ROOT / "web" / "frontend" / "dist"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_executor()
    try:
        yield
    finally:
        shutdown_executor()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CityBehavEx Web API",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    _include_router_flat(app, experiments_router, prefix="/api")
    _include_router_flat(app, charts_router, prefix="/api")
    _include_router_flat(app, timeline_router, prefix="/api")

    if _FRONTEND_DIST.is_dir():
        _mount_frontend(app, _FRONTEND_DIST)

    return app


def _include_router_flat(app: FastAPI, router: APIRouter, *, prefix: str = "") -> None:
    """Register router routes eagerly.

    FastAPI 0.139 stores included routers as deferred ``_IncludedRouter``
    entries. In this app/version combination those placeholders are not being
    matched by Starlette, so register the concrete API routes directly.
    """
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        app.add_api_route(
            f"{prefix}{route.path}",
            route.endpoint,
            methods=route.methods,
            name=route.name,
            response_model=route.response_model,
            status_code=route.status_code,
            tags=route.tags,
            dependencies=route.dependencies,
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=route.responses,
            deprecated=route.deprecated,
            operation_id=route.operation_id,
            include_in_schema=route.include_in_schema,
            response_class=route.response_class,
            openapi_extra=route.openapi_extra,
        )


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"

    @app.exception_handler(404)
    async def spa_fallback(request: Request, exc):  # noqa: ANN001
        if request.url.path.startswith("/api") or not index.exists():
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(index)

    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")


app = create_app()

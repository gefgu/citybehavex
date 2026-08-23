"""Aggregate all API routers under the ``/api`` prefix."""

from __future__ import annotations

from fastapi import APIRouter

from .charts import router as charts_router
from .experiments import router as experiments_router
from .timeline import router as timeline_router

api_router = APIRouter(prefix="/api")
api_router.include_router(experiments_router)
api_router.include_router(charts_router)
api_router.include_router(timeline_router)

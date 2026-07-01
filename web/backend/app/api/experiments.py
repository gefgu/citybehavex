"""Experiment discovery endpoints (from ``configs/*.yaml``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..experiments import get_experiment, list_experiments
from ..models import ApiResponseWrapper

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def get_experiments(
    with_summary: bool = Query(False, description="Include per-run parquet row/user/date metadata"),
) -> ApiResponseWrapper[list[dict[str, Any]]]:
    return ApiResponseWrapper(data=[e.to_dict(with_summary=with_summary) for e in list_experiments()])


@router.get("/experiments/{exp_id}")
def get_experiment_detail(exp_id: str) -> ApiResponseWrapper[dict[str, Any]]:
    experiment = get_experiment(exp_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}")
    return ApiResponseWrapper(data=experiment.to_dict(with_summary=True))

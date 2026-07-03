"""Experiment discovery endpoints (from ``configs/*.yaml``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from ..experiments import (
    ExperimentMutationError,
    archive_experiment,
    delete_run,
    get_experiment,
    list_experiments,
    update_experiment,
)
from ..models import ApiResponseWrapper

router = APIRouter(tags=["experiments"])


class ExperimentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    agents: int | None = None
    days: int | None = None
    start_date: str | None = None
    granularity_minutes: int | None = None
    car_speed_kmh: float | None = None
    simulation_output: str | None = None
    observed_path: str | None = None
    profiles_enabled: bool | None = None
    profiles_output: str | None = None


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


@router.patch("/experiments/{exp_id}")
def patch_experiment(exp_id: str, payload: ExperimentUpdate) -> ApiResponseWrapper[dict[str, Any]]:
    updates = payload.model_dump(exclude_unset=True)
    try:
        experiment = update_experiment(exp_id, updates)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}") from None
    except ExperimentMutationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ApiResponseWrapper(data=experiment.to_dict(with_summary=True))


@router.post("/experiments/{exp_id}/archive")
def archive_experiment_config(exp_id: str) -> ApiResponseWrapper[dict[str, str]]:
    try:
        archived_path = archive_experiment(exp_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}") from None
    except ExperimentMutationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ApiResponseWrapper(data={"archived_config": str(archived_path.name)})


@router.delete("/experiments/{exp_id}/runs/{run_id}")
def delete_experiment_run(exp_id: str, run_id: str) -> ApiResponseWrapper[dict[str, Any]]:
    try:
        deleted = delete_run(exp_id, run_id)
    except FileNotFoundError:
        experiment = get_experiment(exp_id)
        detail = f"unknown experiment {exp_id!r}" if experiment is None else f"unknown run {run_id!r}"
        raise HTTPException(status_code=404, detail=detail) from None
    return ApiResponseWrapper(data={"deleted": [str(path) for path in deleted]})

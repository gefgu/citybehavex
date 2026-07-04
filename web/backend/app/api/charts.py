"""Comparison-payload endpoint — the JSON that replaces the standalone HTML."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..cache import get_or_build
from ..experiments import get_experiment
from ..home_work_data import DemoFilter, GENDERS, JOBS, get_or_build_home_work, resolve_age_bracket
from ..models import ApiResponseWrapper
from ..payload import build_comparison_payload

router = APIRouter(tags=["charts"])


@router.get("/experiments/{exp_id}/charts")
def get_charts(
    exp_id: str,
    run: Optional[str] = Query(None, description="Run id (timestamp). Defaults to the latest run."),
    refresh: bool = Query(False, description="Bypass the cache and recompute."),
) -> ApiResponseWrapper[dict[str, Any]]:
    experiment = get_experiment(exp_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}")

    selected = experiment.run(run)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"no runs found for experiment {exp_id!r}")
    observed_path = (
        experiment.observed_path
        if experiment.observed_path is not None and experiment.observed_path.exists()
        else None
    )

    payload = get_or_build(
        exp_id,
        selected.run_id,
        selected.path,
        observed_path,
        build=lambda: build_comparison_payload(
            str(selected.path),
            str(observed_path) if observed_path is not None else None,
            experiment.label,
            synthetic_activities_path=str(selected.activities_path),
        ),
        refresh=refresh,
        extra_paths=(selected.social_network_path, selected.activities_path),
    )
    payload = {**payload, "run_id": selected.run_id}
    return ApiResponseWrapper(data=payload)


@router.get("/experiments/{exp_id}/home-work")
def get_home_work(
    exp_id: str,
    run: Optional[str] = Query(None, description="Run id (timestamp). Defaults to the latest run."),
    gender: Optional[str] = Query(None),
    age_bracket: Optional[str] = Query(None),
    job: Optional[str] = Query(None),
    refresh: bool = Query(False, description="Bypass the cache and recompute."),
) -> ApiResponseWrapper[dict[str, Any]]:
    experiment = get_experiment(exp_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}")

    selected = experiment.run(run)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"no runs found for experiment {exp_id!r}")

    if gender is not None and gender not in GENDERS:
        raise HTTPException(status_code=422, detail=f"unknown gender {gender!r}")
    if job is not None and job not in JOBS:
        raise HTTPException(status_code=422, detail=f"unknown job {job!r}")
    try:
        age_min, age_max = resolve_age_bracket(age_bracket)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    observed_path = (
        experiment.observed_path
        if experiment.observed_path is not None and experiment.observed_path.exists()
        else None
    )
    demo = DemoFilter(gender=gender, age_min=age_min, age_max=age_max, job=job)
    payload = get_or_build_home_work(
        selected.path,
        observed_path,
        experiment.profiles_path,
        demo,
        refresh=refresh,
    )
    payload = {**payload, "run_id": selected.run_id}
    return ApiResponseWrapper(data=payload)

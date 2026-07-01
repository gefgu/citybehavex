"""Comparison-payload endpoint — the JSON that replaces the standalone HTML."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..cache import get_or_build
from ..experiments import get_experiment
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
    if experiment.observed_path is None or not experiment.observed_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"experiment {exp_id!r} has no observed comparison parquet",
        )

    payload = get_or_build(
        exp_id,
        selected.run_id,
        selected.path,
        experiment.observed_path,
        build=lambda: build_comparison_payload(
            str(selected.path), str(experiment.observed_path), experiment.label
        ),
        refresh=refresh,
    )
    payload = {**payload, "run_id": selected.run_id}
    return ApiResponseWrapper(data=payload)

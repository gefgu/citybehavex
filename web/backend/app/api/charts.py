"""Comparison-payload endpoint — the JSON that replaces the standalone HTML."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..cache import get_or_build
from ..executor import get_executor
from ..experiments import get_experiment
from ..home_work_data import DemoFilter, GENDERS, JOBS, get_or_build_home_work, resolve_age_bracket
from ..models import ApiResponseWrapper
from ..payload import build_comparison_payload, build_network_validation_payload

router = APIRouter(tags=["charts"])


def _picklable_nv_config(nv_cfg: Any) -> Any:
    """``network_validation_config`` is a pydantic model, already picklable
    on its own -- convert to a plain ``SimpleNamespace`` defensively anyway,
    so a `ProcessPoolExecutor` dispatch never depends on that being true and
    ``getattr(nv_cfg, "x", default)`` call sites in ``build_network_validation_payload``
    keep working unchanged either way.
    """
    if nv_cfg is None:
        return None
    if hasattr(nv_cfg, "model_dump"):
        return SimpleNamespace(**nv_cfg.model_dump())
    return nv_cfg


@router.get("/experiments/{exp_id}/charts")
async def get_charts(
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
    road_nodes_path = (
        experiment.road_nodes_path
        if experiment.road_nodes_path is not None and experiment.road_nodes_path.exists()
        else None
    )
    road_edges_path = (
        experiment.road_edges_path
        if experiment.road_edges_path is not None and experiment.road_edges_path.exists()
        else None
    )
    time_use_path = (
        experiment.time_use_path
        if experiment.time_use_path is not None and experiment.time_use_path.exists()
        else None
    )

    payload = await get_or_build(
        exp_id,
        selected.run_id,
        selected.path,
        observed_path,
        build_fn=build_comparison_payload,
        build_kwargs=dict(
            synthetic_path=str(selected.path),
            observed_path=str(observed_path) if observed_path is not None else None,
            observed_label=experiment.label,
            synthetic_activities_path=str(selected.activities_path),
            time_use_path=str(time_use_path) if time_use_path is not None else None,
            time_use_label=experiment.time_use_label,
            time_use_country=experiment.time_use_country,
            time_use_survey=experiment.time_use_survey,
            time_use_weight_col=experiment.time_use_weight_col,
            road_nodes_path=str(road_nodes_path) if road_nodes_path is not None else None,
            road_edges_path=str(road_edges_path) if road_edges_path is not None else None,
            special_days=experiment.special_days,
        ),
        executor=get_executor(),
        refresh=refresh,
        extra_paths=tuple(
            p
            for p in (
                selected.social_network_path,
                getattr(selected, "encounters_path", None),
                selected.activities_path,
                time_use_path,
                road_nodes_path,
                road_edges_path,
            )
            if p is not None
        ),
    )
    payload = {**payload, "run_id": selected.run_id}
    return ApiResponseWrapper(data=payload)


@router.get("/experiments/{exp_id}/network-validation")
async def get_network_validation(
    exp_id: str,
    run: Optional[str] = Query(None, description="Run id (timestamp). Defaults to the latest run."),
    refresh: bool = Query(False, description="Bypass the cache and recompute."),
) -> ApiResponseWrapper[dict[str, Any]]:
    """Split out of ``/charts`` so its build time (still the largest single
    section for shanghai/yjmob even after moving the graph computation to
    Rust -- see ``build_network_validation_payload``) doesn't block first
    paint of the rest of the charts. Cached separately under its own key
    (``{exp_id}__network_validation``) so it never collides with the main
    comparison payload's cache entry for the same run.
    """
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

    payload = await get_or_build(
        f"{exp_id}__network_validation",
        selected.run_id,
        selected.path,
        observed_path,
        build_fn=build_network_validation_payload,
        build_kwargs=dict(
            synthetic_path=str(selected.path),
            observed_path=str(observed_path) if observed_path is not None else None,
            network_validation_config=_picklable_nv_config(
                getattr(experiment, "network_validation_config", None)
            ),
        ),
        executor=get_executor(),
        refresh=refresh,
        extra_paths=tuple(
            p
            for p in (
                selected.social_network_path,
                getattr(selected, "encounters_path", None),
            )
            if p is not None
        ),
    )
    payload = {**payload, "run_id": selected.run_id}
    return ApiResponseWrapper(data=payload)


@router.get("/experiments/{exp_id}/home-work")
async def get_home_work(
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
    payload = await get_or_build_home_work(
        selected.path,
        observed_path,
        experiment.profiles_path,
        demo,
        exp_id=exp_id,
        run_id=selected.run_id,
        refresh=refresh,
    )
    payload = {**payload, "run_id": selected.run_id}
    return ApiResponseWrapper(data=payload)

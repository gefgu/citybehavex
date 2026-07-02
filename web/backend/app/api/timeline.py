"""Timeline-view endpoints: run metadata, viewport/time-filtered agent legs,
and a single clicked agent's profile/trips/encounters."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import duckdb
from citybehavex.profiles import AgentProfile, profile_to_narrative
from fastapi import APIRouter, HTTPException, Query

from ..datasource import quote_path, run_summary
from ..experiments import Experiment, Run, get_experiment
from ..models import ApiResponseWrapper
from ..timeline_data import (
    legs_index_path,
    query_active_legs,
    query_agent_encounters,
    query_agent_trips,
    run_bbox,
)

router = APIRouter(tags=["timeline"])

_MAX_WINDOW = timedelta(hours=6)


def _resolve_run(exp_id: str, run_id: Optional[str]) -> tuple[Experiment, Run]:
    experiment = get_experiment(exp_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}")
    selected = experiment.run(run_id)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"no runs found for experiment {exp_id!r}")
    return experiment, selected


@router.get("/experiments/{exp_id}/timeline/meta")
def get_timeline_meta(
    exp_id: str, run: Optional[str] = Query(None, description="Run id. Defaults to the latest run.")
) -> ApiResponseWrapper[dict[str, Any]]:
    experiment, selected = _resolve_run(exp_id, run)
    summary = run_summary(selected.path)
    bbox = run_bbox(exp_id, selected)
    payload = {
        "run_id": selected.run_id,
        "date_start": summary.get("date_start"),
        "date_end": summary.get("date_end"),
        "bbox": bbox,
        "agents_total": summary.get("uids"),
        "has_profiles": bool(experiment.profiles_path and experiment.profiles_path.exists()),
        "has_encounters": selected.encounters_path.exists(),
        "car_speed_kmh": experiment.params.get("car_speed_kmh"),
    }
    return ApiResponseWrapper(data=payload)


@router.get("/experiments/{exp_id}/timeline/legs")
def get_timeline_legs(
    exp_id: str,
    since: datetime,
    until: datetime,
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    run: Optional[str] = Query(None),
    max_agents: int = Query(2000, ge=1, le=5000),
) -> ApiResponseWrapper[dict[str, Any]]:
    _experiment, selected = _resolve_run(exp_id, run)
    if until <= since:
        raise HTTPException(status_code=422, detail="until must be after since")
    if (until - since) > _MAX_WINDOW:
        raise HTTPException(status_code=422, detail="requested window too large (max 6h of sim time per request)")

    legs_path = legs_index_path(exp_id, selected)
    segments, truncated = query_active_legs(
        legs_path, since, until, (min_lat, min_lng, max_lat, max_lng), max_agents
    )
    payload = {
        "run_id": selected.run_id,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "agent_count": len({s["uid"] for s in segments}),
        "truncated": truncated,
        "segments": segments,
    }
    return ApiResponseWrapper(data=payload)


@router.get("/experiments/{exp_id}/timeline/agents/{uid}")
def get_timeline_agent(
    exp_id: str, uid: int, run: Optional[str] = Query(None)
) -> ApiResponseWrapper[dict[str, Any]]:
    experiment, selected = _resolve_run(exp_id, run)
    warnings: list[str] = []
    profile_dict: Optional[dict[str, Any]] = None
    narrative: Optional[str] = None

    if experiment.profiles_path and experiment.profiles_path.exists():
        con = duckdb.connect()
        try:
            row = con.execute(
                f"SELECT * FROM read_parquet('{quote_path(experiment.profiles_path)}') WHERE uid = $uid",
                {"uid": uid},
            ).fetchone()
            cols = [d[0] for d in con.description] if row else []
        finally:
            con.close()
        if row:
            profile_dict = dict(zip(cols, row))
            narrative = profile_to_narrative(AgentProfile.model_validate(profile_dict))
        else:
            warnings.append("uid not found in agent profiles")
    else:
        warnings.append("no agent profiles available for this experiment")

    trips = query_agent_trips(selected.path, uid)
    encounters: list[dict[str, Any]] = []
    if selected.encounters_path.exists():
        encounters = query_agent_encounters(selected.encounters_path, uid)
    else:
        warnings.append("no encounters data available for this experiment")

    payload = {
        "uid": uid,
        "run_id": selected.run_id,
        "profile": profile_dict,
        "narrative": narrative,
        "trips": trips,
        "encounters": encounters,
        "warnings": warnings,
    }
    return ApiResponseWrapper(data=payload)

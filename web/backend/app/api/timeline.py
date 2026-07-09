"""Timeline-view endpoints: run metadata, viewport/time-filtered agent legs,
and a single clicked agent's profile/trips/encounters."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import duckdb
from citybehavex.activities import build_catalog
from citybehavex.profiles import AgentProfile, profile_to_narrative
from fastapi import APIRouter, HTTPException, Query

from ..datasource import quote_path, run_summary
from ..experiments import Experiment, Run, get_experiment
from ..models import ApiResponseWrapper
from ..timeline_data import (
    group_trips_by_location,
    legs_index_path,
    moving_index_path,
    query_active_legs,
    query_agent_crp,
    query_agent_encounters,
    query_agent_encounter_counts,
    query_agent_social_friends,
    query_agent_trips,
    query_activity_at_stop,
    query_stop_activities,
    run_bbox,
)

router = APIRouter(tags=["timeline"])

_MAX_WINDOW = timedelta(hours=6)
_ACTIVITY_BY_ID = {activity.idx: activity for activity in build_catalog()}


def _resolve_run(exp_id: str, run_id: Optional[str]) -> tuple[Experiment, Run]:
    experiment = get_experiment(exp_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"unknown experiment {exp_id!r}")
    selected = experiment.run(run_id)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"no runs found for experiment {exp_id!r}")
    return experiment, selected


def _activity_fields(activity_id: Any) -> dict[str, Any]:
    if activity_id is None:
        return {"activity_name": None, "activity_description": None}
    try:
        activity = _ACTIVITY_BY_ID.get(int(activity_id))
    except (TypeError, ValueError):
        activity = None
    if activity is None:
        return {"activity_name": None, "activity_description": None}
    return {
        "activity_name": activity.name,
        "activity_description": activity.description,
    }


def _query_profiles_by_uid(
    profiles_path: Optional[Any],
    uids: list[int],
) -> dict[int, tuple[dict[str, Any], str]]:
    if not profiles_path or not profiles_path.exists() or not uids:
        return {}

    unique_uids = sorted({int(uid) for uid in uids})
    uid_values = ", ".join(f"({uid})" for uid in unique_uids)
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
                SELECT p.*
                FROM read_parquet('{quote_path(profiles_path)}') p
                JOIN (VALUES {uid_values}) AS requested(uid) USING (uid)
            """
        ).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()

    profiles: dict[int, tuple[dict[str, Any], str]] = {}
    for row in rows:
        profile_dict = dict(zip(cols, row))
        narrative = profile_to_narrative(AgentProfile.model_validate(profile_dict))
        profiles[int(profile_dict["uid"])] = (profile_dict, narrative)
    return profiles


def _profile_artifact_warning(
    profiles_path: Optional[Any],
    uid: int,
    agents_total: Optional[int],
) -> str:
    if not profiles_path or not profiles_path.exists():
        return "no agent profiles available for this experiment"

    con = duckdb.connect()
    try:
        row = con.execute(
            f"""
                SELECT count(*) AS rows, min(uid) AS min_uid, max(uid) AS max_uid
                FROM read_parquet('{quote_path(profiles_path)}')
            """
        ).fetchone()
    finally:
        con.close()

    rows = int(row[0]) if row and row[0] is not None else 0
    min_uid = row[1] if row else None
    max_uid = row[2] if row else None
    if agents_total is not None and rows < int(agents_total):
        return f"profile artifact has {rows} rows for {int(agents_total)} agents; no profile row for uid {uid}"
    if min_uid is not None and max_uid is not None:
        return f"uid {uid} not found in agent profiles (profile uid range {min_uid}..{max_uid})"
    return f"uid {uid} not found in agent profiles"


def _crp_agent_id(display_uid: int) -> int:
    return int(display_uid) - 1


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
    experiment, selected = _resolve_run(exp_id, run)
    if until <= since:
        raise HTTPException(status_code=422, detail="until must be after since")
    if (until - since) > _MAX_WINDOW:
        raise HTTPException(status_code=422, detail="requested window too large (max 6h of sim time per request)")

    legs_path = legs_index_path(exp_id, selected)
    moving_path = moving_index_path(exp_id, selected)
    segments, truncated = query_active_legs(
        legs_path,
        since,
        until,
        (min_lat, min_lng, max_lat, max_lng),
        max_agents,
        moving_path,
        experiment.profiles_path,
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
    summary = run_summary(selected.path)
    agents_total = summary.get("uids")

    profiles = _query_profiles_by_uid(experiment.profiles_path, [uid])
    if profiles:
        profile_dict, narrative = profiles[uid]
    else:
        warnings.append(_profile_artifact_warning(experiment.profiles_path, uid, agents_total))

    trips = query_agent_trips(selected.path, uid)
    has_activities_table = selected.activities_path.exists()
    if has_activities_table:
        # Post-fix run: each row is already one real stop, no lat/lng-merge
        # guesswork needed — attach that stop's real micro-activities.
        activities_by_stop = query_stop_activities(selected.activities_path, uid)
        for trip in trips:
            trip.pop("activity", None)
            stop_id = trip.pop("stop_id", None)
            stop_activities = activities_by_stop.get(int(stop_id), []) if stop_id is not None else []
            trip["activities"] = [
                {**a, **_activity_fields(a.get("activity"))} for a in stop_activities
            ]
    else:
        # Legacy run: fall back to the lat/lng-adjacency merge workaround.
        for trip in trips:
            trip.pop("stop_id", None)
            trip.update(_activity_fields(trip.get("activity")))
        trips = group_trips_by_location(trips)

    encounters: list[dict[str, Any]] = []
    if selected.encounters_path.exists():
        encounters = query_agent_encounters(selected.encounters_path, selected.path, uid)
        contact_profiles = _query_profiles_by_uid(
            experiment.profiles_path,
            [int(e["contact_uid"]) for e in encounters],
        )
        for encounter in encounters:
            stop_id = encounter.pop("stop_id", None)
            activity_row = None
            if has_activities_table and stop_id is not None and encounter.get("stop_arrival") is not None:
                activity_row = query_activity_at_stop(
                    selected.activities_path, int(encounter["contact_uid"]), int(stop_id), encounter["ts"]
                )
                encounter.update(_activity_fields(activity_row.get("activity") if activity_row else None))
            else:
                encounter.update(_activity_fields(encounter.get("activity")))
            contact = contact_profiles.get(int(encounter["contact_uid"]))
            if contact:
                contact_profile, contact_narrative = contact
                encounter["contact_profile"] = contact_profile
                encounter["contact_narrative"] = contact_narrative
            else:
                encounter["contact_profile"] = None
                encounter["contact_narrative"] = None
            encounter["location_warning"] = (
                None if encounter.get("stop_arrival") is not None else "no active stop found for contact at encounter time"
            )
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


@router.get("/experiments/{exp_id}/timeline/agents/{uid}/crp")
def get_timeline_agent_crp(
    exp_id: str, uid: int, run: Optional[str] = Query(None)
) -> ApiResponseWrapper[dict[str, Any]]:
    """ddCRP diary-selection state for one agent: T_a, alpha_a, and per-diary
    usage counts + profile similarity, split by day-type bank."""
    _experiment, selected = _resolve_run(exp_id, run)
    warnings: list[str] = []
    diaries: list[dict[str, Any]] = []
    T_a: Optional[float] = None
    alpha_a: Optional[float] = None

    if selected.crp_path.exists():
        rows = query_agent_crp(selected.crp_path, _crp_agent_id(uid))
        if rows:
            T_a = float(rows[0]["T_a"])
            alpha_a = float(rows[0]["alpha_a"])
            diaries = [
                {
                    "diary_id": r["diary_id"],
                    "day_type": r["day_type"],
                    "sim": float(r["sim"]),
                    "usage_count": int(r["usage_count"]),
                }
                for r in rows
            ]
        else:
            warnings.append("uid not found in ddCRP diary selection data")
    else:
        warnings.append("no ddCRP diary selection data available for this run")

    payload = {
        "uid": uid,
        "run_id": selected.run_id,
        "T_a": T_a,
        "alpha_a": alpha_a,
        "diaries": diaries,
        "warnings": warnings,
    }
    return ApiResponseWrapper(data=payload)


@router.get("/experiments/{exp_id}/timeline/agents/{uid}/social")
def get_timeline_agent_social(
    exp_id: str, uid: int, run: Optional[str] = Query(None)
) -> ApiResponseWrapper[dict[str, Any]]:
    """Initial social graph neighborhood for one displayed timeline agent."""
    experiment, selected = _resolve_run(exp_id, run)
    warnings: list[str] = []
    parameters: dict[str, Any] = {
        "degree": 0,
        "total_social_strength": 0.0,
        "social_graph_k": experiment.params.get("social_graph_k"),
        "layout": None,
        "kind": None,
        "directed": None,
        "rho": experiment.params.get("rho"),
        "gamma": experiment.params.get("gamma"),
        "alpha": experiment.params.get("alpha"),
        "dt_update_mob_sim_hours": experiment.params.get("dt_update_mob_sim_hours"),
        "indipendency_window_hours": experiment.params.get("indipendency_window_hours"),
    }
    friends: list[dict[str, Any]] = []

    if selected.social_network_path.exists():
        social_params, friends, social_warnings = query_agent_social_friends(selected.social_network_path, uid)
        parameters.update({key: value for key, value in social_params.items() if value is not None})
        warnings.extend(social_warnings)
    else:
        warnings.append("no social network sidecar available for this run")

    if selected.encounters_path.exists():
        encounter_counts = query_agent_encounter_counts(selected.encounters_path, uid)
    else:
        encounter_counts = {}
        warnings.append("no encounters data available for this experiment")

    friend_profiles = _query_profiles_by_uid(
        experiment.profiles_path,
        [int(friend["uid"]) for friend in friends],
    )
    for friend in friends:
        friend_uid = int(friend["uid"])
        profile = friend_profiles.get(friend_uid)
        friend["encounter_count"] = encounter_counts.get(friend_uid, 0)
        if profile:
            friend_profile, _friend_narrative = profile
            friend["profile"] = friend_profile
            friend["name"] = friend_profile.get("name")
        else:
            friend["profile"] = None
            friend["name"] = None

    payload = {
        "uid": uid,
        "run_id": selected.run_id,
        "parameters": parameters,
        "friends": friends,
        "warnings": warnings,
    }
    return ApiResponseWrapper(data=payload)

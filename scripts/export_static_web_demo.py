#!/usr/bin/env python3
"""Export a static, GitHub Pages-friendly copy of the web demo data.

The normal web app talks to FastAPI endpoints backed by parquet files. This
script materializes the same endpoint-shaped JSON under
``web/frontend/public/demo-data`` so the Vite frontend can run without a
backend when ``VITE_STATIC_DEMO=true``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import yaml

from web.backend.app.api.charts import (
    _chart_build_kwargs,
    _config_cache_key,
    _picklable_config,
    _picklable_nv_config,
)
from web.backend.app.api.timeline import (
    _activity_fields,
    _crp_agent_id,
    _diary_descriptions,
    _profile_artifact_warning,
    _query_profiles_by_uid,
)
from web.backend.app.cache import get_or_build
from web.backend.app.config import REPO_ROOT
from web.backend.app.experiments import Experiment, Run, get_experiment
from web.backend.app.home_work_data import DemoFilter, build_home_work
from web.backend.app.payload import (
    build_chart_base_payload,
    build_chart_section_payload,
    build_metrics_export_payload,
    build_network_validation_payload,
)
from web.backend.app.timeline_data import (
    group_trips_by_location,
    legs_index_path,
    moving_index_path,
    query_activity_at_stop,
    query_active_legs,
    query_agent_crp,
    query_agent_encounter_counts,
    query_agent_encounters,
    query_agent_social_friends,
    query_agent_trips,
    query_stop_activities,
    run_bbox,
)
from web.backend.app.datasource import run_summary

DEFAULT_SECTIONS: tuple[tuple[str, str], ...] = (
    ("micro-activity", "all"),
    ("time-use", "all"),
    ("activity", "all"),
    ("motifs", "all"),
    ("profiles", "all"),
    ("social-network", "all"),
    ("metrics", "all"),
    ("transport-spatial", "all"),
    ("distributions", "all"),
    ("mobility-laws", "all"),
    ("stvd", "all"),
)


def _wrapped(data: Any) -> dict[str, Any]:
    return {"data": data}


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False, default=_json_default),
        encoding="utf-8",
    )


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sanitize_experiment(experiment: Experiment, allow_observed: bool, label: str | None) -> Experiment:
    return replace(
        experiment,
        label=label or experiment.label,
        observed_path=experiment.observed_path if allow_observed else None,
    )


def _filter_runs(experiment: Experiment, run_id: str) -> Experiment:
    selected = experiment.run(run_id)
    if selected is None:
        raise RuntimeError(f"{experiment.id}: run {run_id!r} not found")
    return replace(experiment, runs=[selected])


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _parquet_columns(path: Path) -> list[str]:
    con = duckdb.connect()
    try:
        rows = con.execute(f"SELECT name FROM parquet_schema('{_quote_path(path)}')").fetchall()
    finally:
        con.close()
    return [row[0] for row in rows if row[0] not in {"schema", "duckdb_schema"}]


def _copy_user_sample(src: Path, dst: Path, uid_col: str, max_uid: int, *, zero_based: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lower = 0 if zero_based else 1
    upper = max_uid - 1 if zero_based else max_uid
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{_quote_path(src)}')
                WHERE "{uid_col}" BETWEEN {lower} AND {upper}
            )
            TO '{_quote_path(dst)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def _copy_encounter_sample(src: Path, dst: Path, max_uid: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{_quote_path(src)}')
                WHERE agent BETWEEN 1 AND {max_uid}
                  AND contact BETWEEN 1 AND {max_uid}
            )
            TO '{_quote_path(dst)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def _copy_social_sample(src: Path, dst: Path, max_uid: int) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    nodes = [node for node in payload.get("nodes", []) if len(node) >= 4 and int(node[3]) <= max_uid]
    kept_zero_based = {int(node[3]) - 1 for node in nodes}
    edges = [
        edge
        for edge in payload.get("edges", [])
        if len(edge) >= 2 and int(edge[0]) in kept_zero_based and int(edge[1]) in kept_zero_based
    ]
    degrees = [0] * max_uid
    for edge in edges:
        source = int(edge[0])
        target = int(edge[1])
        if 0 <= source < max_uid:
            degrees[source] += 1
        if 0 <= target < max_uid:
            degrees[target] += 1

    sampled = dict(payload)
    sampled["node_count"] = len(nodes)
    sampled["edge_count"] = len(edges)
    sampled["nodes"] = nodes
    sampled["edges"] = edges
    sampled["degrees"] = degrees
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(sampled, separators=(",", ":")), encoding="utf-8")


def _sample_run(experiment: Experiment, max_uid: int) -> Experiment:
    selected = experiment.runs[0]
    sample_dir = REPO_ROOT / "data" / "static_demo_samples" / experiment.id / selected.run_id / f"first_{max_uid}"
    sample_path = sample_dir / f"{selected.path.stem}_first{max_uid}{selected.path.suffix}"
    print(f"[{experiment.id}] materializing first {max_uid} users -> {sample_path}", flush=True)

    _copy_user_sample(selected.path, sample_path, "uid", max_uid)
    for source, target, uid_col, zero_based in (
        (selected.activities_path, sample_path.with_name(f"{sample_path.stem}_activities{sample_path.suffix}"), "uid", False),
        (selected.moving_path, sample_path.with_name(f"{sample_path.stem}_moving{sample_path.suffix}"), "uid", False),
        (selected.crp_path, sample_path.with_name(f"{sample_path.stem}_crp{sample_path.suffix}"), "agent", True),
    ):
        if source.exists():
            _copy_user_sample(source, target, uid_col, max_uid, zero_based=zero_based)
    if selected.encounters_path.exists():
        _copy_encounter_sample(
            selected.encounters_path,
            sample_path.with_name(f"{sample_path.stem}_encounters{sample_path.suffix}"),
            max_uid,
        )
    if selected.social_network_path.exists():
        _copy_social_sample(
            selected.social_network_path,
            sample_path.with_name(f"{sample_path.stem}_social_network.json"),
            max_uid,
        )

    profiles_path = experiment.profiles_path
    sampled_profiles_path = profiles_path
    if profiles_path is not None and profiles_path.exists() and "uid" in _parquet_columns(profiles_path):
        sampled_profiles_path = sample_dir / f"{profiles_path.stem}_first{max_uid}{profiles_path.suffix}"
        _copy_user_sample(profiles_path, sampled_profiles_path, "uid", max_uid)

    sampled_run = Run(
        run_id=f"{selected.run_id}_first{max_uid}",
        path=sample_path,
        mtime=sample_path.stat().st_mtime,
    )
    return replace(
        experiment,
        runs=[sampled_run],
        profiles_path=sampled_profiles_path,
        profiles_output=sampled_profiles_path,
    )


def _sample_observed(experiment: Experiment, max_users: int) -> Experiment:
    observed_path = experiment.observed_path
    if observed_path is None or not observed_path.exists():
        raise RuntimeError(f"{experiment.id}: cannot sample missing observed path")

    columns = _parquet_columns(observed_path)
    uid_col = "uid" if "uid" in columns else "user_id" if "user_id" in columns else None
    if uid_col is None:
        raise RuntimeError(f"{experiment.id}: observed path has no uid/user_id column")

    sample_dir = REPO_ROOT / "data" / "static_demo_samples" / experiment.id / "observed" / f"first_{max_users}"
    sample_path = sample_dir / f"{observed_path.stem}_first{max_users}{observed_path.suffix}"
    print(f"[{experiment.id}] materializing first {max_users} observed users -> {sample_path}", flush=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                WITH first_users AS (
                    SELECT DISTINCT "{uid_col}" AS sampled_uid
                    FROM read_parquet('{_quote_path(observed_path)}')
                    ORDER BY sampled_uid
                    LIMIT {max_users}
                )
                SELECT obs.*
                FROM read_parquet('{_quote_path(observed_path)}') obs
                JOIN first_users ON obs."{uid_col}" = first_users.sampled_uid
            )
            TO '{_quote_path(sample_path)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()

    return replace(experiment, observed_path=sample_path)


def _validate_expected_agents(experiment: Experiment, expected_agents: int | None) -> None:
    if expected_agents is None:
        return
    selected = experiment.runs[0]
    summary = run_summary(selected.path)
    actual = summary.get("uids")
    if actual != expected_agents:
        raise RuntimeError(
            f"{experiment.id} run {selected.run_id} has {actual} agents; "
            f"manifest expected {expected_agents}. Generate/pin the intended demo run first."
        )


async def _build_chart_payloads(
    out_dir: Path,
    experiment: Experiment,
    sections: list[tuple[str, str]],
) -> None:
    selected = experiment.runs[0]
    print(f"[{experiment.id}] exporting chart payloads", flush=True)
    observed_path = (
        experiment.observed_path
        if experiment.observed_path is not None and experiment.observed_path.exists()
        else None
    )
    time_use_path = (
        experiment.time_use_path
        if experiment.time_use_path is not None and experiment.time_use_path.exists()
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
    kwargs = _chart_build_kwargs(experiment, selected, observed_path, time_use_path)

    base = await get_or_build(
        experiment.id,
        selected.run_id,
        selected.path,
        observed_path,
        build_fn=build_chart_base_payload,
        build_kwargs=kwargs,
        extra_paths=tuple(
            p
            for p in (
                selected.social_network_path,
                selected.encounters_path,
                selected.activities_path,
                selected.moving_path,
                time_use_path,
                road_nodes_path,
                road_edges_path,
            )
            if p is not None
        ),
        extra_key={
            "transport_spatial": _config_cache_key(experiment, "transport_spatial_config"),
            "evaluation_adaptation": _config_cache_key(experiment, "evaluation_adaptation_config"),
            "static_export_observed": str(observed_path) if observed_path else None,
        },
    )
    _write_json(out_dir / "charts" / "base.json", _wrapped({**base, "run_id": selected.run_id}))

    for section, filter_key in sections:
        try:
            payload = await get_or_build(
                experiment.id,
                selected.run_id,
                selected.path,
                observed_path,
                build_fn=build_chart_section_payload,
                build_kwargs={**kwargs, "section": section, "filter_key": filter_key},
                extra_paths=tuple(
                    p for p in (selected.activities_path, selected.moving_path, time_use_path) if p is not None
                ),
                extra_key={
                    "section": section,
                    "filter": filter_key,
                    "transport_spatial": _config_cache_key(experiment, "transport_spatial_config"),
                    "evaluation_adaptation": _config_cache_key(
                        experiment, "evaluation_adaptation_config"
                    ),
                    "static_export_observed": str(observed_path) if observed_path else None,
                },
            )
        except ValueError:
            continue
        _write_json(
            out_dir / "charts" / "sections" / section / f"{filter_key}.json",
            _wrapped({**payload, "run_id": selected.run_id}),
        )

    metrics = await get_or_build(
        f"{experiment.id}__metrics_export",
        selected.run_id,
        selected.path,
        observed_path,
        build_fn=build_metrics_export_payload,
        build_kwargs=kwargs,
        extra_paths=tuple(p for p in (selected.activities_path, time_use_path) if p is not None),
        extra_key={
            "format": "json",
            "evaluation_adaptation": _config_cache_key(experiment, "evaluation_adaptation_config"),
            "static_export_observed": str(observed_path) if observed_path else None,
        },
    )
    _write_json(
        out_dir / "metrics-export.json",
        {"experiment_id": experiment.id, "run_id": selected.run_id, **metrics},
    )

    network_validation = await get_or_build(
        f"{experiment.id}__network_validation",
        selected.run_id,
        selected.path,
        observed_path,
        build_fn=build_network_validation_payload,
        build_kwargs={
            "synthetic_path": str(selected.path),
            "observed_path": str(observed_path) if observed_path is not None else None,
            "network_validation_config": _picklable_nv_config(experiment.network_validation_config),
        },
        extra_paths=tuple(p for p in (selected.social_network_path, selected.encounters_path) if p is not None),
        extra_key={"static_export_observed": str(observed_path) if observed_path else None},
    )
    _write_json(
        out_dir / "network-validation.json",
        _wrapped({**network_validation, "run_id": selected.run_id}),
    )

    home_work = await get_or_build(
        f"{experiment.id}__home_work",
        selected.run_id,
        selected.path,
        observed_path,
        build_fn=build_home_work,
        build_kwargs={
            "synthetic_path": selected.path,
            "observed_path": observed_path,
            "profiles_path": experiment.profiles_path,
            "demo": DemoFilter(),
        },
        extra_paths=(experiment.profiles_path,) if experiment.profiles_path else (),
        extra_key={
            "demo": {
                "gender": None,
                "age_min": None,
                "age_max": None,
                "job": None,
            },
            "static_export_observed": str(observed_path) if observed_path else None,
        },
    )
    _write_json(out_dir / "home-work" / "all.json", _wrapped({**home_work, "run_id": selected.run_id}))
    print(f"[{experiment.id}] chart payloads complete", flush=True)


def _timeline_meta(experiment: Experiment) -> dict[str, Any]:
    selected = experiment.runs[0]
    summary = run_summary(selected.path)
    return {
        "run_id": selected.run_id,
        "date_start": summary.get("date_start"),
        "date_end": summary.get("date_end"),
        "bbox": run_bbox(experiment.id, selected),
        "agents_total": summary.get("uids"),
        "has_profiles": bool(experiment.profiles_path and experiment.profiles_path.exists()),
        "has_encounters": selected.encounters_path.exists(),
        "car_speed_kmh": experiment.params.get("car_speed_kmh"),
    }


def _export_timeline_chunks(
    out_dir: Path,
    experiment: Experiment,
    *,
    chunk_hours: int,
    max_agents: int,
    max_days: int | None = None,
) -> list[dict[str, Any]]:
    selected = experiment.runs[0]
    print(f"[{experiment.id}] exporting timeline chunks", flush=True)
    meta = _timeline_meta(experiment)
    _write_json(out_dir / "timeline" / "meta.json", _wrapped(meta))
    bbox = meta["bbox"]
    if not meta["date_start"] or not meta["date_end"] or bbox is None:
        _write_json(out_dir / "timeline" / "chunks.json", _wrapped({"chunks": []}))
        return []

    start = _parse_dt(meta["date_start"])
    end = _parse_dt(meta["date_end"])
    if max_days is not None:
        end = min(end, start + timedelta(days=max_days))
    step = timedelta(hours=chunk_hours)
    legs_path = legs_index_path(experiment.id, selected)
    moving_path = moving_index_path(experiment.id, selected)
    chunks: list[dict[str, Any]] = []
    current = start
    index = 0
    while current < end:
        until = min(current + step, end)
        segments, truncated = query_active_legs(
            legs_path,
            current,
            until,
            (bbox["min_lat"], bbox["min_lng"], bbox["max_lat"], bbox["max_lng"]),
            max_agents,
            moving_path,
            experiment.profiles_path,
        )
        chunk_name = f"{index:05d}.json"
        payload = {
            "run_id": selected.run_id,
            "since": current.isoformat(),
            "until": until.isoformat(),
            "agent_count": len({s["uid"] for s in segments}),
            "truncated": truncated,
            "segments": segments,
        }
        _write_json(out_dir / "timeline" / "legs" / chunk_name, _wrapped(payload))
        chunks.append({"file": chunk_name, "since": payload["since"], "until": payload["until"]})
        current = until
        index += 1

    _write_json(out_dir / "timeline" / "chunks.json", _wrapped({"chunks": chunks}))
    print(f"[{experiment.id}] exported {len(chunks)} timeline chunks", flush=True)
    return chunks


def _export_agent_details(out_dir: Path, experiment: Experiment, max_agents: int) -> None:
    selected = experiment.runs[0]
    print(f"[{experiment.id}] exporting agent details for {max_agents} agents", flush=True)
    for uid in range(1, max_agents + 1):
        agent = _timeline_agent_payload(experiment, selected, uid)
        crp = _timeline_agent_crp_payload(experiment, selected, uid)
        social = _timeline_agent_social_payload(experiment, selected, uid)
        agent_dir = out_dir / "timeline" / "agents" / str(uid)
        _write_json(agent_dir / "profile.json", _wrapped(agent))
        _write_json(agent_dir / "crp.json", _wrapped(crp))
        _write_json(agent_dir / "social.json", _wrapped(social))
    print(f"[{experiment.id}] agent details complete", flush=True)


def _timeline_agent_payload(experiment: Experiment, selected: Run, uid: int) -> dict[str, Any]:
    warnings: list[str] = []
    profile_dict = None
    narrative = None
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
        activities_by_stop = query_stop_activities(selected.activities_path, uid)
        for trip in trips:
            trip.pop("activity", None)
            stop_id = trip.pop("stop_id", None)
            stop_activities = activities_by_stop.get(int(stop_id), []) if stop_id is not None else []
            trip["activities"] = [
                {**activity, **_activity_fields(activity.get("activity"))}
                for activity in stop_activities
            ]
    else:
        for trip in trips:
            trip.pop("stop_id", None)
            trip.update(_activity_fields(trip.get("activity")))
        trips = group_trips_by_location(trips)

    encounters: list[dict[str, Any]] = []
    if selected.encounters_path.exists():
        encounters = query_agent_encounters(selected.encounters_path, selected.path, uid)
        contact_profiles = _query_profiles_by_uid(
            experiment.profiles_path,
            [int(encounter["contact_uid"]) for encounter in encounters],
        )
        for encounter in encounters:
            stop_id = encounter.pop("stop_id", None)
            activity_row = None
            if has_activities_table and stop_id is not None and encounter.get("stop_arrival") is not None:
                activity_row = query_activity_at_stop(
                    selected.activities_path,
                    int(encounter["contact_uid"]),
                    int(stop_id),
                    encounter["ts"],
                )
                encounter.update(_activity_fields(activity_row.get("activity") if activity_row else None))
                if activity_row and activity_row.get("dwell_minutes") is not None:
                    encounter["dwell_minutes"] = activity_row["dwell_minutes"]
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
                None
                if encounter.get("stop_arrival") is not None
                else "no active stop found for contact at encounter time"
            )
    else:
        warnings.append("no encounters data available for this experiment")

    return {
        "uid": uid,
        "run_id": selected.run_id,
        "profile": profile_dict,
        "narrative": narrative,
        "trips": trips,
        "encounters": encounters,
        "warnings": warnings,
    }


def _timeline_agent_crp_payload(experiment: Experiment, selected: Run, uid: int) -> dict[str, Any]:
    warnings: list[str] = []
    diaries: list[dict[str, Any]] = []
    T_a = None
    alpha_a = None

    if selected.crp_path.exists():
        rows = query_agent_crp(selected.crp_path, _crp_agent_id(uid))
        if rows:
            T_a = float(rows[0]["T_a"])
            alpha_a = float(rows[0]["alpha_a"])
            description_by_diary = _diary_descriptions(
                experiment,
                {str(row["day_type"]) for row in rows if row.get("day_type") is not None},
            )
            diaries = [
                {
                    "diary_id": row["diary_id"],
                    "day_type": row["day_type"],
                    "sim": float(row["sim"]),
                    "usage_count": int(row["usage_count"]),
                    **description_by_diary.get((str(row["day_type"]), str(row["diary_id"])), {}),
                }
                for row in rows
            ]
        else:
            warnings.append("uid not found in ddCRP diary selection data")
    else:
        warnings.append("no ddCRP diary selection data available for this run")

    return {
        "uid": uid,
        "run_id": selected.run_id,
        "T_a": T_a,
        "alpha_a": alpha_a,
        "diaries": diaries,
        "warnings": warnings,
    }


def _timeline_agent_social_payload(experiment: Experiment, selected: Run, uid: int) -> dict[str, Any]:
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

    return {
        "uid": uid,
        "run_id": selected.run_id,
        "parameters": parameters,
        "friends": friends,
        "warnings": warnings,
    }


async def export_static_demo(manifest_path: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    output_dir = (REPO_ROOT / manifest.get("output_dir", "web/frontend/public/demo-data")).resolve()
    chunk_hours = int(manifest.get("timeline_chunk_hours", 6))
    max_agents = int(manifest.get("timeline_max_agents", 500))
    export_agent_details = bool(manifest.get("export_agent_details", True))
    sections = [tuple(item) for item in manifest.get("chart_sections", DEFAULT_SECTIONS)]

    prepared: list[tuple[dict[str, Any], Experiment]] = []
    for entry in manifest["experiments"]:
        exp_id = entry["id"]
        run_id = entry["run_id"]
        base_experiment = get_experiment(exp_id)
        if base_experiment is None:
            raise RuntimeError(f"unknown experiment {exp_id!r}")
        experiment = _sanitize_experiment(
            _filter_runs(base_experiment, run_id),
            allow_observed=bool(entry.get("allow_observed", False)),
            label=entry.get("label"),
        )
        sample_agents = entry.get("sample_agents")
        if sample_agents is not None:
            experiment = _sample_run(experiment, int(sample_agents))
        observed_sample_agents = entry.get("observed_sample_agents")
        if observed_sample_agents is not None:
            experiment = _sample_observed(experiment, int(observed_sample_agents))
        _validate_expected_agents(experiment, entry.get("expected_agents"))
        prepared.append((entry, experiment))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments_payload: list[dict[str, Any]] = []
    detail_payloads: dict[str, dict[str, Any]] = {}

    for entry, experiment in prepared:
        exp_id = entry["id"]
        run_id = entry["run_id"]
        exp_out = output_dir / exp_id / run_id
        detail = experiment.to_dict(with_summary=True)
        detail["static_demo"] = True
        detail["observed_sanitized"] = not bool(entry.get("allow_observed", False))
        experiments_payload.append(detail)
        detail_payloads[exp_id] = detail

        await _build_chart_payloads(exp_out, experiment, sections)
        _export_timeline_chunks(
            exp_out,
            experiment,
            chunk_hours=chunk_hours,
            max_agents=max_agents,
            max_days=entry.get("timeline_days"),
        )
        if export_agent_details:
            _export_agent_details(exp_out, experiment, max_agents=max_agents)

    _write_json(output_dir / "experiments.json", _wrapped(experiments_payload))
    for exp_id, detail in detail_payloads.items():
        _write_json(output_dir / "experiments" / f"{exp_id}.json", _wrapped(detail))
    _write_json(
        output_dir / "manifest.json",
        {
            "generated_from": str(manifest_path.relative_to(REPO_ROOT)),
            "timeline_chunk_hours": chunk_hours,
            "timeline_max_agents": max_agents,
            "experiments": [
                {"id": e["id"], "run_id": e["run_id"], "allow_observed": bool(e.get("allow_observed", False))}
                for e in manifest["experiments"]
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="web/demo_export.yaml",
        help="Path to the static demo export manifest.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(export_static_demo((REPO_ROOT / args.manifest).resolve()))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

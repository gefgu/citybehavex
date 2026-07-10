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

import yaml

from web.backend.app.api.charts import (
    _chart_build_kwargs,
    _config_cache_key,
    _picklable_config,
    _picklable_nv_config,
)
from web.backend.app.api.timeline import (
    get_timeline_agent,
    get_timeline_agent_crp,
    get_timeline_agent_social,
)
from web.backend.app.cache import get_or_build
from web.backend.app.config import REPO_ROOT
from web.backend.app.experiments import Experiment, get_experiment
from web.backend.app.home_work_data import DemoFilter, build_home_work
from web.backend.app.payload import (
    build_chart_base_payload,
    build_chart_section_payload,
    build_metrics_export_payload,
    build_network_validation_payload,
)
from web.backend.app.timeline_data import (
    legs_index_path,
    moving_index_path,
    query_active_legs,
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
        agent = get_timeline_agent(experiment.id, uid, selected.run_id).data
        crp = get_timeline_agent_crp(experiment.id, uid, selected.run_id).data
        social = get_timeline_agent_social(experiment.id, uid, selected.run_id).data
        agent_dir = out_dir / "timeline" / "agents" / str(uid)
        _write_json(agent_dir / "profile.json", _wrapped(agent))
        _write_json(agent_dir / "crp.json", _wrapped(crp))
        _write_json(agent_dir / "social.json", _wrapped(social))
    print(f"[{experiment.id}] agent details complete", flush=True)


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
        _export_timeline_chunks(exp_out, experiment, chunk_hours=chunk_hours, max_agents=max_agents)
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

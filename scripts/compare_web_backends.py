#!/usr/bin/env python3
"""Compare Python and Rust web backend JSON responses.

Run both servers first:

  .venv/bin/python -m uvicorn app.main:app --app-dir web/backend --port 8000
  CBX_WEB_RS_PORT=8001 cargo run -p citybehavex-web

This harness is intentionally HTTP-level: it catches route/envelope/status
drift that unit tests on payload helpers miss. It knows about the one accepted
deviation in the Rust migration -- a Python Polars null-handling bug
documented in RUST_BACKEND_MIGRATION.md that bakes a spurious ~20015 km jump
into every transport leg -- which surfaces in two places: transport-spatial's
`mean_jump_km` (per-mode aggregate) and `jump_ecdf`'s per-leg x-coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


KNOWN_TRANSPORT_EXCEPTION = (
    "data",
    "transport_spatial",
    "summary",
    "*",
    "modes",
    "*",
    "mean_jump_km",
)

# Same root cause as KNOWN_TRANSPORT_EXCEPTION (Python's null-unsafe
# `min_horizontal` clamp bakes a spurious ~20015 km jump into every leg, not
# just the mode-level mean) surfacing in the per-leg ECDF. Unlike the mean,
# this isn't a uniform per-point offset: corrupting a subset of legs to
# ~20015 km reshuffles that mode's sort order/rank, which shifts both the
# x-coordinate *and* the y-coordinate (empirical percentile) of every
# downstream point in the affected series -- so the whole `jump_ecdf`
# subtree is incomparable point-for-point wherever the bug bites, not just
# the directly-corrupted values. Exempt the whole subtree rather than
# individual coordinates.
KNOWN_JUMP_ECDF_EXCEPTION = (
    "data",
    "transport_spatial",
    "jump_ecdf",
)

# Explicitly out-of-scope per the original refactor plan and documented in
# RUST_BACKEND_MIGRATION.md: `truncated_powerlaw_dataset`'s bounded
# coarse-to-fine grid search (Rust) vs scipy's Trust-Region-Reflective
# `curve_fit` (Python) is an accepted, permanent numeric divergence, not a
# bug -- both the fit parameters and the raw scatter binning they drive can
# differ substantially. Scoped to just the two blocks that use it
# (`travel_distance`, `radius_of_gyration`); `daily_locations` and
# `distance_frequency` use different, non-fitted dataset builders and should
# still be compared normally.
KNOWN_TRUNCATED_POWERLAW_BLOCKS = ("travel_distance", "radius_of_gyration")


def is_known_truncated_powerlaw_exception(path: tuple[str, ...]) -> bool:
    return (
        len(path) >= 6
        and path[0] == "data"
        and path[1] == "mobility_laws"
        and path[2] == "groups"
        and path[4] == "blocks"
        and path[5] in KNOWN_TRUNCATED_POWERLAW_BLOCKS
    )

# Also documented in RUST_BACKEND_MIGRATION.md as an accepted, permanent
# divergence: Rust's k-means (seeded from sorted percentile points) isn't
# the same algorithm as Python's `sklearn.cluster.KMeans(random_state=0,
# n_init=10)`, so a borderline user in noisy real observed data can land in
# a different Routiner/Regular/Scouter bucket on the two backends -- every
# underlying per-user metric (regularity/diversity/stationarity/entropy) is
# separately unit-tested as exact, but `profiles.box`'s per-bucket quantiles
# and `profiles.scatter`'s per-point cluster label are both downstream of
# which bucket each user landed in, so they inherit the divergence.
# Synthetic-side clustering has matched exactly in every check so far (only
# observed data is noisy enough to have borderline cases), but there's no
# per-label path segment to exempt just the observed side generically
# across experiments, so the whole subtree is exempted.
KNOWN_PROFILES_CLUSTER_EXCEPTION_PREFIXES = (
    ("data", "profiles", "box"),
    ("data", "profiles", "scatter"),
)


def is_known_profiles_cluster_exception(path: tuple[str, ...]) -> bool:
    return any(path_has_prefix(prefix, path) for prefix in KNOWN_PROFILES_CLUSTER_EXCEPTION_PREFIXES)

# Kept in sync with web/backend/app/home_work_data.py's constants. This
# harness doesn't import that module (it talks HTTP-only, deliberately, so it
# still works once the Python backend is retired) so the lists are
# duplicated here on purpose.
HOME_WORK_GENDERS = ["male", "female"]
HOME_WORK_AGE_BRACKETS = ["16_24", "25_34", "35_44", "45_59", "60_80"]

_MAX_TIMELINE_WINDOW = timedelta(hours=6)


@dataclass
class Response:
    status: int
    body: Any


def fetch(base: str, path: str) -> Response:
    url = base.rstrip("/") + path
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=120) as res:
            text = res.read().decode("utf-8")
            return Response(res.status, json.loads(text) if text else None)
    except HTTPError as exc:
        text = exc.read().decode("utf-8")
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = text
        return Response(exc.code, body)
    except OSError as exc:
        # Connection dropped mid-response (e.g. a panicking request handler
        # that aborted the socket before completing the response) rather than
        # a clean HTTP error status. Surface it as a diffable failure instead
        # of letting the whole harness run crash on one bad endpoint.
        return Response(0, f"connection error: {exc}")


def path_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if len(pattern) != len(path):
        return False
    return all(p == "*" or p == x for p, x in zip(pattern, path))


def path_has_prefix(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def is_known_exception(path: tuple[str, ...]) -> bool:
    return (
        path_matches(KNOWN_TRANSPORT_EXCEPTION, path)
        or path_has_prefix(KNOWN_JUMP_ECDF_EXCEPTION, path)
        or is_known_truncated_powerlaw_exception(path)
        or is_known_profiles_cluster_exception(path)
    )


# GeoJSON `features` lists that aren't emitted in a guaranteed order on
# either side (neither backend sorts; each reflects its own internal
# hashmap/grouping iteration order, which GeoJSON rendering doesn't depend
# on) -- sort by `properties.area` (the H3 cell hex ID) before the normal
# positional list diff so real content bugs are still caught.
# - `stvd`'s per-resolution map layer: verified byte-for-byte identical once
#   aligned by area.
# - `/home-work`'s per-resolution density map: verified identical on every
#   shared cell except one non-deterministic tie-break (see
#   RUST_BACKEND_MIGRATION.md's "/home-work" entry -- Python's own
#   `any_value(lat)`/`any_value(lng)` picking an arbitrary representative
#   point within a tied fine H3 cell isn't itself deterministic in DuckDB,
#   so there's no "more correct" answer to replicate bit-for-bit there).
UNORDERED_FEATURES_PATHS = (
    ("data", "stvd", "groups", "*", "block", "layers", "*", "features"),
    ("data", "home", "*", "layers", "*", "features"),
)


def is_unordered_features_path(path: tuple[str, ...]) -> bool:
    return any(path_matches(pattern, path) for pattern in UNORDERED_FEATURES_PATHS)


def close_numbers(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def diff(a: Any, b: Any, path: tuple[str, ...] = ()) -> list[str]:
    if is_known_exception(path):
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{'.'.join((*path, key))}: missing in Python")
            elif key not in b:
                out.append(f"{'.'.join((*path, key))}: missing in Rust")
            else:
                out.extend(diff(a[key], b[key], (*path, key)))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        if is_unordered_features_path(path):
            a = sorted(a, key=lambda f: f.get("properties", {}).get("area", ""))
            b = sorted(b, key=lambda f: f.get("properties", {}).get("area", ""))
        if len(a) != len(b):
            out.append(f"{'.'.join(path)}: list length {len(a)} != {len(b)}")
        for i, (av, bv) in enumerate(zip(a, b)):
            out.extend(diff(av, bv, (*path, str(i))))
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return [] if close_numbers(float(a), float(b)) else [f"{'.'.join(path)}: {a!r} != {b!r}"]
    return [] if a == b else [f"{'.'.join(path)}: {a!r} != {b!r}"]


def query(path: str, **params: Any) -> str:
    clean = {k: v for k, v in params.items() if v is not None}
    return path if not clean else f"{path}?{urlencode(clean)}"


def _timeline_endpoints(python_base: str, exp_id: str, run_id: str) -> list[str]:
    """Discover a valid legs window/bbox and agent uid from the *Python*
    backend's own responses (both backends then get fetched with the exact
    same query string later), so the discovery step doesn't bias the
    comparison. Returns [] if the run has no locatable data (e.g. empty
    parquet) rather than failing the whole harness."""
    out: list[str] = []
    meta = fetch(python_base, query(f"/api/experiments/{exp_id}/timeline/meta", run=run_id))
    if meta.status != 200 or not meta.body:
        return out
    data = meta.body.get("data") or {}
    bbox = data.get("bbox")
    date_start = data.get("date_start")
    if not bbox or not date_start:
        return out
    try:
        since_dt = datetime.fromisoformat(date_start)
    except ValueError:
        return out
    until_dt = since_dt + _MAX_TIMELINE_WINDOW
    legs_ep = query(
        f"/api/experiments/{exp_id}/timeline/legs",
        run=run_id,
        since=since_dt.isoformat(),
        until=until_dt.isoformat(),
        min_lat=bbox["min_lat"],
        min_lng=bbox["min_lng"],
        max_lat=bbox["max_lat"],
        max_lng=bbox["max_lng"],
    )
    out.append(legs_ep)

    legs = fetch(python_base, legs_ep)
    segments = ((legs.body or {}).get("data") or {}).get("segments") or []
    if not segments:
        return out
    uid = segments[0]["uid"]
    for suffix in ("", "/crp", "/social"):
        out.append(query(f"/api/experiments/{exp_id}/timeline/agents/{uid}{suffix}", run=run_id))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="http://localhost:8000")
    parser.add_argument("--rust", default="http://localhost:8001")
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--filter", action="append", default=["all"])
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--include-slow", action="store_true")
    args = parser.parse_args()

    py_exps = fetch(args.python, "/api/experiments?with_summary=true")
    rs_exps = fetch(args.rust, "/api/experiments?with_summary=true")
    failures = []
    if py_exps.status != rs_exps.status:
        failures.append(f"/api/experiments status {py_exps.status} != {rs_exps.status}")
    failures.extend(diff(py_exps.body, rs_exps.body))
    if failures:
        print("\n".join(failures[:50]), file=sys.stderr)
        return 1

    sections = args.section or [
        "metrics",
        "distributions",
        "transport-spatial",
        "activity",
        "mobility-laws",
        "micro-activity",
        "time-use",
        "motifs",
        "stvd",
        "profiles",
        "social-network",
    ]
    # A representative (not full-cartesian) set of home/work demographic
    # filters: unfiltered, one endpoint per gender/age-bracket, and one
    # combined filter to exercise the AND-of-filters codepath.
    home_work_filters: list[dict[str, str]] = [{}]
    home_work_filters += [{"gender": g} for g in HOME_WORK_GENDERS]
    home_work_filters += [{"age_bracket": b} for b in HOME_WORK_AGE_BRACKETS]
    home_work_filters.append({"gender": HOME_WORK_GENDERS[0], "age_bracket": HOME_WORK_AGE_BRACKETS[0]})

    endpoints: list[str] = []
    for exp in py_exps.body["data"]:
        exp_id = exp["id"]
        runs = exp.get("runs", [])[: args.max_runs]
        endpoints.append(f"/api/experiments/{exp_id}/charts")
        for run in runs:
            run_id = run["run_id"]
            endpoints.append(query(f"/api/experiments/{exp_id}/charts", run=run_id))
            endpoints.append(query(f"/api/experiments/{exp_id}/metrics-export", run=run_id, format="json"))
            if args.include_slow:
                endpoints.append(query(f"/api/experiments/{exp_id}/network-validation", run=run_id))
            for section in sections:
                for filter_key in args.filter:
                    endpoints.append(query(f"/api/experiments/{exp_id}/charts/{section}", run=run_id, filter=filter_key))

            for hw_filter in home_work_filters:
                endpoints.append(query(f"/api/experiments/{exp_id}/home-work", run=run_id, **hw_filter))

            endpoints.append(query(f"/api/experiments/{exp_id}/timeline/meta", run=run_id))
            if args.include_slow:
                endpoints.extend(_timeline_endpoints(args.python, exp_id, run_id))

    for endpoint in endpoints:
        py = fetch(args.python, endpoint)
        rs = fetch(args.rust, endpoint)
        if py.status != rs.status:
            failures.append(f"{endpoint}: status {py.status} != {rs.status}")
            continue
        failures.extend(f"{endpoint}: {msg}" for msg in diff(py.body, rs.body))

    if failures:
        print(f"{len(failures)} parity difference(s):", file=sys.stderr)
        print("\n".join(failures[:200]), file=sys.stderr)
        return 1
    print(f"OK: {len(endpoints) + 1} endpoint responses matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


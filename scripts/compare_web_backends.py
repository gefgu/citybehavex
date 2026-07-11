#!/usr/bin/env python3
"""Compare Python and Rust web backend JSON responses.

Run both servers first:

  .venv/bin/python -m uvicorn app.main:app --app-dir web/backend --port 8000
  CBX_WEB_RS_PORT=8001 cargo run -p citybehavex-web

This harness is intentionally HTTP-level: it catches route/envelope/status
drift that unit tests on payload helpers miss. It knows about the one accepted
deviation in the Rust migration: transport-spatial `mean_jump_km` fixes a
Python Polars null-handling bug documented in RUST_BACKEND_MIGRATION.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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


def path_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if len(pattern) != len(path):
        return False
    return all(p == "*" or p == x for p, x in zip(pattern, path))


def is_known_exception(path: tuple[str, ...]) -> bool:
    return path_matches(KNOWN_TRANSPORT_EXCEPTION, path)


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


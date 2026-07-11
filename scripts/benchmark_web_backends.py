#!/usr/bin/env python3
"""Tiny HTTP timing harness for Python vs Rust web backends."""

from __future__ import annotations

import argparse
import statistics
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def get(url: str) -> float:
    start = time.perf_counter()
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=300) as res:
        res.read()
        if res.status >= 400:
            raise RuntimeError(f"{url}: HTTP {res.status}")
    return time.perf_counter() - start


def path(exp_id: str, endpoint: str, **params: str) -> str:
    q = urlencode({k: v for k, v in params.items() if v is not None})
    suffix = f"?{q}" if q else ""
    return f"/api/experiments/{exp_id}/{endpoint}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--run")
    parser.add_argument("--python", default="http://localhost:8000")
    parser.add_argument("--rust", default="http://localhost:8001")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-network-validation", action="store_true")
    args = parser.parse_args()

    endpoints = [
        path(args.experiment, "charts", run=args.run),
        path(args.experiment, "charts/metrics", run=args.run, filter="all"),
        path(args.experiment, "charts/distributions", run=args.run, filter="all"),
        path(args.experiment, "charts/transport-spatial", run=args.run, filter="all"),
    ]
    if args.include_network_validation:
        endpoints.append(path(args.experiment, "network-validation", run=args.run))

    for endpoint in endpoints:
        print(endpoint)
        for name, base in (("python", args.python), ("rust", args.rust)):
            samples = [get(base.rstrip("/") + endpoint) for _ in range(args.repeats)]
            print(
                f"  {name:6s} median={statistics.median(samples):.3f}s "
                f"min={min(samples):.3f}s max={max(samples):.3f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


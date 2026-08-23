"""Process pool for CPU-bound comparison-payload builds.

The backend runs as a single uvicorn process (no ``--workers``). Route
handlers are plain ``def``, so Starlette already offloads them to its own
threadpool -- but that only helps with I/O waits, not the CPU-bound
pandas/numpy/Rust work in a payload build, which all shares one GIL. Two
concurrent builds (e.g. two browser tabs on different experiments) contend
for that one GIL and stall each other, and even simple unrelated endpoints
running in other threadpool threads get starved while a build is in
progress. Dispatching builds to a real ``ProcessPoolExecutor`` instead gives
each build its own process (its own GIL), so they run truly in parallel
across CPU cores.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

_executor: ProcessPoolExecutor | None = None


def _default_worker_count() -> int:
    override = os.environ.get("CBX_WEB_BUILD_WORKERS")
    if override:
        return max(1, int(override))
    cpu = os.cpu_count() or 2
    # Capped at 4 by default (not cpu_count workers): each build loads at
    # least one, often two, multi-GB parquets fully into memory, so more
    # workers isn't free -- this is a starting default, tune via the env
    # var above per deployment.
    return max(1, min(4, cpu - 1))


def init_executor() -> None:
    global _executor
    if _executor is not None:
        return
    _executor = ProcessPoolExecutor(
        max_workers=_default_worker_count(),
        # spawn, not fork: forking a process that already has background
        # threads (uvicorn's asyncio loop, Starlette's anyio threadpool)
        # risks inheriting a locked mutex and deadlocking a worker. spawn's
        # slower cold start is paid once per worker (workers are reused
        # across many builds, not respawned per request).
        mp_context=mp.get_context("spawn"),
    )


def shutdown_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


def get_executor() -> ProcessPoolExecutor:
    if _executor is None:
        raise RuntimeError("build executor not initialized -- call init_executor() at app startup")
    return _executor

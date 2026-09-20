"""DuckDB helpers for cheap parquet metadata.

The heavy scientific metrics still go through fastmob/pandas (see ``payload.py``);
DuckDB is used here only for the fast, tabular work the Experiments page needs:
row counts, distinct users and the datetime span of a run's parquet, plus
efficient column-projected loading.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future
from hashlib import sha256
from pathlib import Path
from typing import Any

import duckdb

from .config import CACHE_DIR
from .reports_bridge import detect_column

RUN_SUMMARY_CACHE_CAPACITY = 512
RUN_SUMMARY_CACHE_VERSION = "v1"
_RunSummaryCacheKey = tuple[str, int | None, int | None]
_CachedRunSummary = tuple[dict[str, Any] | None, str | None]
_run_summary_cache: OrderedDict[_RunSummaryCacheKey, _CachedRunSummary] = OrderedDict()
_run_summary_cache_lock = threading.Lock()
_run_summary_inflight: dict[_RunSummaryCacheKey, Future[_CachedRunSummary]] = {}


def quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


_quote = quote_path


def parquet_columns(path: Path) -> list[str]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT name FROM parquet_schema('{_quote(path)}')"
        ).fetchall()
        # parquet_schema lists nested/root entries; keep leaf column names.
        return [r[0] for r in rows if r[0] not in {"schema", "duckdb_schema"}]
    finally:
        con.close()


def run_summary(path: Path) -> dict[str, Any]:
    """Row count, distinct-user count and datetime span for a run parquet.

    Columns are auto-detected (schemas differ across cities) so this works for
    both synthetic and observed tables.
    """
    columns = parquet_columns(path)

    class _Cols:
        def __init__(self, names: list[str]):
            self.columns = names

    uid_col = detect_column(_Cols(columns), _UID_CANDIDATES)
    dt_col = detect_column(_Cols(columns), _DATETIME_CANDIDATES)

    select = ["count(*) AS rows"]
    if uid_col:
        select.append(f'count(DISTINCT "{uid_col}") AS uids')
    if dt_col:
        select.append(f'min("{dt_col}"::VARCHAR) AS dt_min')
        select.append(f'max("{dt_col}"::VARCHAR) AS dt_max')

    con = duckdb.connect()
    try:
        row = con.execute(
            f"SELECT {', '.join(select)} FROM read_parquet('{_quote(path)}')"
        ).fetchone()
    finally:
        con.close()

    result: dict[str, Any] = {"rows": int(row[0])}
    idx = 1
    if uid_col:
        result["uids"] = int(row[idx]) if row[idx] is not None else None
        idx += 1
    if dt_col:
        result["date_start"] = row[idx]
        result["date_end"] = row[idx + 1]
    return result


def _run_summary_cache_key(path: Path) -> _RunSummaryCacheKey:
    try:
        stat = path.stat()
    except OSError:
        return (str(path.resolve()), None, None)
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _run_summary_disk_cache_path(key: _RunSummaryCacheKey) -> Path:
    encoded_key = json.dumps(
        {"version": RUN_SUMMARY_CACHE_VERSION, "path": key[0], "mtime_ns": key[1], "size": key[2]},
        sort_keys=True,
    )
    digest = sha256(encoded_key.encode("utf-8")).hexdigest()
    return CACHE_DIR / "run_summaries" / f"{RUN_SUMMARY_CACHE_VERSION}__{digest}.json"


def _read_run_summary_disk_cache(path: Path) -> _CachedRunSummary | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    error = value.get("error")
    if summary is not None and not isinstance(summary, dict):
        return None
    if error is not None and not isinstance(error, str):
        return None
    if summary is None and error is None:
        return None
    return (dict(summary) if summary is not None else None, error)


def _write_run_summary_disk_cache(path: Path, value: _CachedRunSummary) -> None:
    summary, error = value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temp_path.write_text(
                json.dumps({"summary": summary, "error": error}), encoding="utf-8"
            )
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    except OSError:
        # A read-only or unavailable cache directory must not make the
        # experiments endpoint unavailable; fall back to the in-memory cache.
        return


def _remember_run_summary(key: _RunSummaryCacheKey, value: _CachedRunSummary) -> None:
    _run_summary_cache[key] = value
    _run_summary_cache.move_to_end(key)
    while len(_run_summary_cache) > RUN_SUMMARY_CACHE_CAPACITY:
        _run_summary_cache.popitem(last=False)


def cached_run_summary(path: Path) -> _CachedRunSummary:
    """Return a cached parquet summary, durable across backend restarts.

    Cache entries are keyed by the resolved source path, nanosecond mtime, and
    file size, so changed outputs automatically miss the cache. Concurrent
    cold-cache callers for the same run share one computation.
    """
    key = _run_summary_cache_key(path)
    with _run_summary_cache_lock:
        cached = _run_summary_cache.get(key)
        if cached is not None:
            _run_summary_cache.move_to_end(key)
            summary, error = cached
            return (dict(summary) if summary is not None else None, error)

        future = _run_summary_inflight.get(key)
        owner = future is None
        if owner:
            future = Future()
            _run_summary_inflight[key] = future

    if not owner:
        summary, error = future.result()
        return (dict(summary) if summary is not None else None, error)

    try:
        cache_path = _run_summary_disk_cache_path(key)
        computed = _read_run_summary_disk_cache(cache_path)
        if computed is None:
            try:
                computed = (run_summary(path), None)
            except Exception as exc:  # noqa: BLE001 - callers surface summary errors as metadata
                computed = (None, str(exc))
            _write_run_summary_disk_cache(cache_path, computed)

        with _run_summary_cache_lock:
            _remember_run_summary(key, computed)
        future.set_result(computed)
    except Exception as exc:  # noqa: BLE001 - always release concurrent callers
        future.set_exception(exc)
        raise
    finally:
        with _run_summary_cache_lock:
            _run_summary_inflight.pop(key, None)

    summary, error = computed
    return (dict(summary) if summary is not None else None, error)


# Kept in sync with citybehavex.reports.comparison candidate lists.
_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]
_DATETIME_CANDIDATES = [
    "datetime", "start_timestamp", "timestamp", "check-in_time",
    "start_time", "_start_time", "checkin_time", "time", "date",
]

"""DuckDB helpers for cheap parquet metadata.

The heavy scientific metrics still go through skmob2/pandas (see ``payload.py``);
DuckDB is used here only for the fast, tabular work the Experiments page needs:
row counts, distinct users and the datetime span of a run's parquet, plus
efficient column-projected loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import duckdb

from .reports_bridge import detect_column


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


# Kept in sync with citybehavex.reports.comparison candidate lists.
_UID_CANDIDATES = ["uid", "user_id", "user", "agent_id", "userid"]
_DATETIME_CANDIDATES = [
    "datetime", "start_timestamp", "timestamp", "check-in_time",
    "start_time", "_start_time", "checkin_time", "time", "date",
]

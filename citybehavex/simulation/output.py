from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

def _stamp_path(path: str, ts: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{ts}{p.suffix}"))


class _IncrementalParquetWriter:
    """Appends DataFrame chunks to a parquet file as they arrive, opening the
    writer lazily on the first non-empty chunk (parquet needs a schema up
    front). Used to stream per-day waypoint chunks straight to disk instead
    of accumulating the whole run's waypoints in memory."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def write(self, chunk: pd.DataFrame | pl.DataFrame) -> None:
        if len(chunk) == 0:
            return
        if isinstance(chunk, pl.DataFrame):
            table = chunk.to_arrow()
        else:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)
        self.rows_written += len(chunk)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

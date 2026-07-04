from __future__ import annotations

import pandas as pd

from citybehavex.simulation.runner import _IncrementalParquetWriter


def test_incremental_parquet_writer_appends_chunks_in_order(tmp_path):
    path = str(tmp_path / "moving.parquet")
    writer = _IncrementalParquetWriter(path)

    chunk1 = pd.DataFrame({"uid": [1, 1], "lat": [48.85, 48.86]})
    chunk2 = pd.DataFrame({"uid": [2, 2], "lat": [48.90, 48.91]})
    writer.write(chunk1)
    writer.write(chunk2)
    writer.close()

    assert writer.rows_written == 4
    result = pd.read_parquet(path)
    assert result["uid"].tolist() == [1, 1, 2, 2]
    assert result["lat"].tolist() == [48.85, 48.86, 48.90, 48.91]


def test_incremental_parquet_writer_skips_empty_chunks_and_never_opens_file(tmp_path):
    path = tmp_path / "moving.parquet"
    writer = _IncrementalParquetWriter(str(path))

    writer.write(pd.DataFrame({"uid": [], "lat": []}))
    writer.close()

    assert writer.rows_written == 0
    assert not path.exists()

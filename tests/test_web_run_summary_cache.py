from __future__ import annotations

import threading

import pytest

from web.backend.app import datasource


@pytest.fixture(autouse=True)
def _isolated_run_summary_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(datasource, "CACHE_DIR", tmp_path / ".web_cache")
    with datasource._run_summary_cache_lock:
        datasource._run_summary_cache.clear()
        datasource._run_summary_inflight.clear()
    yield
    with datasource._run_summary_cache_lock:
        datasource._run_summary_cache.clear()
        datasource._run_summary_inflight.clear()


def test_run_summary_disk_cache_survives_memory_cache_reset(monkeypatch, tmp_path):
    source = tmp_path / "run.parquet"
    source.write_bytes(b"first")
    calls = 0

    def build_summary(path):
        nonlocal calls
        calls += 1
        assert path == source
        return {"rows": 12, "uids": 3}

    monkeypatch.setattr(datasource, "run_summary", build_summary)
    assert datasource.cached_run_summary(source) == ({"rows": 12, "uids": 3}, None)
    assert calls == 1

    with datasource._run_summary_cache_lock:
        datasource._run_summary_cache.clear()

    assert datasource.cached_run_summary(source) == ({"rows": 12, "uids": 3}, None)
    assert calls == 1


def test_run_summary_disk_cache_invalidates_when_source_changes(monkeypatch, tmp_path):
    source = tmp_path / "run.parquet"
    source.write_bytes(b"first")
    calls = 0

    def build_summary(path):
        nonlocal calls
        calls += 1
        return {"rows": calls}

    monkeypatch.setattr(datasource, "run_summary", build_summary)
    assert datasource.cached_run_summary(source) == ({"rows": 1}, None)

    source.write_bytes(b"changed-and-larger")
    assert datasource.cached_run_summary(source) == ({"rows": 2}, None)
    assert calls == 2


def test_run_summary_errors_are_cached_on_disk(monkeypatch, tmp_path):
    source = tmp_path / "broken.parquet"
    source.write_bytes(b"not a parquet")
    calls = 0

    def broken_summary(path):
        nonlocal calls
        calls += 1
        raise ValueError("invalid parquet")

    monkeypatch.setattr(datasource, "run_summary", broken_summary)
    assert datasource.cached_run_summary(source) == (None, "invalid parquet")

    with datasource._run_summary_cache_lock:
        datasource._run_summary_cache.clear()

    assert datasource.cached_run_summary(source) == (None, "invalid parquet")
    assert calls == 1


def test_corrupt_disk_cache_is_rebuilt(monkeypatch, tmp_path):
    source = tmp_path / "run.parquet"
    source.write_bytes(b"contents")
    cache_path = datasource._run_summary_disk_cache_path(datasource._run_summary_cache_key(source))
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not json", encoding="utf-8")
    calls = 0

    def build_summary(path):
        nonlocal calls
        calls += 1
        return {"rows": 8}

    monkeypatch.setattr(datasource, "run_summary", build_summary)
    assert datasource.cached_run_summary(source) == ({"rows": 8}, None)
    assert calls == 1


def test_concurrent_cold_summary_requests_share_one_computation(monkeypatch, tmp_path):
    source = tmp_path / "run.parquet"
    source.write_bytes(b"contents")
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def slow_summary(path):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return {"rows": 1}

    monkeypatch.setattr(datasource, "run_summary", slow_summary)
    results = []
    threads = [threading.Thread(target=lambda: results.append(datasource.cached_run_summary(source))) for _ in range(2)]

    threads[0].start()
    assert started.wait(timeout=2)
    threads[1].start()
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert calls == 1
    assert results == [({"rows": 1}, None), ({"rows": 1}, None)]

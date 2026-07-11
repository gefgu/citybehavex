from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from web.backend.app.cache import _key, get_or_build


def test_payload_cache_key_stays_within_filename_limits(tmp_path):
    synthetic = tmp_path / "gparis_simulation_core_trajectories_20260704T104640.parquet"
    observed = tmp_path / "gparis_observed_reference_trajectories_20260704T104640.parquet"
    extras = tuple(
        tmp_path / f"gparis_simulation_core_trajectories_20260704T104640_{suffix}.parquet"
        for suffix in (
            "social_network-1783172814",
            "activities-1783172814",
            "road_graph_nodes-1783171256",
            "road_graph_edges-1783171256",
        )
    )
    for path in (synthetic, observed, *extras):
        path.write_text("x", encoding="utf-8")

    cache_key = _key(
        "gparis_simulation_core",
        "20260704T104640__1783172814__1782927162",
        synthetic,
        observed,
        extras,
    )

    assert cache_key.endswith(".json")
    assert len(cache_key.encode("utf-8")) < 255


def test_payload_cache_key_changes_when_extra_inputs_change(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    observed = tmp_path / "observed.parquet"
    extra_a = tmp_path / "extra-a.parquet"
    extra_b = tmp_path / "extra-b.parquet"
    for path in (synthetic, observed, extra_a, extra_b):
        path.write_text("x", encoding="utf-8")

    key_a = _key("exp", "run", synthetic, observed, (extra_a,))
    key_b = _key("exp", "run", synthetic, observed, (extra_b,))

    assert key_a != key_b


def test_payload_cache_key_changes_with_extra_key(tmp_path):
    synthetic = tmp_path / "synthetic.parquet"
    synthetic.write_text("x", encoding="utf-8")

    key_a = _key("exp", "run", synthetic, None, extra_key={"demo": {"gender": "female"}})
    key_b = _key("exp", "run", synthetic, None, extra_key={"demo": {"gender": "male"}})
    key_same = _key("exp", "run", synthetic, None, extra_key={"demo": {"gender": "female"}})

    assert key_a != key_b
    assert key_a == key_same


def test_get_or_build_reads_from_cache_without_calling_build(tmp_path, monkeypatch):
    import web.backend.app.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    synthetic = tmp_path / "synthetic.parquet"
    synthetic.write_text("x", encoding="utf-8")

    calls = []

    def build():
        calls.append(1)
        return {"n": len(calls)}

    first = asyncio.run(get_or_build("exp", "run", synthetic, None, build_fn=build))
    second = asyncio.run(get_or_build("exp", "run", synthetic, None, build_fn=build))

    assert first == {"n": 1}
    assert second == {"n": 1}
    assert len(calls) == 1


def test_get_or_build_coalesces_concurrent_identical_calls(tmp_path, monkeypatch):
    """Two concurrent callers for the same still-uncached key must trigger
    exactly one build, not one each -- see cache.py's in-flight registry."""
    import web.backend.app.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    synthetic = tmp_path / "synthetic.parquet"
    synthetic.write_text("x", encoding="utf-8")

    calls = []
    started = threading.Event()
    release = threading.Event()

    def slow_build():
        calls.append(1)
        started.set()
        release.wait(timeout=5)
        return {"ok": True}

    async def run_two():
        with ThreadPoolExecutor(max_workers=2) as executor:
            async def first():
                return await get_or_build(
                    "exp", "run", synthetic, None, build_fn=slow_build, executor=executor
                )

            async def second():
                # Only start once the first call's build is actually running,
                # so this exercises a genuine concurrent-with-an-in-flight-build
                # call rather than two sequential ones.
                await asyncio.get_event_loop().run_in_executor(None, started.wait, 5)
                release.set()
                return await get_or_build(
                    "exp", "run", synthetic, None, build_fn=slow_build, executor=executor
                )

            return await asyncio.gather(first(), second())

    results = asyncio.run(run_two())

    assert results[0] == {"ok": True}
    assert results[1] == {"ok": True}
    assert len(calls) == 1


def test_cached_run_summary_reuses_unchanged_file_and_reloads_changed_file(tmp_path, monkeypatch):
    import web.backend.app.datasource as datasource_mod

    datasource_mod._run_summary_cache.clear()
    run = tmp_path / "synthetic.parquet"
    run.write_text("x", encoding="utf-8")
    calls = []

    def fake_run_summary(path):
        calls.append(path)
        return {"rows": len(calls)}

    monkeypatch.setattr(datasource_mod, "run_summary", fake_run_summary)

    first = datasource_mod.cached_run_summary(run)
    second = datasource_mod.cached_run_summary(run)
    run.write_text("xx", encoding="utf-8")
    bumped = int(run.stat().st_mtime) + 2
    os.utime(run, (bumped, bumped))
    third = datasource_mod.cached_run_summary(run)

    assert first == ({"rows": 1}, None)
    assert second == ({"rows": 1}, None)
    assert third == ({"rows": 2}, None)
    assert len(calls) == 2

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import export_static_web_demo


def test_validate_expected_agents_rejects_wrong_run_size(monkeypatch):
    experiment = SimpleNamespace(
        id="yjmob_simulation",
        runs=[SimpleNamespace(run_id="demo", path="demo.parquet")],
    )
    monkeypatch.setattr(
        export_static_web_demo,
        "run_summary",
        lambda path: {"uids": 50_000},
    )

    with pytest.raises(RuntimeError, match="manifest expected 500"):
        export_static_web_demo._validate_expected_agents(experiment, 500)


def test_validate_expected_agents_accepts_matching_run_size(monkeypatch):
    experiment = SimpleNamespace(
        id="gparis_simulation",
        runs=[SimpleNamespace(run_id="demo", path="demo.parquet")],
    )
    monkeypatch.setattr(
        export_static_web_demo,
        "run_summary",
        lambda path: {"uids": 500},
    )

    export_static_web_demo._validate_expected_agents(experiment, 500)

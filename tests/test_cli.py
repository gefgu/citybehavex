from __future__ import annotations

from typer.testing import CliRunner

from citybehavex.cli import app


def test_simulate_runs_with_config(monkeypatch):
    captured = {}

    def fake_run_simulation(config):
        captured["agents"] = config.simulation.agents

    monkeypatch.setattr("citybehavex.cli.run_simulation", fake_run_simulation)
    result = CliRunner().invoke(app, ["simulate", "--agents", "12"])

    assert result.exit_code == 0
    assert captured["agents"] == 12


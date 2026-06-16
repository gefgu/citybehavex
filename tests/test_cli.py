from __future__ import annotations

from typer.testing import CliRunner

from citybehavex.cli import app


def test_simulate_defaults_to_sts_epr(monkeypatch):
    captured = {}

    def fake_run_simulation(config):
        captured["model"] = config.simulation.model

    monkeypatch.setattr("citybehavex.cli.run_simulation", fake_run_simulation)
    result = CliRunner().invoke(app, ["simulate"])

    assert result.exit_code == 0
    assert captured["model"] == "sts_epr"


def test_simulate_ditras_flag_selects_ditras(monkeypatch):
    captured = {}

    def fake_run_simulation(config):
        captured["model"] = config.simulation.model

    monkeypatch.setattr("citybehavex.cli.run_simulation", fake_run_simulation)
    result = CliRunner().invoke(app, ["simulate", "--ditras"])

    assert result.exit_code == 0
    assert captured["model"] == "ditras"

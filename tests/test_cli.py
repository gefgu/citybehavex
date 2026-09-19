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


def test_init_creates_public_project(tmp_path):
    result = CliRunner().invoke(app, ["init", str(tmp_path / "demo")])

    assert result.exit_code == 0
    assert (tmp_path / "demo" / "README.md").exists()
    assert (tmp_path / "demo" / "configs" / "yjmob-1k.yaml").exists()


def test_init_rejects_existing_file_destination(tmp_path):
    target = tmp_path / "demo.txt"
    target.write_text("not a project directory")

    result = CliRunner().invoke(app, ["init", str(target)])

    assert result.exit_code == 1
    assert "not a directory" in result.output


def test_simulate_routes_to_temporary_aligners(monkeypatch):
    captured = {}

    class FakeServer:
        def __enter__(self):
            return "http://127.0.0.1:9999"

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("citybehavex.cli.temporary_aligners", lambda **kwargs: FakeServer())
    monkeypatch.setattr(
        "citybehavex.cli.run_simulation",
        lambda config: captured.update(
            embedding=config.embedding.base_url,
            schedule=config.schedule.alignment_base_url,
        ),
    )
    result = CliRunner().invoke(app, ["simulate", "--start-aligners"])

    assert result.exit_code == 0
    assert captured == {"embedding": "http://127.0.0.1:9999", "schedule": "http://127.0.0.1:9999"}

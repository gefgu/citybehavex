from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_cross_encoder.py"
SPEC = importlib.util.spec_from_file_location("serve_cross_encoder", SCRIPT_PATH)
server = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(server)


def test_role_defaults_and_custom_model_path():
    schedule_parser = __import__("argparse").ArgumentParser()
    server.add_server_arguments(
        schedule_parser,
        default_model="models/modernbert-schedule-aligner",
        default_port=8082,
    )
    schedule_args = schedule_parser.parse_args([])
    assert schedule_args.model_path == "models/modernbert-schedule-aligner"
    assert schedule_args.port == 8082

    activity_parser = __import__("argparse").ArgumentParser()
    server.add_server_arguments(
        activity_parser,
        default_model="models/modernbert-activity-aligner",
        default_port=8083,
    )
    activity_args = activity_parser.parse_args(
        ["--model-path", "my-modernbert-cross-encoder", "--port", "9003"]
    )
    assert activity_args.model_path == "my-modernbert-cross-encoder"
    assert activity_args.port == 9003


def test_run_server_loads_selected_model_and_role_defaults(monkeypatch):
    loaded = {}

    class FakeModel:
        def __init__(self, path, *, device):
            loaded.update(path=path, device=device)

    class FakeServer:
        def __init__(self, address, handler):
            loaded["address"] = address
            loaded["handler"] = handler

        def serve_forever(self):
            loaded["served"] = True

    monkeypatch.setattr(server, "CrossEncoder", FakeModel)
    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeServer)

    server.run_server(
        ["--model-path", "custom-model", "--device", "cuda:1"],
        role="activity",
        default_model="models/modernbert-activity-aligner",
        default_port=8083,
    )

    assert loaded["path"] == "custom-model"
    assert loaded["device"] == "cuda:1"
    assert loaded["address"] == ("127.0.0.1", 8083)
    assert loaded["served"] is True

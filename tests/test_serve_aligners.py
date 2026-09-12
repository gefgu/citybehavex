from __future__ import annotations

import http.client
import importlib.util
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_aligners.py"
SPEC = importlib.util.spec_from_file_location("serve_aligners", SCRIPT_PATH)
server = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(server)


class FakeCrossEncoder:
    instances: list[str] = []

    def __init__(self, path, *, device):
        self.path = path
        self.device = device
        FakeCrossEncoder.instances.append(path)

    def predict(self, pairs, batch_size):
        import numpy as np

        return np.array([0.5 for _ in pairs])


class FakeSentenceTransformer:
    instances: list[str] = []

    def __init__(self, path, *, trust_remote_code, device):
        self.path = path
        FakeSentenceTransformer.instances.append(path)

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        return np.array([[1.0, 2.0, 3.0] for _ in texts])


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch):
    FakeCrossEncoder.instances.clear()
    FakeSentenceTransformer.instances.clear()
    monkeypatch.setattr(server, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(server, "SentenceTransformer", FakeSentenceTransformer)


def _make_registry(**overrides):
    kwargs = dict(
        device="cpu",
        predict_batch_size=32,
        coalesce_window_s=0.001,
        coalesce_max_pairs=64,
        idle_ttl_s=600.0,
        idle_check_interval_s=600.0,
    )
    kwargs.update(overrides)
    return server.ModelRegistry(**kwargs)


def test_get_cross_encoder_loads_once_and_reuses(monkeypatch):
    registry = _make_registry()
    try:
        entry_a = registry.get_cross_encoder("models/modernbert-schedule-aligner")
        entry_b = registry.get_cross_encoder("models/modernbert-schedule-aligner")
        assert entry_a is entry_b
        assert FakeCrossEncoder.instances == ["models/modernbert-schedule-aligner"]

        registry.get_cross_encoder("models/modernbert-activity-aligner")
        assert FakeCrossEncoder.instances == [
            "models/modernbert-schedule-aligner",
            "models/modernbert-activity-aligner",
        ]
    finally:
        registry.stop()


def test_unload_evicts_idle_model_and_reports_missing():
    registry = _make_registry()
    try:
        registry.get_cross_encoder("models/modernbert-schedule-aligner")
        unloaded, reason = registry.unload("models/modernbert-schedule-aligner")
        assert unloaded is True
        assert reason == "unloaded"

        unloaded_again, reason_again = registry.unload("models/modernbert-schedule-aligner")
        assert unloaded_again is False
        assert reason_again == "not loaded"

        registry.get_cross_encoder("models/modernbert-schedule-aligner")
        assert FakeCrossEncoder.instances == [
            "models/modernbert-schedule-aligner",
            "models/modernbert-schedule-aligner",
        ]
    finally:
        registry.stop()


def test_unload_refuses_a_busy_model():
    registry = _make_registry()
    try:
        entry = registry.get_cross_encoder("models/modernbert-schedule-aligner")
        entry._touch_start()  # simulate a request in flight
        unloaded, reason = registry.unload("models/modernbert-schedule-aligner")
        assert unloaded is False
        assert reason == "busy"
    finally:
        registry.stop()


def test_idle_sweep_evicts_after_ttl():
    registry = _make_registry(idle_ttl_s=0.05, idle_check_interval_s=0.02)
    try:
        registry.get_cross_encoder("models/modernbert-schedule-aligner")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with registry._lock:
                if "models/modernbert-schedule-aligner" not in registry._entries:
                    break
            time.sleep(0.02)
        else:
            pytest.fail("idle model was never evicted")
    finally:
        registry.stop()


@pytest.fixture
def running_server():
    registry = _make_registry()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.build_handler(registry))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address
    finally:
        httpd.shutdown()
        registry.stop()


def _post(address, path, payload):
    conn = http.client.HTTPConnection(*address, timeout=5)
    try:
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()


def test_rerank_requires_model_field(running_server):
    status, payload = _post(running_server, "/rerank", {"query": "q", "texts": ["a"]})
    assert status == 400
    assert "model" in payload["error"]


def test_rerank_lazy_loads_and_reuses_across_requests(running_server):
    request = {
        "model": "models/modernbert-schedule-aligner",
        "query": "q",
        "texts": ["a", "b"],
    }
    status, payload = _post(running_server, "/rerank", request)
    assert status == 200
    assert [row["score"] for row in payload] == [0.5, 0.5]

    _post(running_server, "/rerank", request)
    assert FakeCrossEncoder.instances == ["models/modernbert-schedule-aligner"]


def test_unload_route_then_rerank_reloads(running_server):
    request = {"model": "models/modernbert-schedule-aligner", "query": "q", "texts": ["a"]}
    _post(running_server, "/rerank", request)
    status, payload = _post(running_server, "/unload", {"model": "models/modernbert-schedule-aligner"})
    assert status == 200
    assert payload == {"unloaded": True, "reason": "unloaded"}

    _post(running_server, "/rerank", request)
    assert FakeCrossEncoder.instances == [
        "models/modernbert-schedule-aligner",
        "models/modernbert-schedule-aligner",
    ]


def test_embeddings_route_lazy_loads_sentence_transformer(running_server):
    status, payload = _post(
        running_server,
        "/v1/embeddings",
        {"model": "nomic-ai/nomic-embed-text-v1.5", "input": ["hello", "world"]},
    )
    assert status == 200
    assert len(payload["data"]) == 2
    assert payload["data"][0]["embedding"] == [1.0, 2.0, 3.0]
    assert FakeSentenceTransformer.instances == ["nomic-ai/nomic-embed-text-v1.5"]

"""OpenAI-compatible local server for CityBehavEx aligners and embeddings."""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


class _CoalescingScorer:
    def __init__(self, model: CrossEncoder, batch_size: int, window_s: float, max_pairs: int) -> None:
        self.model, self.batch_size, self.window_s, self.max_pairs = model, batch_size, window_s, max_pairs
        self.queue: queue.Queue[tuple[list[list[str]], Future]] = queue.Queue()
        threading.Thread(target=self._drain, daemon=True).start()

    def score(self, pairs: list[list[str]]) -> list[float]:
        future: Future = Future()
        self.queue.put((pairs, future))
        return future.result()

    def _drain(self) -> None:
        while True:
            batch = [self.queue.get()]
            count = len(batch[0][0])
            deadline = time.monotonic() + self.window_s
            while count < self.max_pairs:
                try:
                    item = self.queue.get(timeout=max(0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                batch.append(item)
                count += len(item[0])
            merged: list[list[str]] = []
            spans: list[tuple[int, int]] = []
            for pairs, _ in batch:
                spans.append((len(merged), len(pairs)))
                merged.extend(pairs)
            try:
                scores = self.model.predict(merged, batch_size=self.batch_size).tolist()
                for (start, size), (_, future) in zip(spans, batch):
                    future.set_result(scores[start : start + size])
            except Exception as exc:  # noqa: BLE001
                for _, future in batch:
                    future.set_exception(exc)


class _Entry:
    def __init__(self, device: str) -> None:
        self.device, self.last_used, self.active = device, time.monotonic(), 0
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            self.active += 1

    def end(self) -> None:
        with self.lock:
            self.active -= 1
            self.last_used = time.monotonic()

    def close(self) -> None:
        raise NotImplementedError


class _CrossEncoderEntry(_Entry):
    def __init__(self, model_path: str, *, device: str, predict_batch_size: int, coalesce_window_s: float, coalesce_max_pairs: int) -> None:
        super().__init__(device)
        self.model = CrossEncoder(model_path, device=device)
        self.scorer = _CoalescingScorer(self.model, predict_batch_size, coalesce_window_s, coalesce_max_pairs)

    def score(self, pairs: list[list[str]]) -> list[float]:
        self.start()
        try:
            return self.scorer.score(pairs)
        finally:
            self.end()

    def close(self) -> None:
        del self.model, self.scorer
        _clear_cuda_cache(self.device)


class _EmbeddingEntry(_Entry):
    def __init__(self, model_path: str, *, device: str) -> None:
        super().__init__(device)
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.start()
        try:
            return [row.tolist() for row in self.model.encode(texts, convert_to_numpy=True)]
        finally:
            self.end()

    def close(self) -> None:
        del self.model
        _clear_cuda_cache(self.device)


class ModelRegistry:
    def __init__(self, *, device: str, predict_batch_size: int, coalesce_window_s: float, coalesce_max_pairs: int, idle_ttl_s: float, idle_check_interval_s: float) -> None:
        self.device, self.predict_batch_size = device, predict_batch_size
        self.coalesce_window_s, self.coalesce_max_pairs, self.idle_ttl_s = coalesce_window_s, coalesce_max_pairs, idle_ttl_s
        self._entries: dict[str, _Entry] = {}
        self._lock, self._stop = threading.Lock(), threading.Event()
        threading.Thread(target=self._sweep, args=(idle_check_interval_s,), daemon=True).start()

    def get_cross_encoder(self, model_path: str) -> _CrossEncoderEntry:
        with self._lock:
            entry = self._entries.get(model_path)
            if entry is None:
                entry = _CrossEncoderEntry(model_path, device=self.device, predict_batch_size=self.predict_batch_size, coalesce_window_s=self.coalesce_window_s, coalesce_max_pairs=self.coalesce_max_pairs)
                self._entries[model_path] = entry
            if not isinstance(entry, _CrossEncoderEntry):
                raise ValueError(f"model {model_path!r} is already loaded as an embedding model")
            return entry

    def get_embedder(self, model_path: str) -> _EmbeddingEntry:
        with self._lock:
            entry = self._entries.get(model_path)
            if entry is None:
                entry = _EmbeddingEntry(model_path, device=self.device)
                self._entries[model_path] = entry
            if not isinstance(entry, _EmbeddingEntry):
                raise ValueError(f"model {model_path!r} is already loaded as an aligner")
            return entry

    def unload(self, model_path: str) -> tuple[bool, str]:
        with self._lock:
            entry = self._entries.get(model_path)
            if entry is None:
                return False, "not loaded"
            if entry.active:
                return False, "busy"
            del self._entries[model_path]
        entry.close()
        return True, "unloaded"

    def _sweep(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            with self._lock:
                stale = [(name, entry) for name, entry in self._entries.items() if not entry.active and time.monotonic() - entry.last_used > self.idle_ttl_s]
                for name, _ in stale:
                    del self._entries[name]
            for _, entry in stale:
                entry.close()

    def stop(self) -> None:
        self._stop.set()


def build_handler(registry: ModelRegistry):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            _json_response(self, 200, {"status": "ok"}) if self.path in {"/", "/health"} else _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                path = self.path.rstrip("/")
                model = payload.get("model")
                if not model:
                    raise ValueError("payload must contain a 'model' field")
                if path == "/rerank":
                    pairs = [[payload["query"], str(text)] for text in payload["texts"]]
                    scores = registry.get_cross_encoder(model).score(pairs)
                    _json_response(self, 200, [{"index": i, "score": float(score)} for i, score in enumerate(scores)])
                elif path == "/score_pairs":
                    pairs = payload["pairs"]
                    if not all(isinstance(pair, list) and len(pair) == 2 for pair in pairs):
                        raise ValueError("each pair must contain query and text")
                    scores = registry.get_cross_encoder(model).score([[str(x) for x in pair] for pair in pairs])
                    _json_response(self, 200, [{"index": i, "score": float(score)} for i, score in enumerate(scores)])
                elif path == "/v1/embeddings":
                    texts = payload.get("input")
                    texts = [texts] if isinstance(texts, str) else texts
                    if not isinstance(texts, list):
                        raise ValueError("payload must contain a string or list 'input'")
                    vectors = registry.get_embedder(model).embed([str(text) for text in texts])
                    _json_response(self, 200, {"data": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)], "model": model})
                elif path == "/unload":
                    unloaded, reason = registry.unload(model)
                    _json_response(self, 200, {"unloaded": unloaded, "reason": reason})
                else:
                    _json_response(self, 404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, 400, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            return
    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve CityBehavEx aligners and embedding models.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--coalesce-window-ms", type=float, default=20.0)
    parser.add_argument("--coalesce-max-pairs", type=int, default=2048)
    parser.add_argument("--idle-ttl-s", type=float, default=600.0)
    parser.add_argument("--idle-check-interval-s", type=float, default=30.0)
    return parser


def run_server(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; use --device cpu for slower CPU inference.")
    registry = ModelRegistry(device=args.device, predict_batch_size=args.predict_batch_size, coalesce_window_s=args.coalesce_window_ms / 1000, coalesce_max_pairs=args.coalesce_max_pairs, idle_ttl_s=args.idle_ttl_s, idle_check_interval_s=args.idle_check_interval_s)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(registry))
    print(f"Serving aligners on http://{args.host}:{args.port} (device={args.device})", flush=True)
    try:
        server.serve_forever()
    finally:
        registry.stop()
        server.server_close()


if __name__ == "__main__":
    run_server()

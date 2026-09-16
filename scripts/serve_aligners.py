#!/usr/bin/env python
"""Consolidated lazy-loading server for CrossEncoder aligners and embedding models.

Replaces the old one-process-per-aligner setup (serve_cross_encoder.py +
serve_schedule_aligner.py + serve_activity_aligner.py, one port each). Clients
already send a "model" field on every request (citybehavex/utils/alignment.py's
post_rerank_scores/post_pair_scores, citybehavex/embedding/service.py's
_post_embeddings) naming the exact checkpoint path or model id they want; this
server uses that field to lazily load and cache CrossEncoders (for /rerank,
/score_pairs) and embedding models (for /v1/embeddings) by that same string, and
evicts idle ones after --idle-ttl-s. POST /unload forces an immediate evict.
"""
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


class _CoalescingScorer:
    """Batch concurrent requests into larger CrossEncoder.predict calls."""

    def __init__(
        self,
        model: CrossEncoder,
        predict_batch_size: int,
        coalesce_window_s: float,
        coalesce_max_pairs: int,
    ) -> None:
        self._model = model
        self._predict_batch_size = predict_batch_size
        self._window_s = coalesce_window_s
        self._max_pairs = coalesce_max_pairs
        self._queue: queue.Queue[tuple[list[list[str]], Future]] = queue.Queue()
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def score(self, pairs: list[list[str]]) -> list[float]:
        future: Future = Future()
        self._queue.put((pairs, future))
        return future.result()

    def _drain_loop(self) -> None:
        while True:
            batch: list[tuple[list[list[str]], Future]] = [self._queue.get()]
            total_pairs = len(batch[0][0])
            deadline = time.monotonic() + self._window_s
            while total_pairs < self._max_pairs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                batch.append(item)
                total_pairs += len(item[0])

            merged_pairs: list[list[str]] = []
            spans: list[tuple[int, int]] = []
            for pairs, _future in batch:
                spans.append((len(merged_pairs), len(pairs)))
                merged_pairs.extend(pairs)

            try:
                scores = self._model.predict(
                    merged_pairs, batch_size=self._predict_batch_size
                ).tolist()
            except Exception as exc:  # noqa: BLE001 - propagate to every waiter.
                for _pairs, future in batch:
                    future.set_exception(exc)
                continue

            for (offset, length), (_pairs, future) in zip(spans, batch):
                future.set_result(scores[offset : offset + length])


class _Entry:
    """Common lazy-load bookkeeping (last-used, in-flight count) for one model."""

    def __init__(self) -> None:
        self.last_used = time.monotonic()
        self.active = 0
        self._lock = threading.Lock()

    def _touch_start(self) -> None:
        with self._lock:
            self.active += 1

    def _touch_end(self) -> None:
        with self._lock:
            self.active -= 1
            self.last_used = time.monotonic()

    def close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class _CrossEncoderEntry(_Entry):
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        predict_batch_size: int,
        coalesce_window_s: float,
        coalesce_max_pairs: int,
    ) -> None:
        super().__init__()
        self._model = CrossEncoder(model_path, device=device)
        self._scorer = _CoalescingScorer(
            self._model, predict_batch_size, coalesce_window_s, coalesce_max_pairs
        )

    def score(self, pairs: list[list[str]]) -> list[float]:
        self._touch_start()
        try:
            return self._scorer.score(pairs)
        finally:
            self._touch_end()

    def close(self) -> None:
        del self._model
        del self._scorer
        torch.cuda.empty_cache()


class _EmbeddingEntry(_Entry):
    def __init__(self, model_path: str, *, device: str) -> None:
        super().__init__()
        self._model = SentenceTransformer(model_path, trust_remote_code=True, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._touch_start()
        try:
            vectors = self._model.encode(texts, convert_to_numpy=True)
            return [row.tolist() for row in vectors]
        finally:
            self._touch_end()

    def close(self) -> None:
        del self._model
        torch.cuda.empty_cache()


class ModelRegistry:
    """Lazily loads CrossEncoders/embedding models by id, evicting idle ones."""

    def __init__(
        self,
        *,
        device: str,
        predict_batch_size: int,
        coalesce_window_s: float,
        coalesce_max_pairs: int,
        idle_ttl_s: float,
        idle_check_interval_s: float,
    ) -> None:
        self._device = device
        self._predict_batch_size = predict_batch_size
        self._coalesce_window_s = coalesce_window_s
        self._coalesce_max_pairs = coalesce_max_pairs
        self._idle_ttl_s = idle_ttl_s
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop, args=(idle_check_interval_s,), daemon=True
        )
        self._sweep_thread.start()

    def get_cross_encoder(self, model_path: str) -> _CrossEncoderEntry:
        with self._lock:
            entry = self._entries.get(model_path)
            if entry is None:
                print(f"Loading CrossEncoder {model_path!r}", flush=True)
                entry = _CrossEncoderEntry(
                    model_path,
                    device=self._device,
                    predict_batch_size=self._predict_batch_size,
                    coalesce_window_s=self._coalesce_window_s,
                    coalesce_max_pairs=self._coalesce_max_pairs,
                )
                self._entries[model_path] = entry
            assert isinstance(entry, _CrossEncoderEntry)
            return entry

    def get_embedder(self, model_path: str) -> _EmbeddingEntry:
        with self._lock:
            entry = self._entries.get(model_path)
            if entry is None:
                print(f"Loading embedding model {model_path!r}", flush=True)
                entry = _EmbeddingEntry(model_path, device=self._device)
                self._entries[model_path] = entry
            assert isinstance(entry, _EmbeddingEntry)
            return entry

    def unload(self, model_path: str) -> tuple[bool, str]:
        with self._lock:
            entry = self._entries.get(model_path)
            if entry is None:
                return False, "not loaded"
            if entry.active > 0:
                return False, "busy"
            del self._entries[model_path]
        entry.close()
        print(f"Unloaded {model_path!r}", flush=True)
        return True, "unloaded"

    def _sweep_loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            now = time.monotonic()
            to_close: list[tuple[str, _Entry]] = []
            with self._lock:
                for model_path, entry in list(self._entries.items()):
                    if entry.active == 0 and now - entry.last_used > self._idle_ttl_s:
                        to_close.append((model_path, entry))
                for model_path, _entry in to_close:
                    del self._entries[model_path]
            for model_path, entry in to_close:
                entry.close()
                print(f"Evicted idle model {model_path!r}", flush=True)

    def stop(self) -> None:
        self._stop.set()


def build_handler(registry: ModelRegistry):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            if self.path in {"/health", "/"}:
                _json_response(self, 200, {"status": "ok"})
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            path = self.path.rstrip("/")
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                if path in {"/rerank", "/score_pairs"}:
                    self._handle_score(path, payload)
                elif path == "/v1/embeddings":
                    self._handle_embeddings(payload)
                elif path == "/unload":
                    self._handle_unload(payload)
                else:
                    _json_response(self, 404, {"error": "not found"})
            except Exception as exc:  # noqa: BLE001 - convert to HTTP error.
                _json_response(self, 400, {"error": str(exc)})

        def _handle_score(self, path: str, payload: dict[str, Any]) -> None:
            model = payload.get("model")
            if not model:
                raise ValueError("payload must contain a 'model' field")
            if path == "/score_pairs":
                raw_pairs = payload["pairs"]
                if not isinstance(raw_pairs, list):
                    raise ValueError("payload must contain list pairs")
                pairs = []
                for pair in raw_pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        raise ValueError("each pair must contain query and text")
                    pairs.append([str(pair[0]), str(pair[1])])
            else:
                query = payload["query"]
                texts = payload["texts"]
                if not isinstance(query, str) or not isinstance(texts, list):
                    raise ValueError("payload must contain string query and list texts")
                pairs = [[query, str(text)] for text in texts]
            scores = registry.get_cross_encoder(model).score(pairs)
            _json_response(
                self,
                200,
                [{"index": idx, "score": float(score)} for idx, score in enumerate(scores)],
            )

        def _handle_embeddings(self, payload: dict[str, Any]) -> None:
            model = payload.get("model")
            if not model:
                raise ValueError("payload must contain a 'model' field")
            texts = payload.get("input")
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(texts, list):
                raise ValueError("payload must contain a string or list 'input'")
            vectors = registry.get_embedder(model).embed([str(text) for text in texts])
            _json_response(
                self,
                200,
                {
                    "data": [
                        {"index": idx, "embedding": vector} for idx, vector in enumerate(vectors)
                    ],
                    "model": model,
                },
            )

        def _handle_unload(self, payload: dict[str, Any]) -> None:
            model = payload.get("model")
            if not model:
                raise ValueError("payload must contain a 'model' field")
            unloaded, reason = registry.unload(model)
            _json_response(self, 200, {"unloaded": unloaded, "reason": reason})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve CrossEncoder aligners and embedding models, lazily loaded by model id."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=128,
        help="Batch size used inside CrossEncoder.predict for each merged request.",
    )
    parser.add_argument(
        "--coalesce-window-ms",
        type=float,
        default=20.0,
        help="Maximum wait after the first request for concurrent requests to arrive.",
    )
    parser.add_argument(
        "--coalesce-max-pairs",
        type=int,
        default=2048,
        help="Maximum number of queued pairs to merge into one prediction.",
    )
    parser.add_argument(
        "--idle-ttl-s",
        type=float,
        default=600.0,
        help="Evict a model after this many seconds with no requests.",
    )
    parser.add_argument(
        "--idle-check-interval-s",
        type=float,
        default=30.0,
        help="How often the idle-eviction sweep runs.",
    )
    return parser


def run_server(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    registry = ModelRegistry(
        device=args.device,
        predict_batch_size=args.predict_batch_size,
        coalesce_window_s=args.coalesce_window_ms / 1000.0,
        coalesce_max_pairs=args.coalesce_max_pairs,
        idle_ttl_s=args.idle_ttl_s,
        idle_check_interval_s=args.idle_check_interval_s,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(registry))
    print(
        f"Serving aligners on http://{args.host}:{args.port} "
        f"(lazy load, idle_ttl_s={args.idle_ttl_s})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    run_server()

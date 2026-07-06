#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sentence_transformers.cross_encoder import CrossEncoder


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


class _CoalescingScorer:
    """Batches concurrent /rerank and /score_pairs requests into fewer,
    larger `CrossEncoder.predict()` calls.

    Without this, a `ThreadingHTTPServer` handling N concurrent requests would
    call `model.predict()` N times independently -- CUDA serializes those
    kernel launches on the default stream, so concurrency alone doesn't grow
    the GPU's effective batch size, it mostly just contends. This collects
    whatever requests land within a short window into one merged `predict()`
    call instead.
    """

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


def build_handler(scorer: _CoalescingScorer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            if self.path in {"/health", "/"}:
                _json_response(self, 200, {"status": "ok"})
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            path = self.path.rstrip("/")
            if path not in {"/rerank", "/score_pairs"}:
                _json_response(self, 404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
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
                scores = scorer.score(pairs)
            except Exception as exc:  # noqa: BLE001 - convert to HTTP error.
                _json_response(self, 400, {"error": str(exc)})
                return
            _json_response(
                self,
                200,
                [{"index": idx, "score": float(score)} for idx, score in enumerate(scores)],
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the schedule alignment CrossEncoder.")
    parser.add_argument("--model-path", default="models/modernbert-schedule-aligner")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=128,
        help="Batch size used inside CrossEncoder.predict for each merged request. "
        "128 measured best on a GPU shared with other resident processes (RTX "
        "5090 alongside a persistent vLLM engine) -- 256/512 measured slightly "
        "*worse* (~1400 vs ~1650 pairs/sec) there, likely SM/memory-bandwidth "
        "contention rather than headroom; re-measure if the deployment changes.",
    )
    parser.add_argument(
        "--coalesce-window-ms",
        type=float,
        default=20.0,
        help="After the first queued request, wait up to this long for more "
        "concurrent requests to arrive before running one merged predict() call.",
    )
    parser.add_argument(
        "--coalesce-max-pairs",
        type=int,
        default=2048,
        help="Stop collecting more requests into a merged batch once this many "
        "pairs have been queued, even if the coalesce window hasn't elapsed.",
    )
    args = parser.parse_args()

    model = CrossEncoder(args.model_path, device=args.device)
    scorer = _CoalescingScorer(
        model,
        predict_batch_size=args.predict_batch_size,
        coalesce_window_s=args.coalesce_window_ms / 1000.0,
        coalesce_max_pairs=args.coalesce_max_pairs,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(scorer))
    print(f"Serving schedule aligner on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

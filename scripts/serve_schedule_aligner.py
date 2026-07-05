#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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


def build_handler(model: CrossEncoder):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            if self.path in {"/health", "/"}:
                _json_response(self, 200, {"status": "ok"})
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            if self.path.rstrip("/") != "/rerank":
                _json_response(self, 404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                query = payload["query"]
                texts = payload["texts"]
                if not isinstance(query, str) or not isinstance(texts, list):
                    raise ValueError("payload must contain string query and list texts")
                pairs = [[query, str(text)] for text in texts]
                scores = model.predict(pairs).tolist()
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
    args = parser.parse_args()

    model = CrossEncoder(args.model_path, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(model))
    print(f"Serving schedule aligner on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

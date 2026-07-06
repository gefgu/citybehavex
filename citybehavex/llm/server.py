from __future__ import annotations

import contextlib
import subprocess
import time
from pathlib import Path
from typing import Iterator

import requests

from .config import LLMConfig


def server_reachable(base_url: str, timeout: float) -> bool:
    for path in ("/health", "/v1/models"):
        try:
            resp = requests.get(base_url.rstrip("/") + path, timeout=timeout)
            if resp.ok:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


@contextlib.contextmanager
def vllm_server(config: LLMConfig, *, log_dir: Path | str) -> Iterator[str]:
    """Spawn a local vLLM chat-completions server, yield its base_url, then shut it down."""
    port = config.vllm_port
    base_url = f"http://127.0.0.1:{port}"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "vllm_llm.log"

    cmd = [
        "vllm",
        "serve",
        config.model,
        "--trust-remote-code",
        "--port",
        str(port),
        *config.vllm_extra_args,
    ]
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + config.vllm_startup_timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vllm exited early (code {proc.returncode}); see {log_path}"
                    )
                if server_reachable(base_url, timeout=2.0):
                    break
                time.sleep(2.0)
            else:
                raise TimeoutError(
                    f"vllm did not become ready within "
                    f"{config.vllm_startup_timeout_seconds:.0f}s; see {log_path}"
                )
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


@contextlib.contextmanager
def resolve_llm_server(config: LLMConfig, *, log_dir: Path | str) -> Iterator[str]:
    """Yield a reachable base_url: reuse an already-running server, else auto-launch one."""
    use_auto = config.auto_launch and not (
        config.base_url and server_reachable(config.base_url, timeout=5.0)
    )
    if use_auto:
        with vllm_server(config, log_dir=log_dir) as url:
            yield url
    else:
        yield config.base_url or ""

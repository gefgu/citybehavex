"""CLI helpers for temporary local CityBehavEx services."""
from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from typing import Iterator

import requests

from citybehavex.config.root import CityBehavExConfig


def service_reachable(base_url: str, timeout: float = 2.0) -> bool:
    try:
        return requests.get(base_url.rstrip("/") + "/health", timeout=timeout).ok
    except requests.RequestException:
        return False


@contextlib.contextmanager
def temporary_aligners(*, port: int, device: str, startup_timeout: float = 90.0) -> Iterator[str]:
    """Start an installed aligner server and terminate exactly that child on exit."""
    base_url = f"http://127.0.0.1:{port}"
    if service_reachable(base_url):
        raise RuntimeError(f"aligner port {port} is already serving a process; choose --aligner-port")
    command = [sys.executable, "-m", "citybehavex.aligners.server", "--port", str(port), "--device", device]
    proc = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"aligner server exited early (code {proc.returncode})")
            if service_reachable(base_url):
                yield base_url
                return
            time.sleep(0.25)
        raise TimeoutError(f"aligner server did not become ready within {startup_timeout:.0f}s")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def route_to_local_aligners(config: CityBehavExConfig, base_url: str) -> CityBehavExConfig:
    """Return an in-memory config that sends all aligner/embed calls to one service."""
    return config.model_copy(
        update={
            "embedding": config.embedding.model_copy(
                update={"base_url": base_url, "auto_launch": False}
            ),
            "schedule": config.schedule.model_copy(update={"alignment_base_url": base_url}),
            "activities": config.activities.model_copy(update={"alignment_base_url": base_url}),
            "profiles": config.profiles.model_copy(
                update={
                    "coherence_alignment_base_url": base_url,
                    "ownership_alignment_base_url": base_url,
                }
            ),
        }
    )

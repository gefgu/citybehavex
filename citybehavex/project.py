"""Public project templates, sample-data downloads, and environment diagnostics."""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import random
import shutil
import tarfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from citybehavex.config import load_config
from citybehavex.services import service_reachable

_MANIFEST = "yjmob-1k-manifest.json"
_DOWNLOAD_ATTEMPTS = 4
_DOWNLOAD_TIMEOUT_SECONDS = 60
_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


def init_project(destination: Path) -> Path:
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"destination is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    source = importlib.resources.files("citybehavex.templates").joinpath("yjmob-1k")
    with importlib.resources.as_file(source) as source_path:
        shutil.copytree(source_path, destination, dirs_exist_ok=True)
    return destination


def _manifest() -> dict[str, object]:
    return json.loads(importlib.resources.files("citybehavex.templates").joinpath(_MANIFEST).read_text())


def _retryable_download_error(exc: requests.RequestException) -> bool:
    """Whether a failed release-asset request is worth retrying."""
    if not isinstance(exc, requests.HTTPError):
        return True
    response = exc.response
    return response is not None and response.status_code in _RETRYABLE_HTTP_STATUSES


def _download_archive(url: str, archive: Path, expected_sha256: str) -> None:
    """Download a release asset safely, retrying transient transport failures."""
    partial = archive.with_suffix(archive.suffix + ".part")
    last_error: requests.RequestException | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        try:
            response = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
            try:
                response.raise_for_status()
                digest = hashlib.sha256()
                with partial.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            digest.update(chunk)
            finally:
                response.close()
        except requests.RequestException as exc:
            partial.unlink(missing_ok=True)
            if not _retryable_download_error(exc):
                raise
            last_error = exc
            if attempt < _DOWNLOAD_ATTEMPTS - 1:
                delay = min(8.0, 0.5 * (2**attempt))
                time.sleep(delay + random.uniform(0, delay * 0.5))
            continue

        if digest.hexdigest() != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ValueError("YJMOB sample checksum mismatch")
        partial.replace(archive)
        return

    raise RuntimeError(
        f"download failed after {_DOWNLOAD_ATTEMPTS} attempts; please retry later or use another network"
    ) from last_error


def download_yjmob(destination: Path) -> Path:
    manifest = _manifest()
    url, expected = str(manifest["url"]), str(manifest["sha256"])
    if expected.startswith("REPLACE_"):
        raise ValueError(
            "the YJMOB-1k release asset has not been published yet; "
            "maintainers must update the packaged manifest checksum"
        )
    if not url.startswith("https://"):
        raise ValueError("sample manifest requires an HTTPS URL")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / Path(urlparse(url).path).name
    _download_archive(url, archive, expected)
    with tarfile.open(archive, "r:gz") as bundle:
        root = destination.resolve()
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError("YJMOB sample archive contains an unsafe path")
        bundle.extractall(destination, filter="data")
    archive.unlink(missing_ok=True)
    missing = [name for name in manifest["files"] if not (destination / str(name)).exists()]
    if missing:
        raise ValueError("YJMOB sample is missing expected files: " + ", ".join(missing))
    return destination


def doctor(config_path: str) -> list[str]:
    config = load_config(config_path)
    messages = ["CityBehavEx core: OK"]
    for label, value in (("tessellation", config.tessellation.output), ("comparison", config.comparison.path)):
        if value and not Path(value).exists():
            messages.append(f"Missing {label} input: {value}")
    if config.llm.base_url:
        try:
            response = requests.get(config.llm.base_url.rstrip("/") + "/v1/models", timeout=10)
            response.raise_for_status()
            model_ids = [item.get("id", "<unnamed>") for item in response.json().get("data", [])]
            messages.append("LLM endpoint: OK (" + ", ".join(model_ids or ["no models reported"]) + ")")
        except requests.RequestException as exc:
            messages.append(f"LLM endpoint unavailable: {exc}")
    else:
        messages.append("LLM endpoint: not configured")
    if config.schedule.alignment_base_url:
        status = "OK" if service_reachable(config.schedule.alignment_base_url) else "not reachable"
        messages.append(f"Configured aligner: {status}")
    return messages

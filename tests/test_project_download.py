from __future__ import annotations

import hashlib
import io
import tarfile

import pytest
import requests

from citybehavex import project


def _sample_archive() -> bytes:
    files = {
        "data/yjmob-1k/yjmob_h3_tessellation.parquet": b"tessellation",
        "data/yjmob-1k/yjmob_1k_simulated_sample.parquet": b"sample",
    }
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return content.getvalue()


class _Response:
    def __init__(self, chunks: list[bytes], error: Exception | None = None):
        self.chunks = chunks
        self.error = error
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


def _manifest(payload: bytes) -> dict[str, object]:
    return {
        "url": "https://example.invalid/citybehavex-yjmob-1k.tar.gz",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": [
            "data/yjmob-1k/yjmob_h3_tessellation.parquet",
            "data/yjmob-1k/yjmob_1k_simulated_sample.parquet",
        ],
    }


def test_yjmob_download_retries_a_reset_during_streaming(monkeypatch, tmp_path):
    payload = _sample_archive()
    calls = 0

    def fake_get(url, *, stream, timeout):
        nonlocal calls
        calls += 1
        assert url.startswith("https://")
        assert stream is True
        assert timeout == project._DOWNLOAD_TIMEOUT_SECONDS
        if calls == 1:
            return _Response([payload[:10]], requests.ConnectionError("connection reset"))
        return _Response([payload])

    monkeypatch.setattr(project, "_manifest", lambda: _manifest(payload))
    monkeypatch.setattr(project.requests, "get", fake_get)
    monkeypatch.setattr(project.time, "sleep", lambda seconds: None)

    assert project.download_yjmob(tmp_path) == tmp_path
    assert calls == 2
    assert (tmp_path / "data/yjmob-1k/yjmob_h3_tessellation.parquet").read_bytes() == b"tessellation"
    assert not list(tmp_path.glob("*.part"))


def test_yjmob_download_removes_partial_file_after_retries_fail(monkeypatch, tmp_path):
    payload = _sample_archive()
    calls = 0

    def fake_get(url, *, stream, timeout):
        nonlocal calls
        calls += 1
        raise requests.ConnectionError("connection reset")

    monkeypatch.setattr(project, "_manifest", lambda: _manifest(payload))
    monkeypatch.setattr(project.requests, "get", fake_get)
    monkeypatch.setattr(project.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="download failed after"):
        project.download_yjmob(tmp_path)

    assert calls == project._DOWNLOAD_ATTEMPTS
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.tar.gz"))

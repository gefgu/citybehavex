from __future__ import annotations

import duckdb
import pytest

from citybehavex.tessellation.overture import fetch_with_overture_release_fallback


def test_fetch_with_overture_release_fallback_returns_result_on_success():
    calls: list[str] = []

    def query(release: str) -> str:
        calls.append(release)
        return f"data-for-{release}"

    result = fetch_with_overture_release_fallback("2026-05-20.0", query)

    assert result == "data-for-2026-05-20.0"
    assert calls == ["2026-05-20.0"]


def test_fetch_with_overture_release_fallback_retries_with_latest_release(monkeypatch):
    calls: list[str] = []

    def query(release: str) -> str:
        calls.append(release)
        if release == "2026-05-20.0":
            raise duckdb.IOException(
                'IO Error: No files found that match the pattern '
                '"s3://overturemaps-us-west-2/release/2026-05-20.0/theme=buildings/type=*/*"'
            )
        return f"data-for-{release}"

    monkeypatch.setattr(
        "citybehavex.tessellation.overture.resolve_latest_overture_release",
        lambda: "2026-08-19.0",
    )

    result = fetch_with_overture_release_fallback("2026-05-20.0", query)

    assert result == "data-for-2026-08-19.0"
    assert calls == ["2026-05-20.0", "2026-08-19.0"]


def test_fetch_with_overture_release_fallback_reraises_unrelated_io_errors(monkeypatch):
    def query(release: str) -> str:
        raise duckdb.IOException("IO Error: connection reset")

    monkeypatch.setattr(
        "citybehavex.tessellation.overture.resolve_latest_overture_release",
        lambda: pytest.fail("should not attempt to resolve the latest release"),
    )

    with pytest.raises(duckdb.IOException, match="connection reset"):
        fetch_with_overture_release_fallback("2026-05-20.0", query)


def test_fetch_with_overture_release_fallback_reraises_when_already_latest(monkeypatch):
    def query(release: str) -> str:
        raise duckdb.IOException(
            'IO Error: No files found that match the pattern '
            '"s3://overturemaps-us-west-2/release/2026-08-19.0/theme=buildings/type=*/*"'
        )

    monkeypatch.setattr(
        "citybehavex.tessellation.overture.resolve_latest_overture_release",
        lambda: "2026-08-19.0",
    )

    with pytest.raises(duckdb.IOException, match="No files found"):
        fetch_with_overture_release_fallback("2026-08-19.0", query)

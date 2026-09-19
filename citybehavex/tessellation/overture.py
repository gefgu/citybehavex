"""Overture Maps release resolution and staleness recovery.

Overture Maps keeps only a rolling window of releases on S3 (typically the
last two) and deletes older ones outright -- a release string that worked
when it was pinned into a config eventually 404s. Rather than periodically
bumping a hardcoded date, callers wrap their release-scoped query with
``fetch_with_overture_release_fallback``: it runs the query as configured
and, only if that specific "release no longer exists" error occurs, looks
up the current latest release and retries once.
"""
from __future__ import annotations

from typing import Callable, TypeVar
from xml.etree import ElementTree

import duckdb
import requests
import typer

_BUCKET_URL = "https://overturemaps-us-west-2.s3.amazonaws.com/"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

T = TypeVar("T")


def resolve_latest_overture_release(timeout: float = 10.0) -> str:
    """Return the most recent Overture Maps release name available on S3."""
    response = requests.get(
        _BUCKET_URL,
        params={"list-type": "2", "delimiter": "/", "prefix": "release/"},
        timeout=timeout,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    prefixes = [
        element.text
        for element in root.findall(".//s3:CommonPrefixes/s3:Prefix", _S3_NS)
        if element.text
    ]
    releases = [prefix.removeprefix("release/").removesuffix("/") for prefix in prefixes]
    releases = [release for release in releases if release]
    if not releases:
        raise ValueError("no Overture Maps releases found on S3")
    return max(releases)


def fetch_with_overture_release_fallback(
    configured_release: str, query: Callable[[str], T]
) -> T:
    """Run ``query(configured_release)``, retrying once with the current
    latest release if the configured one no longer exists on S3."""
    try:
        return query(configured_release)
    except duckdb.IOException as exc:
        if "No files found that match the pattern" not in str(exc):
            raise
        latest_release = resolve_latest_overture_release()
        if latest_release == configured_release:
            raise
        typer.echo(
            f"Warning: Overture Maps release {configured_release!r} is no longer "
            f"available; retrying with the current latest release {latest_release!r}. "
            "Consider updating tessellation.overture_release in your config.",
            err=True,
        )
        return query(latest_release)

"""Progressive section builders for the comparison dashboard."""

from __future__ import annotations

from typing import Any, Optional

from . import legacy
from .context import ComparisonContext
from .store import artifact_store

SECTION_NAMES = [
    "distributions",
    "metrics",
    "transport-spatial",
    "activity",
    "mobility-laws",
    "micro-activity",
    "time-use",
    "motifs",
    "stvd",
    "profiles",
    "social-network",
]


def _empty_metrics() -> dict[str, list[Any]]:
    return {"wasserstein": [], "jsd": [], "cpc": [], "time_use": [], "stvd": []}


def _labels(ctx: ComparisonContext) -> dict[str, str]:
    labels = {"synthetic": "synthetic"}
    if ctx.observed_path:
        labels["observed"] = ctx.observed_label
    return labels


def _empty_payload(ctx: ComparisonContext, loaded_filters: Optional[list[str]] = None) -> dict[str, Any]:
    available_filters = legacy._filter_options(ctx.special_day_dicts())
    distribution_filters = legacy._distribution_filter_options(ctx.special_day_dicts())
    return {
        "mode": "comparison" if ctx.observed_path else "synthetic_only",
        "labels": _labels(ctx),
        "available_filters": [legacy._public_filter(meta) for meta in available_filters],
        "distribution_filters": [legacy._public_filter(meta) for meta in distribution_filters],
        "enabled_sections": SECTION_NAMES,
        "loaded_filters": loaded_filters or [],
        "metrics": _empty_metrics(),
        "ecdf": {"groups": []},
        "transport_spatial": None,
        "mobility_laws": None,
        "activity": None,
        "micro_activity_usage": None,
        "time_use_comparison": None,
        "profiles": None,
        "motifs": None,
        "stvd": None,
        "social_network": None,
        "warnings": [],
    }


def _context_from_kwargs(**kwargs: Any) -> ComparisonContext:
    return ComparisonContext.from_kwargs(**kwargs)


def _regular_artifact(ctx: ComparisonContext, filter_key: str, section: str) -> dict[str, Any]:
    key = (*ctx.artifact_key(filter_key), "regular", section)
    return artifact_store.get_or_build(
        key,
        lambda: legacy._build_comparison_payload(
            synthetic_path=ctx.synthetic_path,
            observed_path=ctx.observed_path,
            observed_label=ctx.observed_label,
            synthetic_activities_path=ctx.synthetic_activities_path,
            time_use_path=ctx.time_use_path,
            time_use_label=ctx.time_use_label,
            time_use_country=ctx.time_use_country,
            time_use_survey=ctx.time_use_survey,
            time_use_weight_col=ctx.time_use_weight_col,
            transport_spatial_config=ctx.transport_spatial_config,
            special_days=ctx.special_day_dicts(),
            filter_keys=[filter_key],
            include_progressive_metadata=True,
            include_profiles=False,
            include_social_network=False,
            sections=[section],
        ),
    )


def _profile_artifact(ctx: ComparisonContext) -> dict[str, Any]:
    key = (*ctx.artifact_key("all"), "profiles")
    return artifact_store.get_or_build(
        key,
        lambda: legacy._build_comparison_payload(
            synthetic_path=ctx.synthetic_path,
            observed_path=ctx.observed_path,
            observed_label=ctx.observed_label,
            synthetic_activities_path=ctx.synthetic_activities_path,
            time_use_path=ctx.time_use_path,
            time_use_label=ctx.time_use_label,
            time_use_country=ctx.time_use_country,
            time_use_survey=ctx.time_use_survey,
            time_use_weight_col=ctx.time_use_weight_col,
            transport_spatial_config=ctx.transport_spatial_config,
            special_days=ctx.special_day_dicts(),
            filter_keys=["all"],
            include_progressive_metadata=True,
            include_profiles=True,
            include_social_network=False,
        ),
    )


def build_chart_base_payload(
    synthetic_path: str,
    observed_path: Optional[str],
    observed_label: str,
    synthetic_activities_path: Optional[str] = None,
    time_use_path: Optional[str] = None,
    time_use_label: str = "time-use",
    time_use_country: Optional[str] = None,
    time_use_survey: Optional[int] = None,
    time_use_weight_col: str = "propwt",
    transport_spatial_config: Optional[object] = None,
    special_days: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    kwargs = dict(
        synthetic_path=synthetic_path,
        observed_path=observed_path,
        observed_label=observed_label,
        synthetic_activities_path=synthetic_activities_path,
        time_use_path=time_use_path,
        time_use_label=time_use_label,
        time_use_country=time_use_country,
        time_use_survey=time_use_survey,
        time_use_weight_col=time_use_weight_col,
        transport_spatial_config=transport_spatial_config,
        special_days=special_days,
    )
    ctx = _context_from_kwargs(**kwargs)
    payload = _empty_payload(ctx, loaded_filters=[])
    if kwargs.get("observed_path") and not ctx.observed_path:
        payload["warnings"].append(f"observed comparison parquet not found: {kwargs['observed_path']}")
    return payload


def build_metrics_export_payload(
    synthetic_path: str,
    observed_path: Optional[str],
    observed_label: str,
    synthetic_activities_path: Optional[str] = None,
    time_use_path: Optional[str] = None,
    time_use_label: str = "time-use",
    time_use_country: Optional[str] = None,
    time_use_survey: Optional[int] = None,
    time_use_weight_col: str = "propwt",
    transport_spatial_config: Optional[object] = None,
    special_days: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    ctx = _context_from_kwargs(
        synthetic_path=synthetic_path,
        observed_path=observed_path,
        observed_label=observed_label,
        synthetic_activities_path=synthetic_activities_path,
        time_use_path=time_use_path,
        time_use_label=time_use_label,
        time_use_country=time_use_country,
        time_use_survey=time_use_survey,
        time_use_weight_col=time_use_weight_col,
        transport_spatial_config=transport_spatial_config,
        special_days=special_days,
    )
    key = (*ctx.artifact_key("all"), "metrics-export")
    artifact = artifact_store.get_or_build(
        key,
        lambda: legacy._build_comparison_payload(
            synthetic_path=ctx.synthetic_path,
            observed_path=ctx.observed_path,
            observed_label=ctx.observed_label,
            synthetic_activities_path=ctx.synthetic_activities_path,
            time_use_path=ctx.time_use_path,
            time_use_label=ctx.time_use_label,
            time_use_country=ctx.time_use_country,
            time_use_survey=ctx.time_use_survey,
            time_use_weight_col=ctx.time_use_weight_col,
            transport_spatial_config=ctx.transport_spatial_config,
            special_days=ctx.special_day_dicts(),
            filter_keys=None,
            include_progressive_metadata=True,
            include_profiles=False,
            include_social_network=False,
            sections=["metrics", "time-use"],
        ),
    )
    time_use_table = []
    for group in (artifact.get("time_use_comparison") or {}).get("groups", []):
        for row in group.get("block", {}).get("rows", []):
            time_use_table.append(
                {
                    "filter_key": group["filter_key"],
                    "filter_label": group["filter_label"],
                    **row,
                }
            )
    return {
        "mode": artifact.get("mode", "synthetic_only"),
        "labels": artifact.get("labels", _labels(ctx)),
        "filters": artifact.get(
            "distribution_filters",
            [legacy._public_filter(meta) for meta in legacy._distribution_filter_options(ctx.special_day_dicts())],
        ),
        "metrics": {**_empty_metrics(), **artifact.get("metrics", {})},
        "time_use_table": time_use_table,
        "warnings": artifact.get("warnings", []),
    }


def _section_payload(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    artifact = _regular_artifact(ctx, filter_key, "metrics")
    payload["warnings"] = artifact.get("warnings", [])
    return payload


def build_section_distributions(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "distributions")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["ecdf"] = artifact.get("ecdf", {"groups": []})
    return payload


def build_section_metrics(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "metrics")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["metrics"] = {**_empty_metrics(), **artifact.get("metrics", {})}
    return payload


def build_section_activity(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "activity")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["activity"] = artifact.get("activity")
    return payload


def build_section_mobility_laws(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "mobility-laws")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["mobility_laws"] = artifact.get("mobility_laws")
    return payload


def build_section_transport_spatial(ctx: ComparisonContext, filter_key: str = "all") -> dict[str, Any]:
    artifact = _regular_artifact(ctx, "all", "transport-spatial")
    payload = _empty_payload(ctx, loaded_filters=[])
    payload["warnings"] = artifact.get("warnings", [])
    payload["transport_spatial"] = artifact.get("transport_spatial")
    return payload


def build_section_micro_activity(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "micro-activity")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["micro_activity_usage"] = artifact.get("micro_activity_usage")
    return payload


def build_section_time_use(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "time-use")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["time_use_comparison"] = artifact.get("time_use_comparison")
    return payload


def build_section_motifs(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "motifs")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["motifs"] = artifact.get("motifs")
    return payload


def build_section_stvd(ctx: ComparisonContext, filter_key: str) -> dict[str, Any]:
    artifact = _regular_artifact(ctx, filter_key, "stvd")
    payload = _empty_payload(ctx, loaded_filters=[filter_key])
    payload["warnings"] = artifact.get("warnings", [])
    payload["stvd"] = artifact.get("stvd")
    return payload


def build_section_profiles(ctx: ComparisonContext, filter_key: str = "all") -> dict[str, Any]:
    payload = _empty_payload(ctx, loaded_filters=[])
    artifact = _profile_artifact(ctx)
    payload["warnings"] = artifact.get("warnings", [])
    payload["profiles"] = artifact.get("profiles")
    return payload


def build_section_social_network(ctx: ComparisonContext, filter_key: str = "all") -> dict[str, Any]:
    payload = _empty_payload(ctx, loaded_filters=[])
    try:
        payload["social_network"] = legacy._load_social_network_sidecar(ctx.synthetic_path)
    except Exception as exc:  # noqa: BLE001 - section-level degradation
        payload["warnings"].append(f"social_network: {exc}")
    return payload


SECTION_BUILDERS = {
    "distributions": build_section_distributions,
    "metrics": build_section_metrics,
    "activity": build_section_activity,
    "transport-spatial": build_section_transport_spatial,
    "mobility-laws": build_section_mobility_laws,
    "micro-activity": build_section_micro_activity,
    "time-use": build_section_time_use,
    "motifs": build_section_motifs,
    "stvd": build_section_stvd,
    "profiles": build_section_profiles,
    "social-network": build_section_social_network,
}


def build_chart_section_payload(section: str, filter_key: str = "all", **kwargs: Any) -> dict[str, Any]:
    builder = SECTION_BUILDERS.get(section)
    if builder is None:
        raise ValueError(f"unknown chart section: {section}")
    ctx = _context_from_kwargs(**kwargs)
    return builder(ctx, filter_key)

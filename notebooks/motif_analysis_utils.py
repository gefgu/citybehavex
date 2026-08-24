"""Shared observed-panel motif analysis helpers for exploration notebooks.

These functions intentionally live next to the notebooks rather than in the
application package: they support exploratory analysis of large source panels
without adding a production-facing API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from fastmob import discover_daily_motifs_from_agents

from citybehavex.reports.comparison import _prepare_activity_visits


DAY_GROUP_ORDER = ["All days", "Weekday", "Weekend"]


def classify_observed_daily_motifs(
    path: Path,
    *,
    uid_col: str,
    timestamp_col: str,
    location_col: str | None,
    lat_col: str,
    lng_col: str,
    h3_resolution: int = 8,
) -> dict[str, Any]:
    """Prepare an observed panel and classify every classifiable user-day.

    The report's established HOME/WORK/OTHER heuristic is used because these
    source panels contain location observations but no explicit purposes.
    """
    source_columns = [uid_col, timestamp_col, lat_col, lng_col]
    if location_col is not None:
        source_columns.append(location_col)
    source = pl.read_parquet(path, columns=list(dict.fromkeys(source_columns)))
    prepared = _prepare_activity_visits(
        source,
        label=path.stem,
        uid_col=uid_col,
        datetime_col=timestamp_col,
        activity_col=None,
        location_col=location_col,
        lat_col=lat_col,
        lng_col=lng_col,
        location_resolution=h3_resolution,
    )
    if prepared is None:
        raise ValueError(f"Could not construct visits from {path}.")

    visits = prepared.visits
    input_user_days = visits.select(
        pl.col("uid"), pl.col("start_timestamp").dt.date().alias("date")
    ).unique()
    daily_motifs, _ = discover_daily_motifs_from_agents(
        visits,
        user_id_col="uid",
        location_id_col="location_id",
        purpose_col="purpose",
        timestamp_col="start_timestamp",
        end_timestamp_col="end_timestamp",
    )
    if not isinstance(daily_motifs, pl.DataFrame):
        daily_motifs = pl.from_pandas(daily_motifs)
    daily_motifs = daily_motifs.with_columns(
        pl.col("uid").cast(visits.schema["uid"]),
        pl.col("date").cast(pl.Date),
    )
    if daily_motifs.unique(subset=["uid", "date"]).height != daily_motifs.height:
        raise AssertionError("fastmob returned duplicate motifs for a user-day.")

    skipped = input_user_days.join(
        daily_motifs.select(["uid", "date"]), on=["uid", "date"], how="anti"
    )
    if daily_motifs.height + skipped.height != input_user_days.height:
        raise AssertionError("Classified and skipped user-days do not match input user-days.")

    daily = daily_motifs.with_columns(
        pl.when(pl.col("date").dt.weekday() >= 6)
        .then(pl.lit("Weekend"))
        .otherwise(pl.lit("Weekday"))
        .alias("day_group")
    )
    coverage = pd.DataFrame(
        {
            "metric": [
                "source observations",
                "people",
                "input user-days",
                "classified user-days",
                "skipped user-days",
                "first date",
                "last date",
            ],
            "value": [
                source.height,
                source.select(pl.col(uid_col).n_unique()).item(),
                input_user_days.height,
                daily.height,
                skipped.height,
                daily.select(pl.col("date").min()).item(),
                daily.select(pl.col("date").max()).item(),
            ],
        }
    )
    return {
        "daily": daily,
        "coverage": coverage,
        "skipped_user_days": skipped.sort(["uid", "date"]).head(20).to_pandas(),
        "purpose_heuristic_warning": prepared.warning,
    }


def motif_diversity_and_regularity(daily: pl.DataFrame) -> dict[str, pd.DataFrame]:
    """Return compact per-person diversity and paired regularity tables."""
    all_diversity = daily.group_by("uid").agg(
        pl.col("motif_id").n_unique().alias("distinct_motifs"),
        pl.len().alias("observed_days"),
    ).with_columns(pl.lit("All days").alias("day_group"))
    group_diversity = daily.group_by(["uid", "day_group"]).agg(
        pl.col("motif_id").n_unique().alias("distinct_motifs"),
        pl.len().alias("observed_days"),
    ).select(["uid", "distinct_motifs", "observed_days", "day_group"])
    diversity = pl.concat([all_diversity, group_diversity]).to_pandas()
    diversity["day_group"] = pd.Categorical(
        diversity["day_group"], categories=DAY_GROUP_ORDER, ordered=True
    )

    motif_counts = daily.group_by(["uid", "day_group", "motif_id"]).agg(
        pl.len().alias("motif_days")
    )
    regularity = motif_counts.group_by(["uid", "day_group"]).agg(
        pl.col("motif_days").sum().alias("observed_days"),
        pl.col("motif_days").max().alias("dominant_motif_days"),
    ).with_columns(
        (pl.col("dominant_motif_days") / pl.col("observed_days")).alias("dominant_motif_share")
    ).filter(pl.col("observed_days") >= 2)
    paired = regularity.select(["uid", "day_group", "dominant_motif_share"]).pivot(
        on="day_group", index="uid", values="dominant_motif_share"
    )
    if not {"Weekday", "Weekend"}.issubset(paired.columns):
        raise AssertionError("Both weekday and weekend regularity are required.")
    paired = paired.drop_nulls(["Weekday", "Weekend"]).to_pandas()
    if paired.empty:
        raise AssertionError("No people have two observed days in both day groups.")
    if not paired[["Weekday", "Weekend"]].apply(lambda col: col.between(0, 1)).all().all():
        raise AssertionError("Regularity shares must lie between zero and one.")
    regularity_long = paired.melt(
        id_vars="uid",
        value_vars=["Weekday", "Weekend"],
        var_name="day_group",
        value_name="dominant_motif_share",
    )
    return {
        "diversity": diversity,
        "paired_regularity": paired,
        "regularity_long": regularity_long,
    }


def weekly_motif_progression(daily: pl.DataFrame) -> pd.DataFrame:
    """Weekly medians of per-person motif diversity and regularity."""
    weekly = daily.with_columns(pl.col("date").dt.truncate("1w").alias("week_start"))
    diversity = weekly.group_by(["uid", "week_start"]).agg(
        pl.col("motif_id").n_unique().alias("distinct_motifs")
    )
    counts = weekly.group_by(["uid", "week_start", "motif_id"]).agg(
        pl.len().alias("motif_days")
    )
    regularity = counts.group_by(["uid", "week_start"]).agg(
        pl.col("motif_days").sum().alias("observed_days"),
        pl.col("motif_days").max().alias("dominant_motif_days"),
    ).filter(pl.col("observed_days") >= 2).with_columns(
        (pl.col("dominant_motif_days") / pl.col("observed_days")).alias("dominant_motif_share")
    )
    return diversity.group_by("week_start").agg(
        pl.col("distinct_motifs").median().alias("median_distinct_motifs"),
        pl.len().alias("diversity_people"),
    ).join(
        regularity.group_by("week_start").agg(
            pl.col("dominant_motif_share").median().alias("median_regularity"),
            pl.len().alias("regularity_people"),
        ),
        on="week_start",
        how="inner",
    ).sort("week_start").to_pandas()


def plot_motif_diversity(diversity: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(
        data=diversity,
        x="distinct_motifs",
        hue="day_group",
        hue_order=DAY_GROUP_ORDER,
        discrete=True,
        multiple="dodge",
        shrink=0.8,
        palette="Set2",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Number of distinct daily motifs")
    ax.set_ylabel("People")
    plt.tight_layout()
    plt.show()


def plot_weekday_weekend_regularity(
    regularity_long: pd.DataFrame, paired: pd.DataFrame, title: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.violinplot(
        data=regularity_long,
        x="day_group",
        y="dominant_motif_share",
        order=["Weekday", "Weekend"],
        hue="day_group",
        palette="Set2",
        inner="box",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title(f"{title}: within-person regularity")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Share of days matching dominant motif")
    axes[0].set_ylim(-0.03, 1.03)
    sns.scatterplot(data=paired, x="Weekday", y="Weekend", alpha=0.25, s=24, color="#4C72B0", ax=axes[1])
    axes[1].axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{title}: paired weekday vs weekend")
    axes[1].set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axes[1].set_xlabel("Weekday dominant-motif share")
    axes[1].set_ylabel("Weekend dominant-motif share")
    plt.tight_layout()
    plt.show()


def plot_weekly_progression(weekly: pd.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.lineplot(data=weekly, x="week_start", y="median_distinct_motifs", marker="o", ax=axes[0])
    axes[0].set_title(f"{title}: weekly median diversity")
    axes[0].set_xlabel("Week starting")
    axes[0].set_ylabel("Median distinct motifs per person-week")
    axes[0].tick_params(axis="x", rotation=35)
    sns.lineplot(data=weekly, x="week_start", y="median_regularity", marker="o", ax=axes[1])
    axes[1].set_title(f"{title}: weekly median regularity")
    axes[1].set_xlabel("Week starting")
    axes[1].set_ylabel("Median dominant-motif share")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.show()

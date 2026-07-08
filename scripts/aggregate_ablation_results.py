#!/usr/bin/env python
"""Aggregate ablation/comparison run manifests into paper/ LaTeX tables.

Reads data/ablation_logs/manifest.jsonl, computes mean+-std per
(dataset, variant, metric) cell, and patches specific cells in
paper/ablation.tex and paper/comparision/{spatial,temporal,semantic}_table.tex
via targeted string replacement (captions, prose, and untouched cells are
left byte-identical). Default is --dry-run (prints a diff); pass --apply to write.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# --- manifest loading -------------------------------------------------------


def load_manifest(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_report(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# --- metric extraction -------------------------------------------------------

# metric_key -> (json_path, transform)
WASSERSTEIN_METRICS = {
    "delta_r": (("wasserstein", "jump_lengths_km"), lambda v: v),
    "r_g": (("wasserstein", "radius_of_gyration_km"), lambda v: v),
    "td": (("wasserstein", "trip_duration_min"), lambda v: v),
    "dt": (("wasserstein", "dwell_time_min"), lambda v: v / 60.0),
    "vf": (("wasserstein", "visits_per_user"), lambda v: v),
}
NETWORK_METRICS = {
    "degree": ("degree",),
    "clustering": ("clustering_coefficient",),
    "edge_persistence": ("edge_persistence",),
    "topological_overlap": ("topological_overlap",),
}
JSD_METRICS = {
    "vpd": (("jsd", "activity_distribution"), lambda v: v * 100),
    "atm": (("jsd", "activity_transitions"), lambda v: v * 100),
    "dard": (("jsd", "daily_activity_profile"), lambda v: v * 100),
}


def _get_nested(d: dict, path: tuple[str, ...]) -> Any:
    node = d
    for key in path:
        if node is None or key not in node:
            return None
        node = node[key]
    return node


def extract_metric(report: dict, metric_key: str) -> float | None:
    if metric_key in WASSERSTEIN_METRICS:
        path, transform = WASSERSTEIN_METRICS[metric_key]
        v = _get_nested(report, path)
        return transform(v) if v is not None else None
    if metric_key in JSD_METRICS:
        path, transform = JSD_METRICS[metric_key]
        v = _get_nested(report, path)
        return transform(v) if v is not None else None
    if metric_key in NETWORK_METRICS:
        (sub,) = NETWORK_METRICS[metric_key]
        v = _get_nested(
            report, ("network_validation", "synthetic_vs_random", "wasserstein", sub)
        )
        return v
    raise ValueError(f"unknown metric key {metric_key!r}")


def aggregate(
    manifest_rows: list[dict[str, Any]], metric_keys: Iterable[str]
) -> dict[tuple[str, str], dict[str, tuple[float, float, int]]]:
    """Returns {(dataset, variant): {metric_key: (mean, std, n)}}."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in manifest_rows:
        report = load_report(row["report_json_path"])
        if report is None:
            print(f"WARNING: missing report {row['report_json_path']}, skipping")
            continue
        key = (row["dataset"], row["variant"])
        for metric_key in metric_keys:
            value = extract_metric(report, metric_key)
            if value is not None:
                grouped[key][metric_key].append(value)
        for extra_key in ("rt_minutes", "mem_gb"):
            value = row.get(extra_key)
            if value is not None:
                grouped[key][extra_key].append(value)

    out: dict[tuple[str, str], dict[str, tuple[float, float, int]]] = {}
    for key, metrics in grouped.items():
        out[key] = {}
        for metric_key, values in metrics.items():
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values) if n >= 2 else 0.0
            out[key][metric_key] = (mean, std, n)
    return out


# --- LaTeX cell formatting ---------------------------------------------------


def format_cell(mean: float, std: float, n: int, decimals: int, bold: bool) -> str:
    if n <= 1:
        body = f"{mean:.{decimals}f}"
    else:
        body = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    return f"${body}$"


_MATH_SPAN = re.compile(r"\$.*\$")


def replace_cell_value(cell: str, new_value: str) -> str:
    """Replace only the `$...$` math span in a table cell, preserving all
    surrounding text (leading/trailing whitespace, a trailing `\\\\` row
    terminator on the last cell of a row, etc.)."""
    if _MATH_SPAN.search(cell):
        return _MATH_SPAN.sub(lambda _m: new_value, cell, count=1)
    # No existing math span (e.g. a bare "-" placeholder) -- rebuild the cell,
    # preserving only a trailing row terminator if present.
    trailing = re.search(r"\\\\\s*$", cell)
    suffix = f" {trailing.group(0)}" if trailing else ""
    return f" {new_value}{suffix}"


# --- ablation.tex patching ---------------------------------------------------

ABLATION_VARIANT_COLUMNS = [
    "full",
    "no_profile",
    "no_micro_sched",
    "no_social",
    "no_transport",
    "no_feedback",
]
# 1-indexed position of each variant's value cell within the `&`-split row
# (0=dataset, 1=metric, 2=full, 3=no_profile, ... 7=no_feedback, 8=ref)
ABLATION_COLUMN_INDEX = {v: i + 2 for i, v in enumerate(ABLATION_VARIANT_COLUMNS)}

ABLATION_METRIC_LABELS = {
    "delta_r": r"\$\\Delta r\$",
    "r_g": r"\$r_g\$",
    "td": r"TD\.",
    "dt": r"DT\.",
    "vf": r"Vf\.",
    "rt": r"RT\.",
    "mem": r"Mem\.",
}
NEW_ROW_METRICS = [
    ("degree", "Degree"),
    ("clustering", "Clustering coeff."),
    ("edge_persistence", "Edge persistence"),
    ("topological_overlap", "Topological overlap"),
]

ABLATION_DATASET_LABELS = {
    "gparis": r"\parisData{}",
    "shanghai": "Shanghai",
    "yjmob": "YJMOB",
    "yjmob2": "YJMOB disaster",
}
# datasets whose new network-validation rows get real numbers (not "-")
NETWORK_ROW_DATASETS = {"shanghai", "yjmob", "yjmob2"}


def patch_ablation_tex(
    text: str,
    results: dict[tuple[str, str], dict[str, tuple[float, float, int]]],
    decimals: int = 1,
) -> tuple[str, list[str]]:
    lines = text.split("\n")
    changes: list[str] = []

    # Determine, for each line, which dataset block it belongs to.
    current_dataset: str | None = None
    dataset_by_line: dict[int, str] = {}
    for i, line in enumerate(lines):
        for key, label in ABLATION_DATASET_LABELS.items():
            plain_label = label.replace("\\", "").replace("{}", "")
            if line.strip().startswith(label) or line.strip().startswith(plain_label):
                current_dataset = key
                break
        if current_dataset:
            dataset_by_line[i] = current_dataset

    for i, line in enumerate(lines):
        if r"\textit{" not in line or "&" not in line:
            continue
        dataset = dataset_by_line.get(i)
        if dataset is None:
            continue
        for metric_key, label_pattern in ABLATION_METRIC_LABELS.items():
            if not re.search(label_pattern, line):
                continue
            cells = line.split("&")
            for variant, col_idx in ABLATION_COLUMN_INDEX.items():
                if col_idx >= len(cells):
                    continue
                cell_data = results.get((dataset, variant), {}).get(metric_key)
                if cell_data is None:
                    continue
                mean, std, n = cell_data
                new_cell = format_cell(mean, std, n, decimals, bold=(variant == "full"))
                cells[col_idx] = replace_cell_value(cells[col_idx], new_cell)
                changes.append(
                    f"{dataset}/{variant}/{metric_key}: n={n} mean={mean:.3f} std={std:.3f}"
                )
            lines[i] = "&".join(cells)
            break

    return "\n".join(lines), changes


def insert_network_rows(text: str) -> str:
    """Insert 4 new metric rows after each dataset block's Mem. row."""
    lines = text.split("\n")
    out_lines: list[str] = []
    current_dataset: str | None = None
    for line in lines:
        for key, label in ABLATION_DATASET_LABELS.items():
            plain_label = label.replace("\\", "").replace("{}", "")
            if line.strip().startswith(label) or line.strip().startswith(plain_label):
                current_dataset = key
                break
        out_lines.append(line)
        if current_dataset and re.search(r"\\textit\{Mem\.", line):
            indent = re.match(r"\s*", line).group(0)
            cells_placeholder = " & ".join(["$-$"] * (len(ABLATION_VARIANT_COLUMNS) + 1))
            for _metric_key, row_label in NEW_ROW_METRICS:
                new_line = f"{indent}& \\textit{{{row_label}}} & {cells_placeholder} \\\\"
                out_lines.append(new_line)
    return "\n".join(out_lines)


def has_network_rows(text: str) -> bool:
    return any(label in text for _key, label in NEW_ROW_METRICS)


def patch_network_rows(
    text: str,
    results: dict[tuple[str, str], dict[str, tuple[float, float, int]]],
    decimals: int = 2,
) -> tuple[str, list[str]]:
    lines = text.split("\n")
    changes: list[str] = []
    current_dataset: str | None = None
    dataset_by_line: dict[int, str] = {}
    for i, line in enumerate(lines):
        for key, label in ABLATION_DATASET_LABELS.items():
            plain_label = label.replace("\\", "").replace("{}", "")
            if line.strip().startswith(label) or line.strip().startswith(plain_label):
                current_dataset = key
                break
        if current_dataset:
            dataset_by_line[i] = current_dataset

    for i, line in enumerate(lines):
        if r"\textit{" not in line or "&" not in line:
            continue
        dataset = dataset_by_line.get(i)
        if dataset is None or dataset not in NETWORK_ROW_DATASETS:
            continue
        for metric_key, row_label in NEW_ROW_METRICS:
            if row_label not in line:
                continue
            cells = line.split("&")
            for variant, col_idx in ABLATION_COLUMN_INDEX.items():
                if col_idx >= len(cells):
                    continue
                cell_data = results.get((dataset, variant), {}).get(metric_key)
                if cell_data is None:
                    continue
                mean, std, n = cell_data
                new_cell = format_cell(mean, std, n, decimals, bold=(variant == "full"))
                cells[col_idx] = replace_cell_value(cells[col_idx], new_cell)
                changes.append(
                    f"{dataset}/{variant}/{metric_key}: n={n} mean={mean:.3f} std={std:.3f}"
                )
            lines[i] = "&".join(cells)
            break

    return "\n".join(lines), changes


# --- comparison tables (spatial/temporal/semantic) --------------------------

COMPARISON_DATASET_LABELS = {
    "gparis": [r"\GP{}", r"\parisDataShort{}"],
    "shanghai": [r"\SH{}", "Shanghai"],
}
COMPARISON_SOURCE_LABEL = r"\cbX{}"
COMPARISON_SOURCE_LABEL_ALT = r"\simname"


def patch_comparison_row(
    text: str,
    dataset: str,
    metrics_in_order: list[tuple[str, int]],
    values: dict[str, tuple[float, float, int]],
) -> tuple[str, list[str]]:
    """metrics_in_order: list of (metric_key, decimals) in the row's column order."""
    lines = text.split("\n")
    changes: list[str] = []
    labels = COMPARISON_DATASET_LABELS.get(dataset, [])
    for i, line in enumerate(lines):
        if "&" not in line:
            continue
        if not any(lbl in line for lbl in labels):
            continue
        if COMPARISON_SOURCE_LABEL not in line and COMPARISON_SOURCE_LABEL_ALT not in line:
            continue
        cells = line.split("&")
        # cells[0]=dataset, cells[1]=source, cells[2:]=metric values
        for offset, (metric_key, decimals) in enumerate(metrics_in_order):
            col_idx = 2 + offset
            if col_idx >= len(cells) or metric_key not in values:
                continue
            mean, std, n = values[metric_key]
            new_cell = format_cell(mean, std, n, decimals, bold=True)
            cells[col_idx] = replace_cell_value(cells[col_idx], new_cell)
            changes.append(f"{dataset}/{metric_key}: n={n} mean={mean:.3f} std={std:.3f}")
        lines[i] = "&".join(cells)
    return "\n".join(lines), changes


# --- CLI ----------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ablation_logs/manifest.jsonl")
    parser.add_argument("--ablation-tex", default="paper/ablation.tex")
    parser.add_argument("--spatial-tex", default="paper/comparision/spatial_table.tex")
    parser.add_argument("--temporal-tex", default="paper/comparision/temporal_table.tex")
    parser.add_argument("--semantic-tex", default="paper/comparision/semantic_table.tex")
    parser.add_argument("--targets", nargs="+", choices=["ablation", "comparison"], default=["ablation", "comparison"])
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_rows = load_manifest(args.manifest)

    all_metric_keys = (
        list(WASSERSTEIN_METRICS) + list(NETWORK_METRICS) + list(JSD_METRICS)
    )
    results = aggregate(manifest_rows, all_metric_keys)

    if "ablation" in args.targets:
        path = Path(args.ablation_tex)
        text = path.read_text()
        if not has_network_rows(text):
            text_with_rows = insert_network_rows(text)
        else:
            text_with_rows = text
        patched, changes = patch_ablation_tex(text_with_rows, results)
        patched, net_changes = patch_network_rows(patched, results)
        changes += net_changes
        print(f"=== {path} ({len(changes)} cells) ===")
        for c in changes:
            print(" ", c)
        if args.apply:
            path.write_text(patched)
            print(f"wrote {path}")
        else:
            print("(dry-run, not written; pass --apply to write)")

    if "comparison" in args.targets:
        # gparis full-model rows already available from the 3 session runs
        gparis_full = results.get(("gparis", "full"), {})
        shanghai_full = results.get(("shanghai", "500sample"), {}) or results.get(
            ("shanghai", "full"), {}
        )

        for tex_path, metrics_in_order, per_dataset in [
            (
                Path(args.spatial_tex),
                [("delta_r", 2), ("r_g", 2)],
                {"gparis": gparis_full, "shanghai": shanghai_full},
            ),
            (
                Path(args.temporal_tex),
                [("td", 2), ("dt", 2), ("vf", 2)],
                {"gparis": gparis_full, "shanghai": shanghai_full},
            ),
            (
                Path(args.semantic_tex),
                [("vpd", 2), ("atm", 2), ("dard", 2)],
                {"gparis": gparis_full, "shanghai": shanghai_full},
            ),
        ]:
            text = tex_path.read_text()
            all_changes: list[str] = []
            for dataset, values in per_dataset.items():
                if not values:
                    continue
                text, changes = patch_comparison_row(text, dataset, metrics_in_order, values)
                all_changes += changes
            print(f"=== {tex_path} ({len(all_changes)} cells) ===")
            for c in all_changes:
                print(" ", c)
            if args.apply:
                tex_path.write_text(text)
                print(f"wrote {tex_path}")
            else:
                print("(dry-run, not written; pass --apply to write)")


if __name__ == "__main__":
    main()

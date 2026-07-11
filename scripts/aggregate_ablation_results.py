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
import math
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


def _finite_or_none(v: float | None) -> float | None:
    """NaN/inf slip through as valid JSON floats but would silently poison a
    mean/std (or print literal "nan" into the LaTeX table) -- treat them the
    same as a missing value instead."""
    if v is None:
        return None
    try:
        return v if math.isfinite(v) else None
    except TypeError:
        return v


def extract_metric(report: dict, metric_key: str) -> float | None:
    if metric_key in WASSERSTEIN_METRICS:
        path, transform = WASSERSTEIN_METRICS[metric_key]
        v = _get_nested(report, path)
        return _finite_or_none(transform(v) if v is not None else None)
    if metric_key in JSD_METRICS:
        path, transform = JSD_METRICS[metric_key]
        v = _get_nested(report, path)
        return _finite_or_none(transform(v) if v is not None else None)
    if metric_key in NETWORK_METRICS:
        (sub,) = NETWORK_METRICS[metric_key]
        # Model columns (full, no_profile, ...) carry synthetic_vs_random;
        # the Ref. column's real-vs-real half-A-vs-half-B comparison
        # (built by build_observed_pair_network_validation) carries
        # observed_vs_observed instead -- mutually exclusive per report,
        # so trying both is safe.
        v = _get_nested(
            report, ("network_validation", "synthetic_vs_random", "wasserstein", sub)
        )
        if v is None:
            v = _get_nested(
                report, ("network_validation", "observed_vs_observed", "wasserstein", sub)
            )
        return _finite_or_none(v)
    raise ValueError(f"unknown metric key {metric_key!r}")


def aggregate(
    manifest_rows: list[dict[str, Any]], metric_keys: Iterable[str]
) -> dict[tuple[str, str], dict[str, tuple[float, float, int]]]:
    """Returns {(dataset, variant): {metric_key: (mean, std, n)}}."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in manifest_rows:
        if row.get("report_json_path") is None:
            continue
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


def format_cell(
    mean: float, std: float, n: int, decimals: int, bold: bool, flagged: bool = False
) -> str:
    if n <= 1:
        body = f"{mean:.{decimals}f}"
    else:
        body = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    if flagged:
        # Value looks real but the underlying metric is under suspicion
        # (e.g. a known data-adaptation issue not yet root-caused) -- mark
        # for manual review rather than silently presenting it as trusted.
        body = f"\\gustavo{{{body}}}"
    return f"${body}$"


# (dataset, metric_key) pairs whose computed values are suspect and should
# be flagged with \gustavo{} in the table rather than presented as trusted.
# See EVALUATION_NOTES.md for why each entry is here.
NEEDS_CHECKING: set[tuple[str, str]] = {
    ("yjmob2", "vf"),  # visits_per_user implausibly high (310-371); likely
                       # the same raw-row-vs-collapsed-stays mismatch fixed
                       # for Shanghai, not yet applying cleanly for yjmob2.
    ("yjmob", "vf"),   # same anomaly, plain yjmob (407-496 across many
                       # independent runs) -- not yjmob2-specific.
}


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
REF_COLUMN_INDEX = 8

ABLATION_METRIC_LABELS = {
    "delta_r": r"\$\\Delta r\$",
    "r_g": r"\$r_g\$",
    "td": r"TD\.",
    "dt": r"DT\.",
    "vf": r"Vf\.",
    "rt_minutes": r"RT\.",
    "mem_gb": r"Mem\.",
    "vpd": r"\\textit\{VPD\}",
    "atm": r"\\textit\{ATM\}",
    "dard": r"\\textit\{DARD\}",
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

# Sorted longest-label-first so a prefix collision (e.g. "YJMOB" is a prefix
# of "YJMOB disaster") always resolves to the more specific match.
_SORTED_ABLATION_DATASET_LABELS = sorted(
    ABLATION_DATASET_LABELS.items(), key=lambda kv: len(kv[1]), reverse=True
)


def _detect_dataset(line: str) -> str | None:
    stripped = line.strip()
    for key, label in _SORTED_ABLATION_DATASET_LABELS:
        plain_label = label.replace("\\", "").replace("{}", "")
        if stripped.startswith(label) or stripped.startswith(plain_label):
            return key
    return None


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
        detected = _detect_dataset(line)
        if detected is not None:
            current_dataset = detected
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
            # Bold only the actual best (lowest, since every metric here is
            # a distance/divergence/runtime -- lower means closer to real or
            # cheaper) value in the row, not unconditionally "full".
            row_values = {
                variant: results.get((dataset, variant), {}).get(metric_key)
                for variant, col_idx in ABLATION_COLUMN_INDEX.items()
                if col_idx < len(cells)
            }
            finite = {v: data[0] for v, data in row_values.items() if data is not None}
            best_variant = min(finite, key=finite.get) if finite else None
            for variant, col_idx in ABLATION_COLUMN_INDEX.items():
                if col_idx >= len(cells):
                    continue
                cell_data = row_values.get(variant)
                if cell_data is None:
                    continue
                mean, std, n = cell_data
                new_cell = format_cell(
                    mean, std, n, decimals, bold=(variant == best_variant),
                    flagged=(dataset, metric_key) in NEEDS_CHECKING,
                )
                cells[col_idx] = replace_cell_value(cells[col_idx], new_cell)
                changes.append(
                    f"{dataset}/{variant}/{metric_key}: n={n} mean={mean:.3f} std={std:.3f}"
                )
            # Ref. column (index 8): real-vs-real half-A-vs-half-B comparison,
            # a single point estimate (n=1); not applicable to RT./Mem. since
            # no simulation runs for it.
            if metric_key not in ("rt_minutes", "mem_gb") and REF_COLUMN_INDEX < len(cells):
                ref_data = results.get((dataset, "ref"), {}).get(metric_key)
                if ref_data is not None:
                    ref_mean, ref_std, ref_n = ref_data
                    ref_cell = format_cell(ref_mean, ref_std, ref_n, decimals, bold=False, flagged=False)
                    cells[REF_COLUMN_INDEX] = replace_cell_value(cells[REF_COLUMN_INDEX], ref_cell)
                    changes.append(f"{dataset}/ref/{metric_key}: n={ref_n} mean={ref_mean:.3f}")
            lines[i] = "&".join(cells)
            break

    return "\n".join(lines), changes


def place_network_rows_at_block_end(text: str) -> str:
    """Ensure each dataset block has exactly the 4 network-metric rows
    (Degree, Clustering coeff., Edge persistence, Topological overlap) as
    its LAST rows, before the block's closing \\midrule (or end of table
    for the last block). Idempotent: if the rows don't exist yet, inserts
    them (fresh placeholders); if they exist but are positioned elsewhere
    (e.g. the original layout right after Mem.), moves them to the end
    without touching their already-patched values.
    """
    lines = text.split("\n")
    out_lines: list[str] = []
    current_dataset: str | None = None
    pending_network_lines: list[str] = []
    block_indent: str | None = None

    def flush_pending() -> None:
        nonlocal pending_network_lines, block_indent
        if not pending_network_lines and block_indent is None:
            return
        if not pending_network_lines and block_indent is not None:
            cells_placeholder = " & ".join(["$-$"] * (len(ABLATION_VARIANT_COLUMNS) + 1))
            pending_network_lines = [
                f"{block_indent}& \\textit{{{row_label}}} & {cells_placeholder} \\\\"
                for _metric_key, row_label in NEW_ROW_METRICS
            ]
        out_lines.extend(pending_network_lines)
        pending_network_lines = []
        block_indent = None

    for line in lines:
        detected = _detect_dataset(line)
        if detected is not None:
            if detected != current_dataset:
                flush_pending()
            current_dataset = detected

        if current_dataset and any(label in line for _key, label in NEW_ROW_METRICS):
            if block_indent is None:
                block_indent = re.match(r"\s*", line).group(0)
            pending_network_lines.append(line)
            continue

        if current_dataset and re.search(r"\\textit\{Mem\.", line) and block_indent is None:
            block_indent = re.match(r"\s*", line).group(0)

        if current_dataset and (line.strip() == r"\midrule" or r"\end{tabular}" in line):
            flush_pending()
            current_dataset = None

        out_lines.append(line)

    flush_pending()
    return "\n".join(out_lines)


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
        detected = _detect_dataset(line)
        if detected is not None:
            current_dataset = detected
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
            # Bold only the actual best (lowest) value in the row, not
            # unconditionally "full". no_social is excluded from the
            # comparison (forced dash below), so it can never win it.
            row_values = {
                variant: results.get((dataset, variant), {}).get(metric_key)
                for variant, col_idx in ABLATION_COLUMN_INDEX.items()
                if col_idx < len(cells) and variant != "no_social"
            }
            finite = {v: data[0] for v, data in row_values.items() if data is not None}
            best_variant = min(finite, key=finite.get) if finite else None
            for variant, col_idx in ABLATION_COLUMN_INDEX.items():
                if col_idx >= len(cells):
                    continue
                # -Social removes the module that co-presence/network
                # metrics are meant to characterize -- a graph can still be
                # built from raw location/time co-occurrence, but the
                # numbers wouldn't mean what they mean everywhere else in
                # this row, so this column is dash-only by design.
                if variant == "no_social":
                    cells[col_idx] = replace_cell_value(cells[col_idx], "$-$")
                    continue
                cell_data = row_values.get(variant)
                if cell_data is None:
                    continue
                mean, std, n = cell_data
                new_cell = format_cell(
                    mean, std, n, decimals, bold=(variant == best_variant),
                    flagged=(dataset, metric_key) in NEEDS_CHECKING,
                )
                cells[col_idx] = replace_cell_value(cells[col_idx], new_cell)
                changes.append(
                    f"{dataset}/{variant}/{metric_key}: n={n} mean={mean:.3f} std={std:.3f}"
                )
            # Ref. column: real-vs-real half-A-vs-half-B network comparison
            # (build_observed_pair_network_validation), single point estimate.
            if REF_COLUMN_INDEX < len(cells):
                ref_data = results.get((dataset, "ref"), {}).get(metric_key)
                if ref_data is not None:
                    ref_mean, ref_std, ref_n = ref_data
                    ref_cell = format_cell(ref_mean, ref_std, ref_n, decimals, bold=False, flagged=False)
                    cells[REF_COLUMN_INDEX] = replace_cell_value(cells[REF_COLUMN_INDEX], ref_cell)
                    changes.append(f"{dataset}/ref/{metric_key}: n={ref_n} mean={ref_mean:.3f}")
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
            new_cell = format_cell(
                mean, std, n, decimals, bold=True,
                flagged=(dataset, metric_key) in NEEDS_CHECKING,
            )
            cells[col_idx] = replace_cell_value(cells[col_idx], new_cell)
            changes.append(f"{dataset}/{metric_key}: n={n} mean={mean:.3f} std={std:.3f}")
        lines[i] = "&".join(cells)
    return "\n".join(lines), changes


def _parse_cell_mean(cell: str) -> float | None:
    m = _MATH_SPAN.search(cell)
    if not m:
        return None
    inner = re.sub(r"\\mathbf\{(.*)\}", r"\1", m.group(0)[1:-1])
    num_m = re.search(r"[-\d.]+", inner)
    return float(num_m.group(0)) if num_m else None


def _set_cell_bold(cell: str, bold: bool) -> str:
    m = _MATH_SPAN.search(cell)
    if not m:
        return cell
    inner = re.sub(r"\\mathbf\{(.*)\}", r"\1", m.group(0)[1:-1])
    new_math = f"$\\mathbf{{{inner}}}$" if bold else f"${inner}$"
    return replace_cell_value(cell, new_math)


def rebold_comparison_table(text: str, datasets: list[str], n_cols: int = 8) -> tuple[str, list[str]]:
    """AgentSociety/CitySim rows are static baseline text (never computed by
    this script) and CBX's row is unconditionally bold from
    patch_comparison_row -- neither reflects who's actually best. Re-derive
    bolding per column across all 3 (or however many) source rows in each
    dataset block, bolding only the true minimum (every metric here is a
    W1 distance or JSD -- lower is closer to real, always better)."""
    lines = text.split("\n")
    changes: list[str] = []
    for dataset in datasets:
        labels = COMPARISON_DATASET_LABELS.get(dataset, [])
        row_idx = [
            i for i, l in enumerate(lines)
            if "&" in l and any(lbl in l for lbl in labels) and "Ref." not in l
        ]
        if len(row_idx) < 2:
            continue
        split_rows = [lines[i].split("&") for i in row_idx]
        for col in range(n_cols):
            cell_idx = 2 + col
            vals = [
                (r, _parse_cell_mean(row[cell_idx])) if cell_idx < len(row) else (r, None)
                for r, row in enumerate(split_rows)
            ]
            finite = [(r, v) for r, v in vals if v is not None]
            if not finite:
                continue
            best_r = min(finite, key=lambda rv: rv[1])[0]
            for r, row in enumerate(split_rows):
                if cell_idx >= len(row):
                    continue
                row[cell_idx] = _set_cell_bold(row[cell_idx], r == best_r)
        for i, row in zip(row_idx, split_rows):
            lines[i] = "&".join(row)
        changes.append(f"{dataset}: rebolded across {len(row_idx)} source rows")
    return "\n".join(lines), changes


# --- CLI ----------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ablation_logs/manifest.jsonl")
    parser.add_argument("--ablation-tex", default="paper/ablation.tex")
    parser.add_argument("--comparison-tex", default="paper/comparision_table.tex")
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
        text_with_rows = place_network_rows_at_block_end(text)
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
        # 500-agent comparison-table baseline (AgentSociety/CitySim methodology).
        # Distinct from the ablation table's "full" variant, which for
        # shanghai/yjmob/yjmob2/gparis is the N/2 population-halved baseline.
        gparis_full = results.get(("gparis", "500sample"), {}) or results.get(
            ("gparis", "full"), {}
        )
        shanghai_full = results.get(("shanghai", "500sample"), {}) or results.get(
            ("shanghai", "full"), {}
        )
        # Single merged table: Dataset | Source | dr | rg | TD | DT | Vf | VPD | ATM | DARD
        metrics_in_order = [
            ("delta_r", 2), ("r_g", 2), ("td", 2), ("dt", 2), ("vf", 2),
            ("vpd", 2), ("atm", 2), ("dard", 2),
        ]
        per_dataset = {"gparis": gparis_full, "shanghai": shanghai_full}

        tex_path = Path(args.comparison_tex)
        text = tex_path.read_text()
        all_changes: list[str] = []
        for dataset, values in per_dataset.items():
            if not values:
                continue
            text, changes = patch_comparison_row(text, dataset, metrics_in_order, values)
            all_changes += changes
        text, rebold_changes = rebold_comparison_table(text, list(per_dataset), n_cols=len(metrics_in_order))
        all_changes += rebold_changes
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

import type { EChartsOption } from "echarts";
import type {
  ActivityBlock,
  BarSeries,
  EcdfBlock,
  LawBlock,
  MicroActivityUsageBlock,
  MotifsBlock,
  ProfilesBlock,
  SeriesPoints,
  TimeUseComparisonBlock,
  TransportSpatialBlock,
} from "../api";
import {
  axisCommon,
  baseOption,
  COLORS,
  PROFILE_COLOR,
  ROLE_COLOR,
  ROLE_DASH,
} from "./theme";

function lineOrScatter(s: SeriesPoints) {
  const color = ROLE_COLOR[s.role] ?? COLORS.ink;
  if (s.type === "scatter") {
    return {
      name: s.name,
      type: "scatter" as const,
      data: s.points,
      symbolSize: 7,
      itemStyle: { color: "transparent", borderColor: color, borderWidth: 1.5 },
      z: 3,
    };
  }
  return {
    name: s.name,
    type: "line" as const,
    data: s.points,
    showSymbol: false,
    lineStyle: { color, width: 2, type: ROLE_DASH[s.role] },
    itemStyle: { color },
  };
}

export function ecdfOption(block: EcdfBlock): EChartsOption {
  const unit = block.x_unit ? ` · ${block.x_unit}` : "";
  return {
    ...baseOption(),
    xAxis: axisCommon(block.x_label + unit),
    yAxis: { ...axisCommon("cdf"), min: 0, max: 1 },
    series: block.series.map((s) => ({
      name: s.name,
      type: "line",
      data: s.points,
      showSymbol: false,
      lineStyle: { color: ROLE_COLOR[s.role] ?? COLORS.ink, width: 2, type: ROLE_DASH[s.role] },
      itemStyle: { color: ROLE_COLOR[s.role] ?? COLORS.ink },
    })),
  };
}

export function lawOption(block: LawBlock): EChartsOption {
  const xUnit = block.x_unit ? ` · ${block.x_unit}` : "";
  return {
    ...baseOption(),
    tooltip: { ...baseOption().tooltip, trigger: "item" },
    xAxis: axisCommon(block.x_label + xUnit, block.x_log),
    yAxis: axisCommon("P", true),
    series: block.series.map(lineOrScatter),
  };
}

function groupedBar(categories: string[], series: BarSeries[], yName: string): EChartsOption {
  return {
    ...baseOption(),
    tooltip: { ...baseOption().tooltip, trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "category",
      data: categories,
      axisLabel: { color: COLORS.muted, interval: 0, rotate: categories.some((c) => c.length > 4) ? 40 : 0 },
      axisLine: { lineStyle: { color: COLORS.hairline } },
    },
    yAxis: { ...axisCommon(yName) },
    series: series.map((s) => ({
      name: s.name,
      type: "bar",
      data: s.values,
      barMaxWidth: 36,
      itemStyle: { color: ROLE_COLOR[s.role] ?? COLORS.ink },
    })),
  };
}

export function purposeOption(block: ActivityBlock["purpose"]): EChartsOption {
  return groupedBar(block.categories, block.series, "% of visits");
}

export function transportShareOption(block: TransportSpatialBlock["share"]): EChartsOption {
  return groupedBar(block.categories, block.series, "% of trips");
}

export function motifOption(block: MotifsBlock): EChartsOption {
  const labelKeys = block.motif_label_keys ?? {};
  const rich = (block.motif_label_styles ?? {}) as Record<string, never>;
  return {
    ...baseOption(),
    grid: { left: 64, right: 24, top: 48, bottom: Object.keys(rich).length ? 150 : 90, containLabel: false },
    tooltip: {
      ...baseOption().tooltip,
      trigger: "item",
      formatter: (params: unknown) => {
        const p = params as { seriesName: string; value?: unknown[] };
        const value = p.value ?? [];
        if (value.length >= 6) {
          return [
            `Literature motif: ${value[2]}`,
            `Packed motif ID: ${value[3]}`,
            `Hex ID: ${value[4]}`,
            `${p.seriesName}: ${Number(value[1]).toFixed(2)}%`,
            `Count: ${value[5]}`,
          ].join("<br/>");
        }
        return `${p.seriesName}: ${Number(value[1] ?? 0).toFixed(2)}%`;
      },
    },
    xAxis: {
      type: "category",
      data: block.categories,
      name: "MOTIF ID",
      nameLocation: "middle",
      nameGap: Object.keys(rich).length ? 124 : 62,
      axisLabel: {
        color: COLORS.muted,
        interval: 0,
        margin: 10,
        rotate: Object.keys(rich).length ? 0 : 60,
        fontSize: Object.keys(rich).length ? 22 : 9,
        fontFamily: "ui-monospace, monospace",
        rich,
        formatter: (value: string) => {
          const styleKey = labelKeys[value];
          return styleKey ? `{${styleKey}| }` : value;
        },
      },
      axisLine: { lineStyle: { color: COLORS.hairline } },
      axisTick: { show: false },
    },
    yAxis: { ...axisCommon("% of user-days"), min: 0, max: 100 },
    series: block.series.map((s) => ({
      name: s.name,
      type: "bar",
      data: block.categories.map((hexId, i) => {
        const row = s.rows?.find((r) => r.hex_id === hexId);
        if (!row) return [hexId, s.values[i] ?? 0, "", "", hexId, ""];
        return [
          row.hex_id,
          Number(row.percentage),
          row.literature_motif_id,
          row.motif_id,
          row.hex_id,
          row.count,
        ];
      }),
      dimensions: [
        "hex_id",
        "percentage",
        "literature_motif_id",
        "packed_motif_id",
        "hex_label",
        "count",
      ],
      encode: { x: "hex_id", y: "percentage" },
      barMaxWidth: 32,
      itemStyle: { color: ROLE_COLOR[s.role] ?? COLORS.ink },
    })),
  };
}

function diffHeatmap(
  xLabels: string[],
  yLabels: string[],
  matrix: number[][],
  limit: number,
  labels: string[],
  xName: string,
  matrixMode: "difference" | "raw" = "difference",
): EChartsOption {
  // matrix[row=y][col=x] -> data [colIndex, rowIndex, value]
  const data: [number, number, number][] = [];
  matrix.forEach((row, y) => row.forEach((v, x) => data.push([x, y, v])));
  return {
    ...baseOption(),
    grid: { left: 90, right: 80, top: 40, bottom: 70, containLabel: false },
    tooltip: {
      ...baseOption().tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const d = p as { data: [number, number, number] };
        const suffix = matrixMode === "difference" ? " pp" : "";
        return `${yLabels[d.data[1]]} → ${xLabels[d.data[0]]}<br/>${d.data[2].toFixed(2)}${suffix}`;
      },
    },
    xAxis: {
      type: "category",
      data: xLabels,
      name: xName.toUpperCase(),
      nameLocation: "middle",
      nameGap: 46,
      axisLabel: { color: COLORS.muted, interval: 0, rotate: 40 },
      splitArea: { show: true },
    },
    yAxis: {
      type: "category",
      data: yLabels,
      axisLabel: { color: COLORS.muted, interval: 0 },
      splitArea: { show: true },
    },
    visualMap: {
      min: -limit,
      max: limit,
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "center",
      text: matrixMode === "difference" ? [`${labels[1]} higher`, `${labels[0]} higher`] : ["high", "low"],
      textStyle: { color: COLORS.muted, fontSize: 10 },
      inRange: { color: [COLORS.coral, "#f7f7f7", COLORS.forest] },
    },
    series: [
      {
        name: matrixMode === "difference" ? `${labels[1]} - ${labels[0]}` : labels[0],
        type: "heatmap",
        data,
        label: { show: xLabels.length <= 8, fontSize: 10, formatter: (p: unknown) => (p as { data: number[] }).data[2].toFixed(0) },
      },
    ],
  };
}

export function transitionOption(block: ActivityBlock["transition_difference"]): EChartsOption {
  return diffHeatmap(block.categories, block.categories, block.matrix, block.limit, block.labels, "to activity", block.matrix_mode);
}

export function dailyActivityOption(
  block: NonNullable<ActivityBlock["daily_activity_difference"]>,
): EChartsOption {
  const perHour = Math.max(1, Math.round(block.n_bins / 24));
  const xLabels = Array.from({ length: block.n_bins }, (_, i) => {
    const h = Math.floor(i / perHour);
    return i % perHour === 0 ? `${String(h).padStart(2, "0")}:00` : "";
  });
  const opt = diffHeatmap(xLabels, block.categories, block.matrix, block.limit, block.labels, "time of day", block.matrix_mode);
  (opt.series as { label: { show: boolean } }[])[0].label.show = false;
  return opt;
}

export function microActivityUsageOption(block: MicroActivityUsageBlock): EChartsOption {
  const palette = [
    COLORS.coral,
    COLORS.forest,
    COLORS.mustard,
    COLORS.info,
    COLORS.peach,
    COLORS.mint,
    COLORS.yellow,
    "#6b5b95",
    "#008c95",
    "#9a6324",
    "#7b3f61",
    "#2f4f4f",
    "#bc5090",
    "#58508d",
    "#006d2c",
  ];
  return {
    ...baseOption(),
    tooltip: {
      ...baseOption().tooltip,
      trigger: "axis",
      valueFormatter: (value: unknown) => `${Number(value).toFixed(2)}%`,
    },
    legend: { ...baseOption().legend, type: "scroll", top: 8 },
    xAxis: {
      type: "category",
      data: block.x,
      name: "TIME OF DAY",
      nameLocation: "middle",
      nameGap: 40,
      axisLabel: {
        color: COLORS.muted,
        fontSize: 11,
        interval: Math.max(0, Math.round(block.n_bins / 12) - 1),
      },
      axisLine: { lineStyle: { color: COLORS.hairline } },
    },
    yAxis: { ...axisCommon("% of micro-activity time"), min: 0 },
    series: block.series.map((s, i) => ({
      name: s.name,
      type: "line",
      data: s.values,
      stack: "micro-activity usage",
      showSymbol: false,
      areaStyle: { opacity: 0.72 },
      emphasis: { focus: "series" },
      lineStyle: { color: palette[i % palette.length], width: 2 },
      itemStyle: { color: palette[i % palette.length] },
    })),
  };
}

export function timeUseGroupedOption(block: TimeUseComparisonBlock): EChartsOption {
  const observedLabel = block.labels[0] ?? "time-use";
  const syntheticLabel = block.labels[1] ?? "synthetic";
  return {
    ...baseOption(),
    grid: { left: 64, right: 24, top: 48, bottom: 120, containLabel: false },
    tooltip: {
      ...baseOption().tooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (value: unknown) => `${Number(value).toFixed(1)} min/day`,
    },
    legend: { ...baseOption().legend, top: 8 },
    xAxis: {
      type: "category",
      data: block.categories,
      axisLabel: { color: COLORS.muted, interval: 0, rotate: 55, fontSize: 10 },
      axisLine: { lineStyle: { color: COLORS.hairline } },
    },
    yAxis: { ...axisCommon("minutes/day"), min: 0 },
    series: [
      {
        name: observedLabel,
        type: "bar",
        data: block.rows.map((row) => row.observed_minutes),
        barMaxWidth: 18,
        itemStyle: { color: ROLE_COLOR.observed },
      },
      {
        name: syntheticLabel,
        type: "bar",
        data: block.rows.map((row) => row.synthetic_minutes),
        barMaxWidth: 18,
        itemStyle: { color: ROLE_COLOR.synthetic },
      },
    ],
  };
}

export function timeUseDifferenceOption(block: TimeUseComparisonBlock): EChartsOption {
  const maxAbs = Math.max(1, ...block.rows.map((row) => Math.abs(row.difference_minutes)));
  return {
    ...baseOption(),
    grid: { left: 118, right: 32, top: 36, bottom: 48, containLabel: false },
    tooltip: {
      ...baseOption().tooltip,
      trigger: "item",
      formatter: (params: unknown) => {
        const p = params as { dataIndex: number; value: number };
        const row = block.rows[p.dataIndex];
        const pct = row.percent_difference == null ? "n/a" : `${row.percent_difference.toFixed(1)}%`;
        return [
          row.category,
          `Synthetic - ${block.labels[0] ?? "time-use"}: ${Number(p.value).toFixed(1)} min/day`,
          `Observed: ${row.observed_minutes.toFixed(1)} min/day`,
          `Synthetic: ${row.synthetic_minutes.toFixed(1)} min/day`,
          `Difference: ${pct}`,
        ].join("<br/>");
      },
    },
    xAxis: { ...axisCommon("Synthetic - time-use minutes/day"), min: -maxAbs, max: maxAbs },
    yAxis: {
      type: "category",
      data: block.categories,
      inverse: true,
      axisLabel: { color: COLORS.muted, interval: 0, fontSize: 10 },
      axisLine: { lineStyle: { color: COLORS.hairline } },
    },
    series: [
      {
        name: "difference",
        type: "bar",
        data: block.rows.map((row) => row.difference_minutes),
        barMaxWidth: 16,
        itemStyle: {
          color: (params: unknown) => {
            const p = params as { value: number };
            return Number(p.value) >= 0 ? COLORS.forest : COLORS.coral;
          },
        },
      },
    ],
  };
}

export function profileScatterOption(block: ProfilesBlock): EChartsOption {
  const symbols = ["circle", "triangle"];
  const series = block.scatter.flatMap((ds, di) =>
    block.profile_order.map((profile) => ({
      name: `${profile} · ${ds.name}`,
      type: "scatter" as const,
      symbol: symbols[di % symbols.length],
      symbolSize: 8,
      data: ds.points.filter((p) => p.profile === profile).map((p) => [p.x, p.y]),
      itemStyle: { color: PROFILE_COLOR[profile] ?? COLORS.ink, opacity: 0.75 },
    })),
  );
  return {
    ...baseOption(),
    legend: { top: 8, type: "scroll", textStyle: { color: COLORS.muted, fontSize: 11 } },
    tooltip: { ...baseOption().tooltip, trigger: "item" },
    xAxis: axisCommon("degree of return"),
    yAxis: axisCommon("intermittency"),
    series,
  };
}

export function profileBoxOption(block: ProfilesBlock, metric: string): EChartsOption {
  const roleFor = (i: number) => (i === 0 ? "synthetic" : "observed");
  const series = block.datasets.map((ds, di) => ({
    name: ds,
    type: "boxplot" as const,
    data: block.profile_order.map((profile) => block.box[metric]?.[ds]?.[profile] ?? [0, 0, 0, 0, 0]),
    itemStyle: { color: COLORS.canvas, borderColor: ROLE_COLOR[roleFor(di)], borderWidth: 1.5 },
  }));
  return {
    ...baseOption(),
    legend: { top: 8, textStyle: { color: COLORS.muted, fontSize: 11 } },
    tooltip: { ...baseOption().tooltip, trigger: "item" },
    xAxis: {
      type: "category",
      data: block.profile_order,
      axisLabel: { color: COLORS.muted },
      axisLine: { lineStyle: { color: COLORS.hairline } },
    },
    yAxis: { ...axisCommon(metric), min: 0 },
    series,
  };
}

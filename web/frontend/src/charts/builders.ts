import type { EChartsOption } from "echarts";
import type {
  ActivityBlock,
  BarSeries,
  EcdfBlock,
  LawBlock,
  MotifsBlock,
  ProfilesBlock,
  SeriesPoints,
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

export function motifOption(block: MotifsBlock): EChartsOption {
  const opt = groupedBar(block.categories, block.series, "% of user-days");
  // motif ids are long hex strings
  (opt.xAxis as { axisLabel: Record<string, unknown> }).axisLabel = {
    color: COLORS.muted,
    interval: 0,
    rotate: 60,
    fontSize: 9,
    fontFamily: "ui-monospace, monospace",
  };
  (opt.grid as { bottom: number }).bottom = 90;
  return opt;
}

function diffHeatmap(
  xLabels: string[],
  yLabels: string[],
  matrix: number[][],
  limit: number,
  labels: string[],
  xName: string,
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
        return `${yLabels[d.data[1]]} → ${xLabels[d.data[0]]}<br/>${d.data[2].toFixed(2)} pp`;
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
      text: [`${labels[1]} higher`, `${labels[0]} higher`],
      textStyle: { color: COLORS.muted, fontSize: 10 },
      inRange: { color: [COLORS.coral, "#f7f7f7", COLORS.forest] },
    },
    series: [
      {
        name: `${labels[1]} − ${labels[0]}`,
        type: "heatmap",
        data,
        label: { show: xLabels.length <= 8, fontSize: 10, formatter: (p: unknown) => (p as { data: number[] }).data[2].toFixed(0) },
      },
    ],
  };
}

export function transitionOption(block: ActivityBlock["transition_difference"]): EChartsOption {
  return diffHeatmap(block.categories, block.categories, block.matrix, block.limit, block.labels, "to activity");
}

export function dailyActivityOption(
  block: NonNullable<ActivityBlock["daily_activity_difference"]>,
): EChartsOption {
  const perHour = Math.max(1, Math.round(block.n_bins / 24));
  const xLabels = Array.from({ length: block.n_bins }, (_, i) => {
    const h = Math.floor(i / perHour);
    return i % perHour === 0 ? `${String(h).padStart(2, "0")}:00` : "";
  });
  const opt = diffHeatmap(xLabels, block.categories, block.matrix, block.limit, block.labels, "time of day");
  (opt.series as { label: { show: boolean } }[])[0].label.show = false;
  return opt;
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

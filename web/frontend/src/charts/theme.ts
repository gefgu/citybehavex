// Chart palette + shared ECharts styling, derived from DESIGN.md tokens so plots
// match the page. Data series use the signature colors (coral / forest / mustard).

import type { EChartsOption } from "echarts";

export const COLORS = {
  ink: "#181d26",
  body: "#333840",
  muted: "#41454d",
  hairline: "#dddddd",
  canvas: "#ffffff",
  coral: "#aa2d00",
  forest: "#0a2e0e",
  mustard: "#d9a441",
  info: "#254fad",
  cream: "#f5e9d4",
  peach: "#fcab79",
  mint: "#a8d8c4",
  yellow: "#f4d35e",
};

// role -> line/marker color
export const ROLE_COLOR: Record<string, string> = {
  synthetic: COLORS.coral,
  observed: COLORS.forest,
  reference: COLORS.mustard,
};

// role -> dashed pattern for line series
export const ROLE_DASH: Record<string, number[] | undefined> = {
  synthetic: [6, 4],
  observed: undefined,
  reference: [2, 4],
};

// profile -> color for the mobility-profile scatter
export const PROFILE_COLOR: Record<string, string> = {
  Scouter: COLORS.coral,
  Regular: COLORS.mustard,
  Routiner: COLORS.forest,
};

// purpose -> color for the timeline agent map + legend
export const PURPOSE_COLOR: Record<string, string> = {
  HOME: COLORS.forest,
  WORK: COLORS.coral,
  STUDIES: COLORS.info,
  PURCHASE: COLORS.mustard,
  LEISURE: COLORS.mint,
  HEALTH: COLORS.peach,
  OTHER: COLORS.yellow,
};
export const DEFAULT_PURPOSE_COLOR = COLORS.muted;

const FONT = '"Inter Display", Inter, system-ui, sans-serif';
const MONO = 'ui-monospace, "SF Mono", monospace';

export function axisCommon(name: string, log = false) {
  return {
    type: log ? ("log" as const) : ("value" as const),
    name: name.toUpperCase(),
    nameLocation: "middle" as const,
    nameGap: 40,
    nameTextStyle: { fontFamily: MONO, fontSize: 12, color: COLORS.muted },
    axisLabel: { fontFamily: MONO, fontSize: 11, color: COLORS.muted },
    axisLine: { lineStyle: { color: COLORS.hairline } },
    splitLine: { lineStyle: { color: COLORS.hairline, type: [2, 5] as number[] } },
  };
}

// Baseline option merged into every chart for consistent look.
export function baseOption(): EChartsOption {
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: FONT, color: COLORS.body },
    grid: { left: 64, right: 24, top: 48, bottom: 56, containLabel: false },
    legend: {
      top: 8,
      textStyle: { fontFamily: FONT, fontSize: 12, color: COLORS.muted },
    },
    tooltip: {
      backgroundColor: COLORS.canvas,
      borderColor: COLORS.hairline,
      textStyle: { color: COLORS.body, fontFamily: FONT },
    },
  };
}

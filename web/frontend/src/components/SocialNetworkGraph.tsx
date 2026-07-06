import { useCallback, useMemo } from "react";
import type * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import type { SocialNetworkBlock } from "../api";
import { EChart } from "../charts/EChart";
import { COLORS } from "../charts/theme";

type GraphGLControl = {
  minZoom?: number;
  maxZoom?: number;
};

type GraphGLView = {
  _control?: GraphGLControl;
};

type GraphRuntimeChart = {
  getModel: () => {
    getSeriesByIndex: (index: number) => unknown;
  };
  getViewOfSeriesModel: (model: unknown) => GraphGLView | undefined;
};

export function SocialNetworkGraph({ block, title }: { block: SocialNetworkBlock; title?: string }) {
  const deepenGraphZoom = useCallback((chart: echarts.ECharts) => {
    const runtimeChart = chart as unknown as GraphRuntimeChart;
    const model = runtimeChart.getModel().getSeriesByIndex(0);
    const view = model ? runtimeChart.getViewOfSeriesModel(model) : undefined;
    if (!view?._control) return;
    view._control.minZoom = 0.08;
    view._control.maxZoom = 48;
  }, []);

  const option = useMemo<EChartsOption>(() => {
    const data = block.nodes.map((node, index) => ({
      id: String(index),
      name: `agent ${node[3]}`,
      x: node[0],
      y: node[1],
      value: block.degrees?.[index] ?? 0,
      symbolSize: node[2],
      profileType: node[4] ?? null,
      itemStyle: {
        color: node[4] ? COLORS.forest : COLORS.info,
        opacity: 0.9,
      },
    }));
    const edges = block.edges.map((edge) => ({
      source: String(edge[0]),
      target: String(edge[1]),
      value: edge[2] ?? 1,
    }));

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        confine: true,
        formatter: (param: unknown) => {
          const p = param as {
            dataType?: string;
            data?: {
              name?: string;
              value?: number;
              symbolSize?: number;
              profileType?: string | null;
              source?: string;
              target?: string;
            };
          };
          if (p.dataType === "edge") {
            return `edge ${p.data?.source} → ${p.data?.target}`;
          }
          const bits = [
            p.data?.name ?? "agent",
            `degree ${p.data?.value ?? 0}`,
            `size ${Number(p.data?.symbolSize ?? 0).toFixed(1)}`,
          ];
          if (p.data?.profileType) bits.push(String(p.data.profileType));
          return bits.join("<br/>");
        },
      },
      series: [
        {
          type: "graphGL",
          layout: "none",
          data,
          nodes: data,
          links: edges,
          edges,
          roam: true,
          zoom: 2.1,
          draggable: false,
          symbol: "circle",
          lineStyle: {
            color: "rgba(24,29,38,0.16)",
            width: 1,
            opacity: 0.22,
          },
          emphasis: {
            itemStyle: { color: COLORS.coral },
            lineStyle: { opacity: 0.55 },
          },
        },
      ],
    } as EChartsOption;
  }, [block]);

  return (
    <div>
      {title && <h4>{title}</h4>}
      <div className="network-meta">
        <span>{block.node_count.toLocaleString()} nodes</span>
        <span>
          {block.edge_count.toLocaleString()} edges
          {block.edges_sampled ? ` (showing a ${block.edges.length.toLocaleString()}-edge sample)` : ""}
        </span>
        <span>{block.layout}</span>
        <span>k={block.social_graph_k}</span>
      </div>
      <EChart
        option={option}
        className="network-graph"
        preventPageScrollOnWheel
        onOptionApplied={deepenGraphZoom}
      />
    </div>
  );
}

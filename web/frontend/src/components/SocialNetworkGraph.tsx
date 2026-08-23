import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { SocialNetworkBlock } from "../api";
import { EChart } from "../charts/EChart";
import { COLORS } from "../charts/theme";

export function SocialNetworkGraph({ block, title }: { block: SocialNetworkBlock; title?: string }) {
  const option = useMemo<EChartsOption>(() => {
    // graphGL's camera becomes unreliable when a saved layout has a large
    // absolute offset (for example, an uncentred TruncatedSVD projection).
    // Keep the layout's relative geometry while putting every graph in a
    // stable, renderer-friendly coordinate range.
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const node of block.nodes) {
      if (Number.isFinite(node[0])) {
        minX = Math.min(minX, node[0]);
        maxX = Math.max(maxX, node[0]);
      }
      if (Number.isFinite(node[1])) {
        minY = Math.min(minY, node[1]);
        maxY = Math.max(maxY, node[1]);
      }
    }
    const centerX = Number.isFinite(minX) ? (minX + maxX) / 2 : 0;
    const centerY = Number.isFinite(minY) ? (minY + maxY) / 2 : 0;
    const span = Math.max(
      Number.isFinite(minX) ? maxX - minX : 0,
      Number.isFinite(minY) ? maxY - minY : 0,
      1,
    );
    const coordinateScale = 600 / span;
    const data = block.nodes.map((node, index) => ({
      id: String(index),
      name: `agent ${node[3]}`,
      x: Number.isFinite(node[0]) ? (node[0] - centerX) * coordinateScale : 0,
      y: Number.isFinite(node[1]) ? (node[1] - centerY) * coordinateScale : 0,
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
        <span>
          {block.node_count.toLocaleString()} nodes
          {block.nodes_sampled ? ` (showing ${block.nodes.length.toLocaleString()})` : ""}
        </span>
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
      />
    </div>
  );
}

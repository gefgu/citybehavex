import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import type { SocialNetworkBlock } from "../api";
import { EChart } from "../charts/EChart";
import { COLORS } from "../charts/theme";

export function SocialNetworkGraph({ block }: { block: SocialNetworkBlock }) {
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
      <div className="network-meta">
        <span>{block.node_count.toLocaleString()} nodes</span>
        <span>{block.edge_count.toLocaleString()} edges</span>
        <span>{block.layout}</span>
        <span>k={block.social_graph_k}</span>
      </div>
      <EChart option={option} className="network-graph" preventPageScrollOnWheel />
    </div>
  );
}

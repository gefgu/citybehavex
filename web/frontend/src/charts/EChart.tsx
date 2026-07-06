import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import "echarts-gl";
import type { EChartsOption } from "echarts";

// Minimal imperative ECharts wrapper: init on mount, setOption on change,
// resize with the container, dispose on unmount.
export function EChart({
  option,
  className = "echart",
  preventPageScrollOnWheel = false,
  onOptionApplied,
}: {
  option: EChartsOption;
  className?: string;
  preventPageScrollOnWheel?: boolean;
  onOptionApplied?: (chart: echarts.ECharts) => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption(option, true);
    onOptionApplied?.(chart);
  }, [option, onOptionApplied]);

  useEffect(() => {
    if (!ref.current || !preventPageScrollOnWheel) return;

    const el = ref.current;
    const preventWheelScroll = (event: WheelEvent) => {
      event.preventDefault();
    };
    el.addEventListener("wheel", preventWheelScroll, { passive: false });
    return () => {
      el.removeEventListener("wheel", preventWheelScroll);
    };
  }, [preventPageScrollOnWheel]);

  return <div ref={ref} className={className} />;
}

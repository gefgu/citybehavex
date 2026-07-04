import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { EChartsOption } from "echarts";
import { fetchCharts, type ChartPayload } from "../api";
import { EChart } from "../charts/EChart";
import { SocialNetworkGraph } from "../components/SocialNetworkGraph";
import { StvdMap } from "../components/StvdMap";
import {
  dailyActivityOption,
  ecdfOption,
  lawOption,
  microActivityUsageOption,
  motifOption,
  profileBoxOption,
  profileScatterOption,
  purposeOption,
  transitionOption,
} from "../charts/builders";

function ChartCard({
  title,
  option,
  wide = false,
  subtitle,
}: {
  title: string;
  option: EChartsOption;
  wide?: boolean;
  subtitle?: string;
}) {
  return (
    <div className={`chart-card${wide ? " wide" : ""}`}>
      <h4>{title}</h4>
      <EChart option={option} />
      {subtitle && <div className="fit-params">{subtitle}</div>}
    </div>
  );
}

const ECDF_TITLES: Record<string, string> = {
  jump_lengths: "Jump length",
  visits_per_user: "Visits per user",
  radius_of_gyration: "Radius of gyration",
  dwell_time: "Dwell time",
  trip_duration: "Trip duration",
};

type FilterChoice = { key: string; label: string };

const DAY_FILTERS: FilterChoice[] = [
  { key: "all", label: "All" },
  { key: "weekday", label: "Weekday" },
  { key: "weekend", label: "Weekend" },
];

const PERIOD_FILTERS: FilterChoice[] = [
  { key: "all", label: "All day" },
  { key: "morning", label: "Morning" },
  { key: "afternoon", label: "Afternoon" },
  { key: "evening", label: "Evening" },
  { key: "night", label: "Night" },
];

function SegmentedControl({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: FilterChoice[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="segmented-control" aria-label={label}>
      {options.map((option) => (
        <button
          className={option.key === value ? "active" : ""}
          key={option.key}
          onClick={() => onChange(option.key)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function SectionHeading({
  title,
  controls,
}: {
  title: string;
  controls?: ReactNode;
}) {
  return (
    <div className="section-heading-row">
      <div className="section-header">{title}</div>
      {controls && <div className="section-controls">{controls}</div>}
    </div>
  );
}

function metricName(m: { metric_name?: string; name?: string }) {
  return m.metric_name ?? m.name ?? "Metric";
}

function FilteredMetricTable({
  title,
  rows,
  unit,
}: {
  title: string;
  rows: { filter_key?: string; filter_label?: string; metric_name?: string; name?: string; value: number; unit?: string; resolution?: number }[];
  unit?: string;
}) {
  if (rows.length === 0) return null;
  return (
    <div>
      <h4>{title}</h4>
      <table className="metrics">
        <tbody>
          {rows.map((m, i) => (
            <tr key={`${m.filter_key ?? "all"}:${metricName(m)}:${m.resolution ?? i}`}>
              <td>
                <span className="metric-filter">{m.filter_label ?? "All"}</span>
                {m.resolution ? `H3 ${m.resolution}` : metricName(m)}
              </td>
              <td className="value">{m.value.toFixed(4)}</td>
              <td className="unit">{m.unit ?? unit ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Charts() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const run = params.get("run") ?? undefined;
  const [payload, setPayload] = useState<ChartPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dayFilter, setDayFilter] = useState("all");
  const [distributionFilter, setDistributionFilter] = useState("all");
  const setSyncedDayFilter = (next: string) => {
    setDayFilter(next);
    if (["all", "weekday", "weekend"].includes(distributionFilter)) {
      setDistributionFilter(next);
    }
  };

  useEffect(() => {
    setPayload(null);
    setError(null);
    fetchCharts(id, run).then(setPayload).catch((e) => setError(String(e)));
  }, [id, run]);

  const fitSubtitle = useMemo(
    () => (fits: { label: string; params: Record<string, number> }[]) =>
      fits
        .map(
          (f) =>
            `${f.label}: ${Object.entries(f.params)
              .map(([k, v]) => `${k}=${Number(v).toPrecision(3)}`)
              .join(", ")}`,
        )
        .join("  ·  "),
    [],
  );

  if (error) return <div className="state">Failed to load charts: {error}</div>;
  if (!payload)
    return <div className="state">Building comparison… (first load can take a while)</div>;

  const { metrics } = payload;
  const metricFilter = distributionFilter === "all" ? dayFilter : distributionFilter;
  const metricRows = {
    wasserstein: metrics.wasserstein.filter((m) => m.filter_key === metricFilter),
    jsd: metrics.jsd.filter((m) => (m.filter_key ?? "all") === dayFilter),
    cpc: metrics.cpc.filter((m) => m.filter_key === dayFilter),
  };
  const ecdfGroup =
    payload.ecdf.groups.find((group) => group.filter_key === distributionFilter) ??
    payload.ecdf.groups[0];
  const mobilityGroup =
    payload.mobility_laws?.groups.find((group) => group.filter_key === dayFilter) ??
    payload.mobility_laws?.groups[0];
  const activityGroup =
    payload.activity?.groups.find((group) => group.filter_key === dayFilter) ??
    payload.activity?.groups[0];
  const microActivityGroup =
    payload.micro_activity_usage?.groups.find((group) => group.filter_key === dayFilter) ??
    payload.micro_activity_usage?.groups[0];
  const motifGroup =
    payload.motifs?.groups.find((group) => group.filter_key === dayFilter) ??
    payload.motifs?.groups[0];
  const stvdGroup =
    payload.stvd?.groups.find((group) => group.filter_key === dayFilter) ??
    payload.stvd?.groups[0];
  const titleLabel =
    payload.mode === "comparison" && payload.labels.observed
      ? `${payload.labels.observed} vs synthetic`
      : "Synthetic analysis";

  return (
    <>
      <h1 style={{ margin: "48px 0 4px" }}>
        {titleLabel}
      </h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        <Link to="/experiments">experiments</Link> / {id} · run{" "}
        <code>{payload.run_id}</code>
      </p>

      {payload.warnings.length > 0 && (
        <div className="warnings">
          {payload.warnings.length} section(s) skipped: {payload.warnings.join("; ")}
        </div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Metrics day type filter"
            onChange={setSyncedDayFilter}
            options={DAY_FILTERS}
            value={dayFilter}
          />
        }
        title="Metrics"
      />
      {payload.mode === "synthetic_only" ? (
        <div className="network-empty">
          Synthetic-only mode. Add an observed comparison parquet to show Wasserstein,
          Jensen-Shannon and CPC metrics.
        </div>
      ) : (
        <div className="metric-tables">
          <FilteredMetricTable title="Wasserstein distances" rows={metricRows.wasserstein} />
          <FilteredMetricTable title="Jensen-Shannon divergences" rows={metricRows.jsd} />
          <FilteredMetricTable title="Common Part of Commuters" rows={metricRows.cpc} />
        </div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Distribution filter"
            onChange={setDistributionFilter}
            options={[...DAY_FILTERS, ...PERIOD_FILTERS.filter((option) => option.key !== "all")]}
            value={distributionFilter}
          />
        }
        title="Distribution comparisons"
      />
      {ecdfGroup && (
        <div className="chart-grid">
          {Object.entries(ecdfGroup.blocks).map(([key, block]) => (
            <ChartCard key={key} title={`${ECDF_TITLES[key] ?? key} ECDF`} option={ecdfOption(block)} />
          ))}
        </div>
      )}

      {mobilityGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Mobility laws day type filter"
                onChange={setSyncedDayFilter}
                options={DAY_FILTERS}
                value={dayFilter}
              />
            }
            title="Mobility laws"
          />
          <div className="chart-grid">
            {Object.entries(mobilityGroup.blocks).map(([key, block]) => (
              <ChartCard
                key={key}
                title={block.title}
                option={lawOption(block)}
                subtitle={fitSubtitle(block.fits)}
              />
            ))}
          </div>
        </>
      )}

      {activityGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Activity day type filter"
                onChange={setSyncedDayFilter}
                options={DAY_FILTERS}
                value={dayFilter}
              />
            }
            title="Activity comparison"
          />
          <div className="chart-grid">
            <ChartCard title="Visit purpose comparison" option={purposeOption(activityGroup.purpose)} wide />
            <ChartCard
              title={activityGroup.transition_difference.matrix_mode === "raw" ? "Activity transitions" : "Activity transition difference"}
              option={transitionOption(activityGroup.transition_difference)}
            />
            {activityGroup.daily_activity_difference && (
              <ChartCard
                title={activityGroup.daily_activity_difference.matrix_mode === "raw" ? "Daily activity" : "Daily activity difference"}
                option={dailyActivityOption(activityGroup.daily_activity_difference)}
              />
            )}
          </div>
        </>
      )}

      {microActivityGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Micro-activity day type filter"
                onChange={setSyncedDayFilter}
                options={DAY_FILTERS}
                value={dayFilter}
              />
            }
            title="Synthetic micro-activity usage"
          />
          <ChartCard
            title="Mean daily usage over the day"
            option={microActivityUsageOption(microActivityGroup.block)}
            wide
          />
        </>
      )}

      {motifGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Motifs day type filter"
                onChange={setSyncedDayFilter}
                options={DAY_FILTERS}
                value={dayFilter}
              />
            }
            title="Daily motifs"
          />
          <div className="chart-grid">
            <ChartCard title="Motif literature comparison" option={motifOption(motifGroup.block)} wide />
          </div>
        </>
      )}

      {stvdGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="STVD day type filter"
                onChange={setSyncedDayFilter}
                options={DAY_FILTERS}
                value={dayFilter}
              />
            }
            title="Spatial-temporal volume difference"
          />
          <StvdMap block={stvdGroup.block} />
        </>
      )}

      <SectionHeading title="Social network" />
      {payload.social_network ? (
        <SocialNetworkGraph block={payload.social_network} />
      ) : (
        <div className="network-empty">
          No social network sidecar found for this run. Re-run the simulation with the latest code,
          then refresh the chart payload.
        </div>
      )}

      {payload.profiles && (
        <>
          <SectionHeading title="Mobility profiles" />
          <div className="chart-grid">
            <ChartCard
              title="Intermittency vs degree of return"
              option={profileScatterOption(payload.profiles)}
              wide
            />
            {payload.profiles.metrics.map((metric) => (
              <ChartCard
                key={metric}
                title={metric[0].toUpperCase() + metric.slice(1)}
                option={profileBoxOption(payload.profiles!, metric)}
              />
            ))}
          </div>
        </>
      )}

      <div style={{ height: 96 }} />
    </>
  );
}

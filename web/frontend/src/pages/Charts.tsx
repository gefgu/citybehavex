import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { EChartsOption } from "echarts";
import {
  fetchChartSection,
  fetchCharts,
  fetchHomeWork,
  fetchNetworkValidation,
  downloadMetricsExport,
  type ChartPayload,
  type DemographicFilter as DemographicFilterValue,
  type HomeWorkResponse,
  type NetworkValidationComparisonBlock,
  type NetworkValidationResponse,
} from "../api";
import { EChart } from "../charts/EChart";
import { DemographicFilter } from "../components/DemographicFilter";
import { HomeWorkMap } from "../components/HomeWorkMap";
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
  timeUseDifferenceOption,
  timeUseGroupedOption,
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

const NETWORK_VALIDATION_TITLES: Record<string, string> = {
  degree: "Degree",
  clustering_coefficient: "Clustering coefficient",
  edge_persistence: "Edge persistence",
  topological_overlap: "Topological overlap",
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

const FILTERED_SECTIONS = [
  "distributions",
  "metrics",
  "activity",
  "mobility-laws",
  "micro-activity",
  "time-use",
  "motifs",
  "stvd",
];
const STATIC_SECTIONS = ["profiles", "social-network"];
const SECTION_REQUEST_TIMEOUT_MS = 20_000;

function sectionKey(section: string, filter = "all") {
  return `${section}:${filter}`;
}

function defaultSectionRequests(
  dayFilter: string,
  distributionFilter: string,
  fastOnly = false,
): [string, string][] {
  if (fastOnly) {
    return [
      ["time-use", dayFilter],
      ["social-network", "all"],
    ];
  }
  const metricFilter = distributionFilter === "all" ? dayFilter : distributionFilter;
  return [
    ["micro-activity", dayFilter],
    ["time-use", dayFilter],
    ["activity", dayFilter],
    ["motifs", dayFilter],
    ...STATIC_SECTIONS.map((section): [string, string] => [section, "all"]),
    ["metrics", metricFilter],
    ["distributions", distributionFilter],
    ["mobility-laws", dayFilter],
    ["stvd", dayFilter],
  ];
}

function mergeGroups<T extends { filter_key: string }>(
  current: { groups: T[] } | null,
  incoming: { groups: T[] } | null,
): { groups: T[] } | null {
  if (!incoming) return current;
  const byKey = new Map((current?.groups ?? []).map((group) => [group.filter_key, group]));
  for (const group of incoming.groups) byKey.set(group.filter_key, group);
  return { groups: Array.from(byKey.values()) };
}

function mergeMetricRows<T extends { filter_key?: string }>(current: T[], incoming: T[]): T[] {
  const incomingKeys = new Set(incoming.map((row) => row.filter_key ?? "all"));
  return [...current.filter((row) => !incomingKeys.has(row.filter_key ?? "all")), ...incoming];
}

function mergeChartPayload(current: ChartPayload, incoming: ChartPayload): ChartPayload {
  const loaded = new Set([...(current.loaded_filters ?? ["all"]), ...(incoming.loaded_filters ?? [])]);
  return {
    ...current,
    warnings: Array.from(new Set([...current.warnings, ...incoming.warnings])),
    loaded_filters: Array.from(loaded),
    metrics: {
      wasserstein: mergeMetricRows(current.metrics.wasserstein ?? [], incoming.metrics.wasserstein ?? []),
      jsd: mergeMetricRows(current.metrics.jsd ?? [], incoming.metrics.jsd ?? []),
      cpc: mergeMetricRows(current.metrics.cpc ?? [], incoming.metrics.cpc ?? []),
      time_use: mergeMetricRows(current.metrics.time_use ?? [], incoming.metrics.time_use ?? []),
      stvd: mergeMetricRows(current.metrics.stvd ?? [], incoming.metrics.stvd ?? []),
    },
    ecdf: mergeGroups(current.ecdf, incoming.ecdf) ?? current.ecdf,
    mobility_laws: mergeGroups(current.mobility_laws, incoming.mobility_laws),
    activity: mergeGroups(current.activity, incoming.activity),
    micro_activity_usage: mergeGroups(current.micro_activity_usage, incoming.micro_activity_usage),
    time_use_comparison: mergeGroups(current.time_use_comparison, incoming.time_use_comparison),
    motifs: mergeGroups(current.motifs, incoming.motifs),
    stvd: mergeGroups(current.stvd, incoming.stvd),
    profiles: incoming.profiles ?? current.profiles,
    social_network: incoming.social_network ?? current.social_network,
  };
}

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

function NetworkValidationTable({
  validation,
  sourceLabel,
}: {
  validation: NetworkValidationComparisonBlock | undefined;
  sourceLabel: "synthetic" | "observed";
}) {
  if (!validation) return null;
  return (
    <div>
      <h4>{sourceLabel === "synthetic" ? "Synthetic" : "Observed"} vs random Wasserstein</h4>
      <table className="metrics">
        <tbody>
          {Object.entries(NETWORK_VALIDATION_TITLES).map(([key, label]) => {
            const value = validation.wasserstein[key as keyof typeof validation.wasserstein];
            const source = validation.distributions[sourceLabel]?.[key];
            const rnd = validation.distributions.random[key];
            return (
              <tr key={key}>
                <td>
                  {label}
                  <span className="metric-filter">
                    {sourceLabel} n={source?.count ?? 0} · random n={rnd?.count ?? 0}
                  </span>
                </td>
                <td className="value">{value == null ? "n/a" : value.toFixed(4)}</td>
                <td className="unit" />
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function NetworkValidationSection({
  block,
  sourceLabel,
  sourceTitle,
}: {
  block: NetworkValidationComparisonBlock | undefined;
  sourceLabel: "synthetic" | "observed";
  sourceTitle: string;
}) {
  if (!block) return null;
  return (
    <>
      <div className="metric-tables">
        <NetworkValidationTable validation={block} sourceLabel={sourceLabel} />
      </div>
      <div className="network-validation-grid">
        <SocialNetworkGraph block={block.source_network} title={sourceTitle} />
        <SocialNetworkGraph block={block.random_network} title="Degree-preserving random" />
      </div>
    </>
  );
}

export function Charts() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const run = params.get("run") ?? undefined;
  const [payload, setPayload] = useState<ChartPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingSections, setLoadingSections] = useState<Set<string>>(new Set());
  const [sectionErrors, setSectionErrors] = useState<Record<string, string>>({});
  const [dayFilter, setDayFilter] = useState("all");
  const [distributionFilter, setDistributionFilter] = useState("all");
  const [homeWork, setHomeWork] = useState<HomeWorkResponse | null>(null);
  const [networkValidation, setNetworkValidation] = useState<NetworkValidationResponse | null>(null);
  const [networkValidationError, setNetworkValidationError] = useState<string | null>(null);
  const [exportingMetrics, setExportingMetrics] = useState(false);
  const requestScopeRef = useRef(0);
  const loadingSectionsRef = useRef<Set<string>>(new Set());
  const loadedSectionsRef = useRef<Set<string>>(new Set());
  const [demoFilter, setDemoFilter] = useState<DemographicFilterValue>({
    gender: null,
    age_bracket: null,
    job: null,
  });
  // demoFilter is read by the sequential chain below without being a
  // dependency of it (see the comment on demoFilterRef), so the chain
  // doesn't re-run on every filter tweak -- only the dedicated
  // demoFilter-only effect further down does.
  const demoFilterRef = useRef(demoFilter);
  useEffect(() => {
    demoFilterRef.current = demoFilter;
  }, [demoFilter]);
  const demoSafeMode = id.includes("yjmob2");

  const loadSectionRequests = useCallback(
    async (basePayload: ChartPayload, requests: [string, string][], requestScope: number) => {
      const enabled = basePayload.enabled_sections ?? [...FILTERED_SECTIONS, ...STATIC_SECTIONS];
      const jobs: [string, string, string][] = [];
      for (const [section, filter] of requests) {
        if (!enabled.includes(section)) continue;
        const key = sectionKey(section, filter);
        if (loadedSectionsRef.current.has(key) || loadingSectionsRef.current.has(key)) continue;

        loadingSectionsRef.current.add(key);
        jobs.push([section, filter, key]);
      }

      if (jobs.length === 0) return;
      setLoadingSections((current) => {
        const copy = new Set(current);
        for (const [, , key] of jobs) copy.add(key);
        return copy;
      });

      for (const [section, filter, key] of jobs) {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), SECTION_REQUEST_TIMEOUT_MS);
        try {
          const next = await fetchChartSection(id, section, filter, run, controller.signal);
          if (requestScopeRef.current !== requestScope) return;
          setPayload((current) => (current ? mergeChartPayload(current, next) : next));
          loadedSectionsRef.current.add(key);
          setSectionErrors((current) => {
            const copy = { ...current };
            delete copy[key];
            return copy;
          });
        } catch (e) {
          if (requestScopeRef.current === requestScope) {
            const message =
              e instanceof DOMException && e.name === "AbortError"
                ? "Timed out while building this section; reload to retry."
                : String(e);
            setSectionErrors((current) => ({ ...current, [key]: message }));
          }
        } finally {
          window.clearTimeout(timeout);
          if (requestScopeRef.current === requestScope) {
            loadingSectionsRef.current.delete(key);
            setLoadingSections((current) => {
              const copy = new Set(current);
              copy.delete(key);
              return copy;
            });
          }
        }
      }
    },
    [id, run],
  );

  // Sequential on purpose (per-request server load, not just UI ergonomics):
  // charts -> home-work -> network-validation run one after another instead
  // of all three firing in parallel on mount, so a single tab doesn't triple
  // the concurrent load on the backend's comparison-payload builder.
  useEffect(() => {
    let cancelled = false;
    requestScopeRef.current += 1;
    setPayload(null);
    setError(null);
    setHomeWork(null);
    setNetworkValidation(null);
    setNetworkValidationError(null);
    loadingSectionsRef.current = new Set();
    loadedSectionsRef.current = new Set();
    setLoadingSections(new Set());
    setSectionErrors({});

    (async () => {
      let chartsResult: ChartPayload;
      try {
        chartsResult = await fetchCharts(id, run);
      } catch (e) {
        if (!cancelled) setError(String(e));
        return; // nothing else is worth fetching if the main payload failed
      }
      if (cancelled) return;
      setPayload(chartsResult);
      await loadSectionRequests(
        chartsResult,
        defaultSectionRequests("all", "all", demoSafeMode),
        requestScopeRef.current,
      );
      if (cancelled) return;

      try {
        const hw = await fetchHomeWork(id, run, demoFilterRef.current);
        if (!cancelled) setHomeWork(hw);
      } catch {
        if (!cancelled) setHomeWork(null);
      }

      // network_validation is the single largest section to build for
      // shanghai/yjmob-scale simulations (see web/backend/app/api/charts.py's
      // /network-validation route) -- still fetched last, independent of
      // whether home-work succeeded, so one failing section doesn't block
      // the other.
      if (!demoSafeMode) {
        try {
          const nv = await fetchNetworkValidation(id, run);
          if (!cancelled) setNetworkValidation(nv);
        } catch (e) {
          if (!cancelled) setNetworkValidationError(String(e));
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [id, run, loadSectionRequests, demoSafeMode]);

  useEffect(() => {
    if (!payload) return;
    void loadSectionRequests(
      payload,
      defaultSectionRequests(dayFilter, distributionFilter, demoSafeMode),
      requestScopeRef.current,
    );
  }, [payload, dayFilter, distributionFilter, loadSectionRequests, demoSafeMode]);

  // Demographic-filter-only refetch: the sequential chain above already
  // covers the initial home-work fetch on mount/id/run change, so this only
  // needs to react to demoFilter changing on its own.
  const isInitialDemoRender = useRef(true);
  useEffect(() => {
    if (isInitialDemoRender.current) {
      isInitialDemoRender.current = false;
      return;
    }
    let cancelled = false;
    fetchHomeWork(id, run, demoFilter)
      .then((hw) => {
        if (!cancelled) setHomeWork(hw);
      })
      .catch(() => {
        if (!cancelled) setHomeWork(null);
      });
    return () => {
      cancelled = true;
    };
  }, [demoFilter]);

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
  // Day-type filter options (all/weekday/weekend, plus any config-declared
  // special day like "emergency") are computed server-side and attached to
  // every group that's partitioned by day type; derive the toggle list from
  // whichever of those groups is present instead of hardcoding the 3 defaults.
  const dayFilters: FilterChoice[] = payload.available_filters ?? DAY_FILTERS;
  const distributionFilters: FilterChoice[] =
    payload.distribution_filters ??
    [...dayFilters, ...PERIOD_FILTERS.filter((option) => option.key !== "all")];
  const dayFilterKeys = dayFilters.map((f) => f.key);
  const setSyncedDayFilter = (next: string) => {
    setDayFilter(next);
    if (dayFilterKeys.includes(distributionFilter)) {
      setDistributionFilter(next);
    }
  };
  const metricFilter = distributionFilter === "all" ? dayFilter : distributionFilter;
  const metricRows = {
    wasserstein: (metrics.wasserstein ?? []).filter((m) => m.filter_key === metricFilter),
    jsd: (metrics.jsd ?? []).filter((m) => (m.filter_key ?? "all") === dayFilter),
    cpc: (metrics.cpc ?? []).filter((m) => m.filter_key === dayFilter),
    time_use: (metrics.time_use ?? []).filter((m) => m.filter_key === dayFilter),
    stvd: (metrics.stvd ?? []).filter((m) => m.filter_key === metricFilter),
  };
  const ecdfGroup =
    payload.ecdf.groups.find((group) => group.filter_key === distributionFilter) ??
    (distributionFilter === "all" ? payload.ecdf.groups[0] : undefined);
  const mobilityGroup =
    payload.mobility_laws?.groups.find((group) => group.filter_key === dayFilter) ??
    (dayFilter === "all" ? payload.mobility_laws?.groups[0] : undefined);
  const activityGroup =
    payload.activity?.groups.find((group) => group.filter_key === dayFilter) ??
    (dayFilter === "all" ? payload.activity?.groups[0] : undefined);
  const microActivityGroup =
    payload.micro_activity_usage?.groups.find((group) => group.filter_key === dayFilter) ??
    (dayFilter === "all" ? payload.micro_activity_usage?.groups[0] : undefined);
  const timeUseGroup =
    payload.time_use_comparison?.groups.find((group) => group.filter_key === dayFilter) ??
    (dayFilter === "all" ? payload.time_use_comparison?.groups[0] : undefined);
  const motifGroup =
    payload.motifs?.groups.find((group) => group.filter_key === dayFilter) ??
    (dayFilter === "all" ? payload.motifs?.groups[0] : undefined);
  const stvdGroup =
    payload.stvd?.groups.find((group) => group.filter_key === dayFilter) ??
    (dayFilter === "all" ? payload.stvd?.groups[0] : undefined);
  const isSectionLoading = (section: string, filter = "all") =>
    loadingSections.has(sectionKey(section, filter));
  const sectionError = (section: string, filter = "all") =>
    sectionErrors[sectionKey(section, filter)];
  const metricSectionFilter = distributionFilter === "all" ? dayFilter : distributionFilter;
  const metricsLoading = isSectionLoading("metrics", metricSectionFilter);
  const distributionFilterLoading = isSectionLoading("distributions", distributionFilter);
  const titleLabel =
    payload.mode === "comparison" && payload.labels.observed
      ? `${payload.labels.observed} vs synthetic`
      : "Synthetic analysis";
  const handleMetricsExport = async () => {
    setExportingMetrics(true);
    try {
      const blob = await downloadMetricsExport(id, run);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `citybehavex-${id}-${payload.run_id}-metrics.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } finally {
      setExportingMetrics(false);
    }
  };

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
          <>
            <SegmentedControl
              label="Metrics day type filter"
              onChange={setSyncedDayFilter}
              options={dayFilters}
              value={dayFilter}
            />
            <button
              className="btn btn-secondary"
              disabled={exportingMetrics}
              onClick={() => void handleMetricsExport()}
              type="button"
            >
              {exportingMetrics ? "Exporting..." : "Export JSON"}
            </button>
          </>
        }
        title="Metrics"
      />
      {payload.mode === "synthetic_only" ? (
        <div className="network-empty">
          Synthetic-only mode. Add an observed comparison parquet to show Wasserstein,
          Jensen-Shannon and CPC metrics.
        </div>
      ) : metricsLoading ? (
        <div className="state">Building selected metrics…</div>
      ) : (
        <div className="metric-tables">
          <FilteredMetricTable title="Wasserstein distances" rows={metricRows.wasserstein} />
          <FilteredMetricTable title="Jensen-Shannon divergences" rows={metricRows.jsd} />
          <FilteredMetricTable title="Common Part of Commuters" rows={metricRows.cpc} />
          <FilteredMetricTable title="Time-use metrics" rows={metricRows.time_use} />
          <FilteredMetricTable title="STVD distances" rows={metricRows.stvd} />
        </div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Distribution filter"
            onChange={setDistributionFilter}
            options={distributionFilters}
            value={distributionFilter}
          />
        }
        title="Distribution comparisons"
      />
      {distributionFilterLoading && (
        <div className="state">Building {distributionFilter} distribution…</div>
      )}
      {sectionError("distributions", distributionFilter) && (
        <div className="state">Failed to load distribution: {sectionError("distributions", distributionFilter)}</div>
      )}
      {!distributionFilterLoading && ecdfGroup && (
        <div className="chart-grid">
          {Object.entries(ecdfGroup.blocks).map(([key, block]) => (
            <ChartCard key={key} title={`${ECDF_TITLES[key] ?? key} ECDF`} option={ecdfOption(block)} />
          ))}
        </div>
      )}

      {isSectionLoading("mobility-laws", dayFilter) && <div className="state">Building mobility laws…</div>}
      {!isSectionLoading("mobility-laws", dayFilter) && mobilityGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Mobility laws day type filter"
                onChange={setSyncedDayFilter}
                options={dayFilters}
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

      {isSectionLoading("activity", dayFilter) && <div className="state">Building activity comparison…</div>}
      {!isSectionLoading("activity", dayFilter) && activityGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Activity day type filter"
                onChange={setSyncedDayFilter}
                options={dayFilters}
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

      {isSectionLoading("micro-activity", dayFilter) && <div className="state">Building micro-activity usage…</div>}
      {!isSectionLoading("micro-activity", dayFilter) && microActivityGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Micro-activity day type filter"
                onChange={setSyncedDayFilter}
                options={dayFilters}
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

      {isSectionLoading("time-use", dayFilter) && <div className="state">Building time-use comparison…</div>}
      {!isSectionLoading("time-use", dayFilter) && timeUseGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Time-use day type filter"
                onChange={setSyncedDayFilter}
                options={dayFilters}
                value={dayFilter}
              />
            }
            title="Time-use comparison"
          />
          <div className="chart-grid">
            <ChartCard
              title="Mean daily minutes"
              option={timeUseGroupedOption(timeUseGroup.block)}
              wide
            />
            <ChartCard
              title="Synthetic difference from time-use"
              option={timeUseDifferenceOption(timeUseGroup.block)}
              wide
            />
          </div>
        </>
      )}

      {isSectionLoading("motifs", dayFilter) && <div className="state">Building motifs…</div>}
      {!isSectionLoading("motifs", dayFilter) && motifGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="Motifs day type filter"
                onChange={setSyncedDayFilter}
                options={dayFilters}
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

      {isSectionLoading("stvd", dayFilter) && <div className="state">Building STVD map…</div>}
      {!isSectionLoading("stvd", dayFilter) && stvdGroup && (
        <>
          <SectionHeading
            controls={
              <SegmentedControl
                label="STVD day type filter"
                onChange={setSyncedDayFilter}
                options={dayFilters}
                value={dayFilter}
              />
            }
            title="Spatial-temporal volume difference"
          />
          <StvdMap block={stvdGroup.block} />
        </>
      )}

      {homeWork && (
        <>
          <SectionHeading
            controls={
              homeWork.has_profiles ? (
                <DemographicFilter
                  options={homeWork.filter_options}
                  value={demoFilter}
                  onChange={setDemoFilter}
                />
              ) : undefined
            }
            title="Home locations"
          />
          {homeWork.has_profiles && (
            <p className="hw-match-note">
              {homeWork.matched_agents} of {homeWork.total_synthetic_agents} synthetic agents match
              &nbsp;&middot;&nbsp; real population shown unfiltered (no demographics available)
            </p>
          )}
          <HomeWorkMap
            block={homeWork.home}
            syntheticLabel={payload.labels.synthetic}
            observedLabel={payload.labels.observed}
          />

          <SectionHeading title="Work locations" />
          <HomeWorkMap
            block={homeWork.work}
            syntheticLabel={payload.labels.synthetic}
            observedLabel={payload.labels.observed}
          />
        </>
      )}

      <SectionHeading title="Social network" />
      {networkValidation?.network_validation ? (
        <>
          <NetworkValidationSection
            block={networkValidation.network_validation.synthetic_vs_random}
            sourceLabel="synthetic"
            sourceTitle="Synthetic social + encounters"
          />
          <NetworkValidationSection
            block={networkValidation.network_validation.observed_vs_random}
            sourceLabel="observed"
            sourceTitle="Observed daily co-presence"
          />
        </>
      ) : payload.social_network ? (
        <SocialNetworkGraph block={payload.social_network} title="Initial social graph" />
      ) : networkValidationError ? (
        <div className="state">Failed to load network validation: {networkValidationError}</div>
      ) : networkValidation === null && !demoSafeMode ? (
        <div className="state">Building social network validation… (fetched separately from the rest of the charts)</div>
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

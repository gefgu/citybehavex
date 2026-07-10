import { useCallback, useEffect, useRef, useState } from "react";
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
  type LawBlock,
  type NetworkValidationComparisonBlock,
  type NetworkValidationMetricComparisonBlock,
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
  transportShareOption,
  transitionOption,
} from "../charts/builders";

function ChartCard({
  title,
  option,
  wide = false,
  subtitle,
  helpKey,
}: {
  title: string;
  option: EChartsOption;
  wide?: boolean;
  subtitle?: ReactNode;
  helpKey?: string;
}) {
  return (
    <div className={`chart-card${wide ? " wide" : ""}`}>
      <h4><TitleWithHelp label={title} helpKey={helpKey} /></h4>
      <EChart option={option} />
      {subtitle}
    </div>
  );
}

const CALIBRATION_HELP: Record<string, string> = {
  "Wasserstein distances": "Tune the metric-specific generator knobs, then rerun. Wasserstein falls when the synthetic distribution shifts toward the observed distribution.",
  "Jensen-Shannon divergences": "Tune activity, schedule, and motif settings or retrain aligners. JSD falls when the categorical profiles assign similar probability mass.",
  "Common Part of Commuters": "Improve home/work placement with profiles.home_* and profiles.work_distance_*. CPC rises when synthetic commute flows overlap observed OD flows.",
  "Time-use metrics": "Tune activities.durations.*, activities.kappa, and activities.temperature. Time-use error falls when synthetic daily minutes match survey shares.",
  "STVD distances": "Tune home/work placement, gravity, and schedule timing. STVD falls when volume appears in the same places at the same times.",
  "Spatial-temporal volume difference": "Tune spatial placement and schedule timing. The map improves when volume errors shrink in both H3 cells and time bins.",
  "Jump lengths": "Adjust simulation.gravity_deterrence_exponent, simulation.rho, and transport routing. These control destination distance and exploration pressure.",
  "Jump length": "Adjust simulation.gravity_deterrence_exponent, simulation.rho, and transport routing. These control destination distance and exploration pressure.",
  "jump_length": "Adjust simulation.gravity_deterrence_exponent, simulation.rho, and transport routing. These control destination distance and exploration pressure.",
  "Visits per user": "Tune diaries.location_count_*, schedule.alpha_beta_*, and simulation.gamma. These change how many distinct stops agents accumulate.",
  "Radius of gyration": "Tune simulation.gravity_deterrence_exponent, profiles.work_distance_*, and simulation.rho. These change each agent's activity-space spread.",
  "radius_of_gyration": "Tune simulation.gravity_deterrence_exponent, profiles.work_distance_*, and simulation.rho. These change each agent's activity-space spread.",
  "Dwell time": "Tune activities.durations.* and simulation.granularity_minutes. Dwell-time fit improves when stop durations match the observed stay process.",
  "Trip duration (car)": "Tune car_speed_kmh, road_network routing, and max_leg_waypoints. Trip duration tracks route distance and speed assumptions.",
  "Trip duration": "Tune car_speed_kmh, road_network routing, and max_leg_waypoints. Trip duration tracks route distance and speed assumptions.",
  "Activity distribution": "Tune activities.kappa, activities.temperature, and the activity aligner. They control which micro-activities are selected inside diary blocks.",
  "Daily activity profile": "Tune schedule.temperature_beta_* and activity durations. The profile improves when activity timing and duration match by time of day.",
  "Daily motifs": "Tune diaries.motif_exploration_rate, diaries.location_count_*, and schedule.alpha_beta_*. Motifs improve when daily stop patterns have matching complexity.",
  "STVD-EMD": "Tune spatial placement, gravity deterrence, and schedule timing. EMD falls when spatial-temporal volume moves closer to observed cells.",
  "Mean absolute time-use share difference": "Tune activities.durations.* and activity alignment. The error falls when each activity category consumes the right share of the day.",
  "Travel-distance mobility law": "Tune gravity_deterrence_exponent, simulation.rho, and road-network distance. The fitted tail follows the generated trip-distance process.",
  "Radius-of-gyration mobility law": "Tune work-distance placement, gravity deterrence, and exploration. These determine each agent's spatial range.",
  "Daily visited locations": "Tune diaries.location_count_mu, diaries.location_count_sigma, max_locations, and schedule exploration. These set daily stop-count shape.",
  "Distance-frequency visitation law": "Tune simulation.gamma and schedule.alpha_beta_*. Preferential return controls how often agents revisit near/frequent places.",
  "Trips by transport mode": "Tune vehicle ownership, walking/bike thresholds, and road/rail availability. Mode share changes when eligibility and thresholds change.",
  "Trips": "Tune vehicle ownership, walking/bike thresholds, and road/rail availability. Trip counts by mode follow mode eligibility and routing availability.",
  "Share": "Tune ownership, thresholds, and road/rail availability. Share improves when mode choice proportions match observed behavior.",
  "Mean jump": "Tune gravity deterrence and mode thresholds. Mean jump improves when each transport mode covers realistic distance ranges.",
  "Mean duration": "Tune speeds and routing. Mean duration follows path distance, waypoints, and per-mode speed assumptions.",
  "Jump length by transport mode": "Tune transport thresholds and routing speeds. Mode-specific distance improves when mode assignment matches trip-length regimes.",
  "Visit purpose comparison": "Tune diary prompts, schedule aligner, and activity purpose mapping. Purpose share follows macro-diary composition.",
  "Activity transition difference": "Tune schedule aligner and diary pools. Transition gaps fall when consecutive macro-activities resemble observed sequences.",
  "Activity transitions": "Tune schedule aligner and diary pools. Transition gaps fall when consecutive macro-activities resemble observed sequences.",
  "Daily activity difference": "Tune activity durations and activity aligner temperature. Time-bin differences fall when activity timing is better calibrated.",
  "Daily activity": "Tune activity durations and activity aligner temperature. Time-bin differences fall when activity timing is better calibrated.",
  "Mean daily usage over the day": "Tune activities.durations.*, activities.kappa, and the activity aligner. Usage curves improve when micro-activities occupy the right times.",
  "Mean daily minutes": "Tune duration overrides and diary prompts. Category minutes improve when generated routines allocate the right daily time budget.",
  "Synthetic difference from time-use": "Tune the categories with the largest bars first through activities.durations.* or diary prompts. The chart is signed error by category.",
  "Motif literature comparison": "Tune motif_exploration_rate, location-count distribution, and schedule exploration. Motif shares reflect daily sequence complexity.",
  "Home locations": "Tune profiles.home_building_weight, profiles.home_poi_inverse_weight, and home anchors. Home maps improve when residences land in realistic cells.",
  "Work locations": "Tune profiles.work_distance_*, work_poi_weight, and work_building_weight. Work maps improve when employment anchors and commute distances match.",
  "Social network": "Tune social degree, similarity, and dynamic friendship settings. Network metrics improve when durable ties and repeated encounters match baselines.",
  "Degree": "Tune social.degree_*, social.social_graph_k, and max_dynamic_degree. Degree tracks how many durable and encounter ties each agent has.",
  "Clustering coefficient": "Tune social.similarity_temperature, H3 resolutions, and dynamic friendship thresholds. Clustering rises when friends share local neighborhoods.",
  "Edge persistence": "Tune encounter_window_hours, regularity_threshold, and schedule repeatability. Persistence improves when repeated co-presence is calibrated.",
  "Topological overlap": "Tune social.topological_overlap_threshold and profile similarity. Overlap improves when connected agents share neighbor sets realistically.",
  "Synthetic vs observed Wasserstein": "Tune the social config against observed network summaries. Lower values mean synthetic network metric distributions moved closer to observed.",
  "Synthetic vs random Wasserstein": "Tune social tie formation if the synthetic graph is too random-like or too structured. This compares against a degree-preserving baseline.",
  "Observed vs random Wasserstein": "Use comparison.network_validation.* to make observed co-presence construction meaningful. Bad location/time grouping can distort this baseline.",
  "Intermittency vs degree of return": "Tune simulation.gamma, schedule.alpha_beta_*, and diary diversity. These profile axes separate routine from exploratory mobility.",
  "Intermittency": "Tune schedule.alpha_beta_* and simulation.gamma. Intermittency changes when agents alternate between routine and exploratory days.",
  "intermittency": "Tune schedule.alpha_beta_* and simulation.gamma. Intermittency changes when agents alternate between routine and exploratory days.",
  "Degree of return": "Tune simulation.gamma and schedule reuse. Degree of return rises when agents repeatedly revisit the same locations.",
  "degree_of_return": "Tune simulation.gamma and schedule reuse. Degree of return rises when agents repeatedly revisit the same locations.",
  "Visits": "Tune diaries.location_count_* and schedule exploration. Visit counts reflect how many stops each agent-day generates.",
  "visits": "Tune diaries.location_count_* and schedule exploration. Visit counts reflect how many stops each agent-day generates.",
  "Mobility profiles": "Tune exploration, return, and schedule reuse. Profile charts improve when routine and exploratory behavior match observed agents.",
};

function calibrationHelpFor(label?: string) {
  if (!label) return null;
  const direct = CALIBRATION_HELP[label];
  if (direct) return direct;
  const withoutEcdf = label.replace(/\s+ECDF$/i, "");
  return CALIBRATION_HELP[withoutEcdf] ?? null;
}

function HelpIcon({ help }: { help: string }) {
  return (
    <span className="metric-help">
      <button aria-label={`Calibration tip: ${help}`} className="metric-help-button" type="button">
        ?
      </button>
      <span className="metric-help-tooltip" role="tooltip">
        {help}
      </span>
    </span>
  );
}

function TitleWithHelp({
  label,
  helpKey,
}: {
  label: ReactNode;
  helpKey?: string;
}) {
  const text = typeof label === "string" ? label : helpKey;
  const help = calibrationHelpFor(helpKey ?? text);
  return (
    <span className="title-with-help">
      <span>{label}</span>
      {help && <HelpIcon help={help} />}
    </span>
  );
}

const PARAM_SYMBOLS: Record<string, string> = {
  beta: "β",
  c: "c",
  eta: "η",
  kappa: "κ",
  mu: "μ",
  r0: "r₀",
  sigma: "σ",
};

function mobilityParamSymbol(key: string, block: LawBlock) {
  if (key === "r0" && block.x_label === "travel distance") return "Δr₀";
  if (key === "r0" && block.x_label === "radius of gyration") return "rᵍ₀";
  return PARAM_SYMBOLS[key] ?? key;
}

function displayLawLabel(label: string) {
  return label.replace(/\bGonzalez\b/g, "González");
}

function formatParamValue(value: number, unit?: string) {
  const formatted = Number(value).toPrecision(3);
  return unit ? `${formatted} ${unit}` : formatted;
}

function mobilityParamUnit(key: string, block: LawBlock) {
  if (key === "r0" || key === "kappa") return block.x_unit || undefined;
  return undefined;
}

function referenceParams(block: LawBlock): { label: string; params: Record<string, string> } | null {
  if (block.title === "Daily visited locations") {
    return { label: "Log-normal reference", params: { mu: "1.00", sigma: "0.5" } };
  }
  if (block.x_label === "r · f") {
    return { label: "Distance-frequency reference", params: { eta: "2.0", mu: "dataset-dependent" } };
  }
  if (block.x_label === "travel distance") {
    return {
      label: "González reference",
      params: { beta: "1.75", r0: "1.5 km", kappa: "400 km" },
    };
  }
  if (block.x_label === "radius of gyration") {
    return {
      label: "González reference",
      params: { r0: "5.8 km", beta: "1.65", kappa: "350 km" },
    };
  }
  return null;
}

function FitParams({ block }: { block: LawBlock }) {
  const reference = referenceParams(block);
  return (
    <div className="fit-params">
      {block.fits.map((fit) => (
        <div className="fit-param-row" key={fit.label}>
          <span className="fit-param-label">{displayLawLabel(fit.label)}</span>
          <span>
            {Object.entries(fit.params).map(([key, value], index) => (
              <span key={key}>
                {index > 0 && ", "}
                {mobilityParamSymbol(key, block)}={formatParamValue(value, mobilityParamUnit(key, block))}
              </span>
            ))}
          </span>
        </div>
      ))}
      {reference && (
        <div className="fit-param-row reference">
          <span className="fit-param-label">{reference.label}</span>
          <span>
            {Object.entries(reference.params).map(([key, value], index) => (
              <span key={key}>
                {index > 0 && ", "}
                {mobilityParamSymbol(key, block)}={value}
              </span>
            ))}
          </span>
        </div>
      )}
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
  "transport-spatial",
  "activity",
  "mobility-laws",
  "micro-activity",
  "time-use",
  "motifs",
  "stvd",
];
const STATIC_SECTIONS = ["profiles", "social-network"];
const SECTION_REQUEST_TIMEOUT_MS = 10 * 60_000;

function sectionKey(section: string, filter = "all") {
  return `${section}:${filter}`;
}

function defaultSectionRequests(
  dayFilter: string,
  distributionFilter: string,
  fastOnly = false,
): [string, string][] {
  if (fastOnly) {
    const metricFilter = distributionFilter === "all" ? dayFilter : distributionFilter;
    return [
      ["time-use", dayFilter],
      ["metrics", metricFilter],
      ["distributions", distributionFilter],
      ["activity", dayFilter],
      ["motifs", dayFilter],
      ["stvd", dayFilter],
      ["profiles", "all"],
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
    ["transport-spatial", "all"],
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
    transport_spatial: incoming.transport_spatial ?? current.transport_spatial,
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
  description,
}: {
  title: string;
  controls?: ReactNode;
  description?: string;
}) {
  return (
    <div className="section-heading-row">
      <div>
        <div className="section-header"><TitleWithHelp label={title} /></div>
        {description && <p className="section-description">{description}</p>}
      </div>
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
      <h4><TitleWithHelp label={title} /></h4>
      <table className="metrics">
        <tbody>
          {rows.map((m, i) => {
            const label = m.resolution ? `H3 ${m.resolution}` : metricName(m);
            return (
              <tr key={`${m.filter_key ?? "all"}:${metricName(m)}:${m.resolution ?? i}`}>
                <td className="metric-name-cell">
                  <span className="metric-filter">{m.filter_label ?? "All"}</span>
                  <span className="metric-name-text">
                    {label}
                  </span>
                </td>
                <td className="value">{m.value.toFixed(4)}</td>
                <td className="unit">{m.unit ?? unit ?? ""}</td>
              </tr>
            );
          })}
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
  const title = `${sourceLabel === "synthetic" ? "Synthetic" : "Observed"} vs random Wasserstein`;
  return (
    <div>
      <h4><TitleWithHelp label={title} /></h4>
      <table className="metrics">
        <tbody>
          {Object.entries(NETWORK_VALIDATION_TITLES).map(([key, label]) => {
            const value = validation.wasserstein[key as keyof typeof validation.wasserstein];
            const source = validation.distributions[sourceLabel]?.[key];
            const rnd = validation.distributions.random[key];
            return (
              <tr key={key}>
                <td>
                  <TitleWithHelp label={label} />
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

function NetworkObservedComparisonTable({
  validation,
}: {
  validation: NetworkValidationMetricComparisonBlock | undefined;
}) {
  if (!validation) return null;
  return (
    <div>
      <h4><TitleWithHelp label="Synthetic vs observed Wasserstein" /></h4>
      <table className="metrics">
        <tbody>
          {Object.entries(NETWORK_VALIDATION_TITLES).map(([key, label]) => {
            const value = validation.wasserstein[key as keyof typeof validation.wasserstein];
            const synthetic = validation.distributions.synthetic[key];
            const observed = validation.distributions.observed[key];
            return (
              <tr key={key}>
                <td>
                  <TitleWithHelp label={label} />
                  <span className="metric-filter">
                    synthetic n={synthetic?.count ?? 0} · observed n={observed?.count ?? 0}
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
  showWasserstein = true,
}: {
  block: NetworkValidationComparisonBlock | undefined;
  sourceLabel: "synthetic" | "observed";
  sourceTitle: string;
  showWasserstein?: boolean;
}) {
  if (!block) return null;
  return (
    <>
      {showWasserstein && (
        <div className="metric-tables">
          <NetworkValidationTable validation={block} sourceLabel={sourceLabel} />
        </div>
      )}
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
                ? "Timed out after 10 minutes while building this section; reload to retry."
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
  const hasMetricRows = Object.values(metricRows).some((rows) => rows.length > 0);
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
  const metricsError = sectionError("metrics", metricSectionFilter);
  const distributionFilterLoading = isSectionLoading("distributions", distributionFilter);
  const mobilityLawsLoading = isSectionLoading("mobility-laws", dayFilter);
  const activityLoading = isSectionLoading("activity", dayFilter);
  const microActivityLoading = isSectionLoading("micro-activity", dayFilter);
  const timeUseLoading = isSectionLoading("time-use", dayFilter);
  const motifsLoading = isSectionLoading("motifs", dayFilter);
  const stvdLoading = isSectionLoading("stvd", dayFilter);
  const profilesLoading = isSectionLoading("profiles");
  const socialNetworkLoading = isSectionLoading("social-network");
  const networkValidationBlock = networkValidation?.network_validation;
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
              aria-label="Export chart metrics as JSON"
              className="btn btn-secondary btn-compact"
              disabled={exportingMetrics}
              onClick={() => void handleMetricsExport()}
              title="Download the current run's validation metrics as a JSON file"
              type="button"
            >
              {exportingMetrics ? "Exporting..." : "Export JSON"}
            </button>
          </>
        }
        description="Summary distances between synthetic and observed behavior; smaller Wasserstein, Jensen-Shannon, time-use, STVD, and CPC gaps indicate closer agreement."
        title="Metrics"
      />
      {payload.mode === "synthetic_only" ? (
        <div className="network-empty">
          Synthetic-only mode. Add an observed comparison parquet to show Wasserstein,
          Jensen-Shannon and CPC metrics.
        </div>
      ) : metricsLoading ? (
        <div className="state">Building selected metrics…</div>
      ) : metricsError ? (
        <div className="state">Failed to load metrics: {metricsError}</div>
      ) : !hasMetricRows ? (
        <div className="state">No metrics returned for {metricSectionFilter}.</div>
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
        description="ECDFs show the share of people or visits below each value, comparing trip lengths, visit counts, mobility radius, dwell time, and trip duration."
        title="Distribution comparisons"
      />
      {distributionFilterLoading && (
        <div className="state">Building {distributionFilter} distribution…</div>
      )}
      {sectionError("distributions", distributionFilter) && (
        <div className="state">Failed to load distribution: {sectionError("distributions", distributionFilter)}</div>
      )}
      {!distributionFilterLoading && !sectionError("distributions", distributionFilter) && !ecdfGroup && (
        <div className="state">No distribution comparison returned for {distributionFilter}.</div>
      )}
      {!distributionFilterLoading && ecdfGroup && (
        <div className="chart-grid">
          {Object.entries(ecdfGroup.blocks).map(([key, block]) => (
            <ChartCard
              key={key}
              title={`${ECDF_TITLES[key] ?? key} ECDF`}
              helpKey={ECDF_TITLES[key] ?? key}
              option={ecdfOption(block)}
            />
          ))}
        </div>
      )}

      {isSectionLoading("transport-spatial") && <div className="state">Building transport mobility…</div>}
      {!isSectionLoading("transport-spatial") && payload.transport_spatial && (
        <>
          <SectionHeading
            description="Breaks trips down by inferred transport mode and compares the distance distribution within each mode."
            title="Transport mobility"
          />
          <div className="chart-grid">
            <ChartCard
              title="Trips by transport mode"
              helpKey="Trips by transport mode"
              option={transportShareOption(payload.transport_spatial.share)}
            />
            <ChartCard
              title="Jump length by transport mode"
              helpKey="Jump length by transport mode"
              option={ecdfOption(payload.transport_spatial.jump_ecdf)}
            />
          </div>
          <div className="metric-tables">
            {Object.entries(payload.transport_spatial.summary).map(([source, block]) => (
              <table className="metrics" key={source}>
                <caption>{source === "observed" ? payload.labels.observed ?? "observed" : "synthetic"}</caption>
                <thead>
                  <tr>
                    <th>Mode</th>
                    <th><TitleWithHelp label="Trips" /></th>
                    <th><TitleWithHelp label="Share" /></th>
                    <th><TitleWithHelp label="Mean jump" /></th>
                    <th><TitleWithHelp label="Mean duration" /></th>
                  </tr>
                </thead>
                <tbody>
                  {block.modes.map((row) => (
                    <tr key={row.mode}>
                      <td>{row.mode}</td>
                      <td>{row.count}</td>
                      <td>{row.percent.toFixed(2)}%</td>
                      <td>{row.mean_jump_km == null ? "n/a" : `${row.mean_jump_km.toFixed(3)} km`}</td>
                      <td>{row.mean_duration_min == null ? "n/a" : `${row.mean_duration_min.toFixed(1)} min`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ))}
          </div>
        </>
      )}
      {sectionError("transport-spatial") && (
        <div className="state">Failed to load transport mobility: {sectionError("transport-spatial")}</div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Mobility laws day type filter"
            onChange={setSyncedDayFilter}
            options={dayFilters}
            value={dayFilter}
          />
        }
        description="Power-law and log-normal fits test whether travel distances, radius of gyration, daily locations, and visitation frequency follow known mobility laws."
        title="Mobility laws"
      />
      {mobilityLawsLoading ? (
        <div className="state">Building mobility laws…</div>
      ) : sectionError("mobility-laws", dayFilter) ? (
        <div className="state">Failed to load mobility laws: {sectionError("mobility-laws", dayFilter)}</div>
      ) : mobilityGroup ? (
          <div className="chart-grid">
            {Object.entries(mobilityGroup.blocks).map(([key, block]) => (
              <ChartCard
                key={key}
                title={block.title}
                helpKey={block.title}
                option={lawOption(block)}
                subtitle={<FitParams block={block} />}
              />
            ))}
          </div>
      ) : (
        <div className="state">No mobility laws returned for {dayFilter}.</div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Activity day type filter"
            onChange={setSyncedDayFilter}
            options={dayFilters}
            value={dayFilter}
          />
        }
        description="Compares visit purposes, transitions between activities, and time-of-day activity intensity between synthetic and observed traces."
        title="Activity comparison"
      />
      {activityLoading ? (
        <div className="state">Building activity comparison…</div>
      ) : sectionError("activity", dayFilter) ? (
        <div className="state">Failed to load activity comparison: {sectionError("activity", dayFilter)}</div>
      ) : activityGroup ? (
          <div className="chart-grid">
            <ChartCard
              title="Visit purpose comparison"
              helpKey="Visit purpose comparison"
              option={purposeOption(activityGroup.purpose)}
              wide
            />
            <ChartCard
              title={activityGroup.transition_difference.matrix_mode === "raw" ? "Activity transitions" : "Activity transition difference"}
              helpKey={activityGroup.transition_difference.matrix_mode === "raw" ? "Activity transitions" : "Activity transition difference"}
              option={transitionOption(activityGroup.transition_difference)}
            />
            {activityGroup.daily_activity_difference && (
              <ChartCard
                title={activityGroup.daily_activity_difference.matrix_mode === "raw" ? "Daily activity" : "Daily activity difference"}
                helpKey={activityGroup.daily_activity_difference.matrix_mode === "raw" ? "Daily activity" : "Daily activity difference"}
                option={dailyActivityOption(activityGroup.daily_activity_difference)}
              />
            )}
          </div>
      ) : (
        <div className="state">No activity comparison returned for {dayFilter}.</div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Micro-activity day type filter"
            onChange={setSyncedDayFilter}
            options={dayFilters}
            value={dayFilter}
          />
        }
        description="Shows how synthetic fine-grained activity labels are distributed through the day, useful for checking generated diary rhythms."
        title="Synthetic micro-activity usage"
      />
      {microActivityLoading ? (
        <div className="state">Building micro-activity usage…</div>
      ) : sectionError("micro-activity", dayFilter) ? (
        <div className="state">Failed to load micro-activity usage: {sectionError("micro-activity", dayFilter)}</div>
      ) : microActivityGroup ? (
          <ChartCard
            title="Mean daily usage over the day"
            helpKey="Mean daily usage over the day"
            option={microActivityUsageOption(microActivityGroup.block)}
            wide
          />
      ) : (
        <div className="state">No micro-activity usage returned for {dayFilter}.</div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Time-use day type filter"
            onChange={setSyncedDayFilter}
            options={dayFilters}
            value={dayFilter}
          />
        }
        description="Compares minutes per day in broad activity categories against time-use survey targets and shows synthetic-minus-reference deviations."
        title="Time-use comparison"
      />
      {timeUseLoading ? (
        <div className="state">Building time-use comparison…</div>
      ) : sectionError("time-use", dayFilter) ? (
        <div className="state">Failed to load time-use comparison: {sectionError("time-use", dayFilter)}</div>
      ) : timeUseGroup ? (
          <div className="chart-grid">
            <ChartCard
              title="Mean daily minutes"
              helpKey="Mean daily minutes"
              option={timeUseGroupedOption(timeUseGroup.block)}
              wide
            />
            <ChartCard
              title="Synthetic difference from time-use"
              helpKey="Synthetic difference from time-use"
              option={timeUseDifferenceOption(timeUseGroup.block)}
              wide
            />
          </div>
      ) : (
        <div className="state">No time-use comparison returned for {dayFilter}. Configure a time-use survey to enable this section.</div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="Motifs day type filter"
            onChange={setSyncedDayFilter}
            options={dayFilters}
            value={dayFilter}
          />
        }
        description="Daily motifs summarize each agent-day as a compact sequence of visited activity types, then compare motif frequencies with literature patterns."
        title="Daily motifs"
      />
      {motifsLoading ? (
        <div className="state">Building motifs…</div>
      ) : sectionError("motifs", dayFilter) ? (
        <div className="state">Failed to load motifs: {sectionError("motifs", dayFilter)}</div>
      ) : motifGroup ? (
          <div className="chart-grid">
            <ChartCard
              title="Motif literature comparison"
              helpKey="Motif literature comparison"
              option={motifOption(motifGroup.block)}
              wide
            />
          </div>
      ) : (
        <div className="state">No motifs returned for {dayFilter}.</div>
      )}

      <SectionHeading
        controls={
          <SegmentedControl
            label="STVD day type filter"
            onChange={setSyncedDayFilter}
            options={dayFilters}
            value={dayFilter}
          />
        }
        description="STVD maps where and when synthetic traffic volume differs from observed volume; values are spatial-temporal cell differences."
        title="Spatial-temporal volume difference"
      />
      {stvdLoading ? (
        <div className="state">Building STVD map…</div>
      ) : sectionError("stvd", dayFilter) ? (
        <div className="state">Failed to load STVD map: {sectionError("stvd", dayFilter)}</div>
      ) : stvdGroup ? (
          <StvdMap block={stvdGroup.block} />
      ) : (
        <div className="state">No STVD map returned for {dayFilter}.</div>
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
            description="Maps estimated home locations by H3 cell so residential spatial density can be compared across synthetic and observed populations."
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

          <SectionHeading
            description="Maps estimated work locations by H3 cell, highlighting whether simulated commuters concentrate in the same employment areas."
            title="Work locations"
          />
          <HomeWorkMap
            block={homeWork.work}
            syntheticLabel={payload.labels.synthetic}
            observedLabel={payload.labels.observed}
          />
        </>
      )}

      <SectionHeading
        description="Validates encounter and social-network structure using degree, clustering, component, and path metrics against random or observed baselines."
        title="Social network"
      />
      {networkValidationBlock ? (
        <>
          {networkValidationBlock.synthetic_vs_observed && (
            <div className="metric-tables">
              <NetworkObservedComparisonTable
                validation={networkValidationBlock.synthetic_vs_observed}
              />
            </div>
          )}
          <NetworkValidationSection
            block={networkValidationBlock.synthetic_vs_random}
            sourceLabel="synthetic"
            sourceTitle="Synthetic social + encounters"
          />
          <NetworkValidationSection
            block={networkValidationBlock.observed_vs_random}
            sourceLabel="observed"
            sourceTitle="Observed daily co-presence"
          />
        </>
      ) : payload.social_network ? (
        <SocialNetworkGraph block={payload.social_network} title="Initial social graph" />
      ) : networkValidationError ? (
        <div className="state">Failed to load network validation: {networkValidationError}</div>
      ) : sectionError("social-network") ? (
        <div className="state">Failed to load social network: {sectionError("social-network")}</div>
      ) : socialNetworkLoading ? (
        <div className="state">Building social network…</div>
      ) : networkValidation === null && !demoSafeMode ? (
        <div className="state">Building social network validation… (fetched separately from the rest of the charts)</div>
      ) : (
        <div className="network-empty">
          No social network sidecar found for this run. Re-run the simulation with the latest code,
          then refresh the chart payload.
        </div>
      )}

      <SectionHeading
        description="Classifies mobility regularity with degree of return, intermittency, and related profile metrics that separate routine from exploratory behavior."
        title="Mobility profiles"
      />
      {profilesLoading ? (
        <div className="state">Building mobility profiles…</div>
      ) : sectionError("profiles") ? (
        <div className="state">Failed to load mobility profiles: {sectionError("profiles")}</div>
      ) : payload.profiles ? (
          <div className="chart-grid">
            <ChartCard
              title="Intermittency vs degree of return"
              helpKey="Intermittency vs degree of return"
              option={profileScatterOption(payload.profiles)}
              wide
            />
            {payload.profiles.metrics.map((metric) => (
              <ChartCard
                key={metric}
                title={metric[0].toUpperCase() + metric.slice(1)}
                helpKey={metric}
                option={profileBoxOption(payload.profiles!, metric)}
              />
            ))}
          </div>
      ) : (
        <div className="state">No mobility profiles returned for this run.</div>
      )}

      <div style={{ height: 96 }} />
    </>
  );
}

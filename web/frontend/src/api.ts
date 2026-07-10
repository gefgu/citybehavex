// Thin fetch wrapper. Every backend response is `{ data: ... }`; we return `data`.
// In static-demo builds, selected API calls are resolved from JSON files emitted
// by scripts/export_static_web_demo.py instead of a live FastAPI backend.

const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === "true";
const STATIC_ROOT = `${import.meta.env.BASE_URL.replace(/\/?$/, "/")}demo-data`;

async function getStaticJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${STATIC_ROOT}/${path.replace(/^\/+/, "")}`, init);
  return readJson<T>(res);
}

async function getStaticRawJson<T>(path: string): Promise<T> {
  const res = await fetch(`${STATIC_ROOT}/${path.replace(/^\/+/, "")}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as T;
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  return readJson<T>(res);
}

async function sendJson<T>(url: string, init: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  return readJson<T>(res);
}

async function readJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  const body = (await res.json()) as { data: T };
  return body.data;
}

export interface Run {
  run_id: string;
  path: string;
  mtime: number;
  summary?: {
    rows: number;
    uids?: number;
    date_start?: string;
    date_end?: string;
  };
  summary_error?: string;
}

export interface Experiment {
  id: string;
  config: string;
  label: string;
  simulation_output: string | null;
  observed_path: string | null;
  observed_exists: boolean;
  time_use_path: string | null;
  time_use_exists: boolean;
  time_use_label: string;
  time_use_country: string | null;
  time_use_survey: number | null;
  time_use_weight_col: string;
  profiles_enabled: boolean;
  profiles_output: string | null;
  profiles_path: string | null;
  profiles_exists: boolean;
  params: Record<string, unknown>;
  runs: Run[];
}

let staticExperimentsCache: Promise<Experiment[]> | null = null;

async function staticExperiments(): Promise<Experiment[]> {
  if (!staticExperimentsCache) {
    staticExperimentsCache = getStaticJson<Experiment[]>("experiments.json");
  }
  return staticExperimentsCache;
}

async function staticRunPath(id: string): Promise<string> {
  const experiments = await staticExperiments();
  const experiment = experiments.find((item) => item.id === id);
  const run = experiment?.runs?.[0]?.run_id;
  if (!run) throw new Error(`Static demo run not found for ${id}`);
  return `${id}/${run}`;
}

export interface ExperimentUpdate {
  label?: string;
  agents?: number;
  days?: number;
  start_date?: string | null;
  granularity_minutes?: number;
  car_speed_kmh?: number;
  simulation_output?: string;
  observed_path?: string | null;
  time_use_path?: string | null;
  time_use_label?: string;
  time_use_country?: string | null;
  time_use_survey?: number | null;
  time_use_weight_col?: string;
  profiles_enabled?: boolean;
  profiles_output?: string;
}

export function fetchExperiments(withSummary = false): Promise<Experiment[]> {
  if (STATIC_DEMO) {
    void withSummary;
    return getStaticJson<Experiment[]>("experiments.json");
  }
  return getJson<Experiment[]>(`/api/experiments?with_summary=${withSummary}`);
}

export function fetchExperiment(id: string): Promise<Experiment> {
  if (STATIC_DEMO) {
    return getStaticJson<Experiment>(`experiments/${encodeURIComponent(id)}.json`);
  }
  return getJson<Experiment>(`/api/experiments/${encodeURIComponent(id)}`);
}

export function updateExperiment(id: string, payload: ExperimentUpdate): Promise<Experiment> {
  if (STATIC_DEMO) {
    void id;
    void payload;
    return Promise.reject(new Error("Static demo experiments cannot be edited."));
  }
  return sendJson<Experiment>(`/api/experiments/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function archiveExperiment(id: string): Promise<{ archived_config: string }> {
  if (STATIC_DEMO) {
    void id;
    return Promise.reject(new Error("Static demo experiments cannot be archived."));
  }
  return sendJson<{ archived_config: string }>(
    `/api/experiments/${encodeURIComponent(id)}/archive`,
    { method: "POST", body: "{}" },
  );
}

export function deleteExperimentRun(id: string, runId: string): Promise<{ deleted: string[] }> {
  if (STATIC_DEMO) {
    void id;
    void runId;
    return Promise.reject(new Error("Static demo runs cannot be deleted."));
  }
  return sendJson<{ deleted: string[] }>(
    `/api/experiments/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE", body: "{}" },
  );
}

export function fetchCharts(id: string, run?: string): Promise<ChartPayload> {
  if (STATIC_DEMO) {
    if (!run) return staticRunPath(id).then((path) => getStaticJson<ChartPayload>(`${path}/charts/base.json`));
    return getStaticJson<ChartPayload>(`${id}/${run}/charts/base.json`);
  }
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<ChartPayload>(`/api/experiments/${encodeURIComponent(id)}/charts${q}`);
}

export function fetchChartSection(
  id: string,
  section: string,
  filter = "all",
  run?: string,
  signal?: AbortSignal,
): Promise<ChartPayload> {
  if (STATIC_DEMO) {
    const file = `${encodeURIComponent(section)}/${encodeURIComponent(filter)}.json`;
    if (!run) {
      return staticRunPath(id).then((path) =>
        getStaticJson<ChartPayload>(`${path}/charts/sections/${file}`, { signal }),
      );
    }
    return getStaticJson<ChartPayload>(
      `${id}/${run}/charts/sections/${file}`,
      { signal },
    );
  }
  const q = new URLSearchParams();
  q.set("filter", filter);
  if (run) q.set("run", run);
  return getJson<ChartPayload>(
    `/api/experiments/${encodeURIComponent(id)}/charts/${encodeURIComponent(section)}?${q.toString()}`,
    { signal },
  );
}

export async function downloadMetricsExport(id: string, run?: string): Promise<Blob> {
  if (STATIC_DEMO) {
    const path = run ? `${id}/${run}` : await staticRunPath(id);
    const payload = await getStaticRawJson<unknown>(`${path}/metrics-export.json`);
    return new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  }
  const q = new URLSearchParams({ format: "json" });
  if (run) q.set("run", run);
  const res = await fetch(
    `/api/experiments/${encodeURIComponent(id)}/metrics-export?${q.toString()}`,
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  return res.blob();
}

export interface HomeWorkFeatureCollection {
  type: string;
  features: {
    type: string;
    geometry: { type: string; coordinates: number[][][] };
    properties: Record<string, unknown> & {
      area: string;
      agent_count: number;
      agent_pct: number;
      color: string;
      class: number;
    };
  }[];
}
export interface HomeWorkPanel {
  center: [number, number] | null;
  layers: Record<string, HomeWorkFeatureCollection>;
  colors: string[];
  breaks: number[];
  total_agents: number;
}
export interface HomeWorkMapBlock {
  synthetic: HomeWorkPanel;
  real: HomeWorkPanel | null;
}
export interface AgeBracket {
  key: string;
  label: string;
  min: number;
  max: number;
}
export interface DemographicFilter {
  gender: string | null;
  age_bracket: string | null;
  job: string | null;
}
export interface HomeWorkResponse {
  run_id: string;
  mode: "comparison" | "synthetic_only";
  has_profiles: boolean;
  matched_agents: number;
  total_synthetic_agents: number;
  filter: DemographicFilter;
  filter_options: { genders: string[]; age_brackets: AgeBracket[]; jobs: string[] };
  home: HomeWorkMapBlock;
  work: HomeWorkMapBlock;
  warnings: string[];
}

function staticHomeWorkFilterKey(filter?: DemographicFilter): string {
  if (!filter?.gender && !filter?.age_bracket && !filter?.job) return "all";
  return [
    filter.gender ? `gender-${filter.gender}` : "gender-all",
    filter.age_bracket ? `age-${filter.age_bracket}` : "age-all",
    filter.job ? `job-${filter.job}` : "job-all",
  ]
    .map((part) => encodeURIComponent(part))
    .join("__");
}

export function fetchHomeWork(
  id: string,
  run?: string,
  filter?: DemographicFilter,
): Promise<HomeWorkResponse> {
  if (STATIC_DEMO) {
    const filterKey = staticHomeWorkFilterKey(filter);
    const fetchStatic = (path: string) =>
      getStaticJson<HomeWorkResponse>(`${path}/home-work/${filterKey}.json`).catch((error) => {
        if (filterKey === "all") throw error;
        return getStaticJson<HomeWorkResponse>(`${path}/home-work/all.json`);
      });
    if (!run) return staticRunPath(id).then(fetchStatic);
    return fetchStatic(`${id}/${run}`);
  }
  const q = new URLSearchParams();
  if (run) q.set("run", run);
  if (filter?.gender) q.set("gender", filter.gender);
  if (filter?.age_bracket) q.set("age_bracket", filter.age_bracket);
  if (filter?.job) q.set("job", filter.job);
  const qs = q.toString();
  return getJson<HomeWorkResponse>(
    `/api/experiments/${encodeURIComponent(id)}/home-work${qs ? `?${qs}` : ""}`,
  );
}

// ---- timeline view ----
export interface TimelineMeta {
  run_id: string;
  date_start: string | null;
  date_end: string | null;
  bbox: { min_lat: number; max_lat: number; min_lng: number; max_lng: number } | null;
  agents_total: number | null;
  has_profiles: boolean;
  has_encounters: boolean;
  car_speed_kmh: number | null;
}

export interface TimelineWaypoint {
  lat: number;
  lng: number;
  t: string;
}

export interface TimelineSegment {
  uid: number;
  kind: "dwell" | "leg";
  t_start: string;
  t_end: string;
  o_lat: number;
  o_lng: number;
  d_lat: number;
  d_lng: number;
  purpose: string;
  category?: string | null;
  gender?: "female" | "man" | "unknown" | null;
  character_sprite?:
    | "female"
    | "man"
    | "men_2"
    | "men_3"
    | "men_4"
    | "men_5"
    | "men_6"
    | "woman_2"
    | "woman_3"
    | "woman_4"
    | "woman_5"
    | "unknown"
    | null;
  mode?: "stay" | "car" | "walk" | "bike" | "rail";
  // Present on "leg" segments from runs with road routing enabled — the
  // road-following path to animate along instead of a straight-line lerp.
  waypoints?: TimelineWaypoint[] | null;
}

export interface TimelineLegsPayload {
  run_id: string;
  since: string;
  until: string;
  agent_count: number;
  truncated: boolean;
  segments: TimelineSegment[];
}

interface StaticTimelineChunkIndex {
  chunks: { file: string; since: string; until: string }[];
}

function segmentIntersectsTime(segment: TimelineSegment, sinceMs: number, untilMs: number): boolean {
  const start = Date.parse(segment.t_start);
  const end = Date.parse(segment.t_end);
  return end >= sinceMs && start <= untilMs;
}

function segmentIntersectsBbox(
  segment: TimelineSegment,
  [minLat, minLng, maxLat, maxLng]: [number, number, number, number],
): boolean {
  const segMinLat = Math.min(segment.o_lat, segment.d_lat);
  const segMaxLat = Math.max(segment.o_lat, segment.d_lat);
  const segMinLng = Math.min(segment.o_lng, segment.d_lng);
  const segMaxLng = Math.max(segment.o_lng, segment.d_lng);
  return segMaxLat >= minLat && segMinLat <= maxLat && segMaxLng >= minLng && segMinLng <= maxLng;
}

async function fetchStaticTimelineLegs(
  id: string,
  params: {
    run?: string;
    since: string;
    until: string;
    bbox: [number, number, number, number];
    maxAgents?: number;
  },
): Promise<TimelineLegsPayload> {
  const path = params.run ? `${id}/${params.run}` : await staticRunPath(id);
  const index = await getStaticJson<StaticTimelineChunkIndex>(`${path}/timeline/chunks.json`);
  const sinceMs = Date.parse(params.since);
  const untilMs = Date.parse(params.until);
  const chunkRefs = index.chunks.filter(
    (chunk) => Date.parse(chunk.until) >= sinceMs && Date.parse(chunk.since) <= untilMs,
  );
  const chunks = await Promise.all(
    chunkRefs.map((chunk) => getStaticJson<TimelineLegsPayload>(`${path}/timeline/legs/${chunk.file}`)),
  );
  const seenAgents = new Set<number>();
  const maxAgents = params.maxAgents ?? 2000;
  const segments: TimelineSegment[] = [];
  let truncated = false;
  for (const chunk of chunks) {
    truncated = truncated || chunk.truncated;
    for (const segment of chunk.segments) {
      if (!segmentIntersectsTime(segment, sinceMs, untilMs)) continue;
      if (!segmentIntersectsBbox(segment, params.bbox)) continue;
      if (!seenAgents.has(segment.uid) && seenAgents.size >= maxAgents) {
        truncated = true;
        continue;
      }
      seenAgents.add(segment.uid);
      segments.push(segment);
    }
  }
  return {
    run_id: params.run ?? path.split("/").slice(-1)[0] ?? "",
    since: params.since,
    until: params.until,
    agent_count: seenAgents.size,
    truncated,
    segments,
  };
}

export interface AgentProfileFields {
  uid: number;
  gender: string;
  name: string;
  age: number;
  education: string;
  health: number;
  household: string;
  job: string;
  has_car: boolean;
  has_bike: boolean;
  home_tile: number;
  work_tile: number;
}

export interface AgentTripActivity {
  arrival: string;
  departure: string;
  purpose: string;
  category?: string | null;
  activity: number | null;
  activity_name: string | null;
  activity_description: string | null;
  trip_duration_minutes: number;
  dwell_minutes: number;
}

export interface AgentTrip {
  arrival: string;
  departure: string;
  lat: number;
  lng: number;
  purpose: string;
  category?: string | null;
  trip_duration_minutes: number;
  dwell_minutes: number;
  activities: AgentTripActivity[];
}

export interface AgentEncounter {
  contact_uid: number;
  ts: string;
  tile: number;
  stop_arrival: string | null;
  stop_departure: string | null;
  lat: number | null;
  lng: number | null;
  purpose: string | null;
  category?: string | null;
  activity: number | null;
  activity_name: string | null;
  activity_description: string | null;
  trip_duration_minutes: number | null;
  dwell_minutes: number | null;
  contact_profile: AgentProfileFields | null;
  contact_narrative: string | null;
  location_warning: string | null;
}

export interface AgentProfilePayload {
  uid: number;
  run_id: string;
  profile: AgentProfileFields | null;
  narrative: string | null;
  trips: AgentTrip[];
  encounters: AgentEncounter[];
  warnings: string[];
}

export function fetchTimelineMeta(id: string, run?: string): Promise<TimelineMeta> {
  if (STATIC_DEMO) {
    if (!run) return staticRunPath(id).then((path) => getStaticJson<TimelineMeta>(`${path}/timeline/meta.json`));
    return getStaticJson<TimelineMeta>(`${id}/${run}/timeline/meta.json`);
  }
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<TimelineMeta>(`/api/experiments/${encodeURIComponent(id)}/timeline/meta${q}`);
}

export function fetchTimelineLegs(
  id: string,
  params: {
    run?: string;
    since: string;
    until: string;
    bbox: [number, number, number, number]; // [minLat, minLng, maxLat, maxLng]
    maxAgents?: number;
  },
): Promise<TimelineLegsPayload> {
  if (STATIC_DEMO) {
    return fetchStaticTimelineLegs(id, params);
  }
  const [minLat, minLng, maxLat, maxLng] = params.bbox;
  const q = new URLSearchParams({
    since: params.since,
    until: params.until,
    min_lat: String(minLat),
    min_lng: String(minLng),
    max_lat: String(maxLat),
    max_lng: String(maxLng),
  });
  if (params.run) q.set("run", params.run);
  if (params.maxAgents) q.set("max_agents", String(params.maxAgents));
  return getJson<TimelineLegsPayload>(
    `/api/experiments/${encodeURIComponent(id)}/timeline/legs?${q.toString()}`,
  );
}

export function fetchTimelineAgent(id: string, uid: number, run?: string): Promise<AgentProfilePayload> {
  if (STATIC_DEMO) {
    if (!run) {
      return staticRunPath(id).then((path) =>
        getStaticJson<AgentProfilePayload>(`${path}/timeline/agents/${uid}/profile.json`),
      );
    }
    return getStaticJson<AgentProfilePayload>(`${id}/${run}/timeline/agents/${uid}/profile.json`);
  }
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<AgentProfilePayload>(
    `/api/experiments/${encodeURIComponent(id)}/timeline/agents/${uid}${q}`,
  );
}

export interface AgentCrpDiary {
  diary_id: string;
  day_type: string;
  sim: number;
  usage_count: number;
  description?: string;
  episodes?: { start: string; end: string; purpose: string }[];
}

export interface AgentCrpPayload {
  uid: number;
  run_id: string;
  T_a: number | null;
  alpha_a: number | null;
  diaries: AgentCrpDiary[];
  warnings: string[];
}

export function fetchTimelineAgentCrp(id: string, uid: number, run?: string): Promise<AgentCrpPayload> {
  if (STATIC_DEMO) {
    if (!run) {
      return staticRunPath(id).then((path) =>
        getStaticJson<AgentCrpPayload>(`${path}/timeline/agents/${uid}/crp.json`),
      );
    }
    return getStaticJson<AgentCrpPayload>(`${id}/${run}/timeline/agents/${uid}/crp.json`);
  }
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<AgentCrpPayload>(
    `/api/experiments/${encodeURIComponent(id)}/timeline/agents/${uid}/crp${q}`,
  );
}

export interface AgentSocialParameters {
  degree: number;
  total_social_strength: number;
  social_graph_k: number | null;
  layout: string | null;
  kind: string | null;
  directed: boolean | null;
  rho?: number | null;
  gamma?: number | null;
  alpha?: number | null;
  dt_update_mob_sim_hours?: number | null;
  indipendency_window_hours?: number | null;
}

export interface AgentSocialFriend {
  uid: number;
  name: string | null;
  profile: AgentProfileFields | null;
  social_strength: number;
  embedding_similarity: number;
  encounter_count: number;
  reciprocated: boolean;
}

export interface AgentSocialPayload {
  uid: number;
  run_id: string;
  parameters: AgentSocialParameters;
  friends: AgentSocialFriend[];
  warnings: string[];
}

export function fetchTimelineAgentSocial(
  id: string,
  uid: number,
  run?: string,
): Promise<AgentSocialPayload> {
  if (STATIC_DEMO) {
    if (!run) {
      return staticRunPath(id).then((path) =>
        getStaticJson<AgentSocialPayload>(`${path}/timeline/agents/${uid}/social.json`),
      );
    }
    return getStaticJson<AgentSocialPayload>(`${id}/${run}/timeline/agents/${uid}/social.json`);
  }
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<AgentSocialPayload>(
    `/api/experiments/${encodeURIComponent(id)}/timeline/agents/${uid}/social${q}`,
  );
}

// ---- payload types (mirrors web/backend/app/payload.py) ----
export interface SeriesPoints {
  name: string;
  role: string;
  points: number[][];
  type?: string;
}
export interface EcdfBlock {
  x_label: string;
  x_unit: string;
  series: SeriesPoints[];
}
export interface LawBlock {
  title: string;
  x_label: string;
  x_unit: string;
  x_log: boolean;
  formula: string;
  series: SeriesPoints[];
  fits: { label: string; params: Record<string, number> }[];
}
export interface BarSeries {
  name: string;
  role: string;
  values: number[];
}
export interface ActivityBlock {
  purpose: { categories: string[]; series: BarSeries[] };
  transition_difference: {
    categories: string[];
    labels: string[];
    matrix_mode: "difference" | "raw";
    matrix: number[][];
    limit: number;
  };
  daily_activity_difference: {
    categories: string[];
    n_bins: number;
    labels: string[];
    matrix_mode: "difference" | "raw";
    matrix: number[][];
    limit: number;
  } | null;
}
export interface FilteredActivityBlock extends ActivityBlock {
  filter_key: string;
  filter_label: string;
}
export interface FilteredBlockMap<T> {
  filter_key: string;
  filter_label: string;
  blocks: Record<string, T>;
}
export interface FilteredSingleBlock<T> {
  filter_key: string;
  filter_label: string;
  block: T;
}
export interface MicroActivityUsageBlock {
  bin_size_minutes: number;
  n_bins: number;
  x: string[];
  series: { activity_id: number; name: string; values: number[] }[];
}
export interface TimeUseComparisonRow {
  category: string;
  mtus_minutes: number;
  simulation_minutes: number;
  observed_minutes: number;
  synthetic_minutes: number;
  difference_minutes: number;
  percent_difference: number | null;
  share_of_day_difference_pct_points: number;
}
export interface TimeUseComparisonBlock {
  categories: string[];
  labels: string[];
  rows: TimeUseComparisonRow[];
}
export interface ProfilesBlock {
  scatter: { name: string; points: { x: number; y: number; profile: string }[] }[];
  profile_order: string[];
  metrics: string[];
  datasets: string[];
  box: Record<string, Record<string, Record<string, number[] | null>>>;
}
export interface MotifsBlock {
  categories: string[];
  series: (BarSeries & { rows?: MotifRow[] })[];
  motif_label_keys?: Record<string, string>;
  motif_label_styles?: Record<string, unknown>;
}
export interface MotifRow {
  literature_motif_id: number | string;
  motif_id: number | string;
  hex_id: string;
  percentage: number;
  count: number;
}
export interface StvdFeatureCollection {
  type: string;
  features: {
    type: string;
    geometry: { type: string; coordinates: number[][][] };
    properties: Record<string, unknown> & { color: string; area: string };
  }[];
}
export interface StvdBlock {
  center: [number, number] | null;
  layers: Record<string, StvdFeatureCollection>;
  colors: string[][];
  threshold: number;
}
export interface SocialNetworkBlock {
  kind: string;
  node_count: number;
  edge_count: number;
  layout: string;
  directed: boolean;
  social_graph_k: number;
  nodes: ([number, number, number, number] | [number, number, number, number, string])[];
  edges: [number, number, number?][];
  nodes_sampled?: boolean;
  edges_sampled?: boolean;
  degrees?: number[];
}
export interface TransportSpatialModeSummary {
  mode: string;
  count: number;
  percent: number;
  mean_jump_km: number | null;
  mean_duration_min: number | null;
}
export interface TransportSpatialBlock {
  summary: Record<string, { total_trips: number; modes: TransportSpatialModeSummary[] }>;
  share: { categories: string[]; series: BarSeries[] };
  jump_ecdf: EcdfBlock;
}
export interface NetworkMetricSummary {
  count: number;
  mean: number | null;
  median: number | null;
  std: number | null;
  p10: number | null;
  p90: number | null;
}
export interface NetworkValidationComparisonBlock {
  comparison: "synthetic_vs_random" | "observed_vs_random";
  random_model: "degree_preserving_rnd";
  wasserstein: {
    degree: number | null;
    clustering_coefficient: number | null;
    edge_persistence: number | null;
    topological_overlap: number | null;
  };
  distributions: {
    synthetic?: Record<string, NetworkMetricSummary>;
    observed?: Record<string, NetworkMetricSummary>;
    random: Record<string, NetworkMetricSummary>;
  };
  source_network: SocialNetworkBlock;
  random_network: SocialNetworkBlock;
}
export interface NetworkValidationMetricComparisonBlock {
  comparison: "synthetic_vs_observed";
  wasserstein: {
    degree: number | null;
    clustering_coefficient: number | null;
    edge_persistence: number | null;
    topological_overlap: number | null;
  };
  distributions: {
    synthetic: Record<string, NetworkMetricSummary>;
    observed: Record<string, NetworkMetricSummary>;
  };
}
export interface NetworkValidationBlock {
  synthetic_vs_random?: NetworkValidationComparisonBlock;
  observed_vs_random?: NetworkValidationComparisonBlock;
  synthetic_vs_observed?: NetworkValidationMetricComparisonBlock;
}
export interface ChartPayload {
  mode: "comparison" | "synthetic_only";
  run_id: string;
  labels: { synthetic: string; observed?: string };
  available_filters?: { key: string; label: string }[];
  distribution_filters?: { key: string; label: string }[];
  enabled_sections?: string[];
  loaded_filters?: string[];
  metrics: {
    wasserstein: { filter_key: string; filter_label: string; metric_name: string; name?: string; value: number; unit?: string }[];
    jsd: { filter_key?: string; filter_label?: string; metric_name?: string; name?: string; value: number }[];
    cpc: { filter_key: string; filter_label: string; resolution: number; value: number }[];
    time_use: { filter_key: string; filter_label: string; metric_name: string; name?: string; value: number; unit?: string }[];
    stvd: { filter_key: string; filter_label: string; metric_name: string; name?: string; resolution: number; value: number; unit?: string }[];
  };
  ecdf: { groups: FilteredBlockMap<EcdfBlock>[] };
  transport_spatial: TransportSpatialBlock | null;
  mobility_laws: { groups: FilteredBlockMap<LawBlock>[] } | null;
  activity: { groups: FilteredActivityBlock[] } | null;
  micro_activity_usage: { groups: FilteredSingleBlock<MicroActivityUsageBlock>[] } | null;
  time_use_comparison: { groups: FilteredSingleBlock<TimeUseComparisonBlock>[] } | null;
  profiles: ProfilesBlock | null;
  motifs: { groups: FilteredSingleBlock<MotifsBlock>[] } | null;
  stvd: { groups: FilteredSingleBlock<StvdBlock>[] } | null;
  social_network: SocialNetworkBlock | null;
  warnings: string[];
}

// network_validation is fetched separately (its own endpoint/cache entry)
// so its build time doesn't block first paint of the rest of the charts --
// see fetchNetworkValidation below and web/backend/app/api/charts.py's
// /network-validation route.
export interface NetworkValidationResponse {
  run_id: string;
  network_validation: NetworkValidationBlock | null;
  warnings: string[];
}

export function fetchNetworkValidation(id: string, run?: string): Promise<NetworkValidationResponse> {
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<NetworkValidationResponse>(
    `/api/experiments/${encodeURIComponent(id)}/network-validation${q}`,
  );
}

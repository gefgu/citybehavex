// Thin fetch wrapper. Every backend response is `{ data: ... }`; we return `data`.

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
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
  profiles_enabled: boolean;
  profiles_output: string | null;
  profiles_path: string | null;
  profiles_exists: boolean;
  params: Record<string, unknown>;
  runs: Run[];
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
  profiles_enabled?: boolean;
  profiles_output?: string;
}

export function fetchExperiments(withSummary = false): Promise<Experiment[]> {
  return getJson<Experiment[]>(`/api/experiments?with_summary=${withSummary}`);
}

export function fetchExperiment(id: string): Promise<Experiment> {
  return getJson<Experiment>(`/api/experiments/${encodeURIComponent(id)}`);
}

export function updateExperiment(id: string, payload: ExperimentUpdate): Promise<Experiment> {
  return sendJson<Experiment>(`/api/experiments/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function archiveExperiment(id: string): Promise<{ archived_config: string }> {
  return sendJson<{ archived_config: string }>(
    `/api/experiments/${encodeURIComponent(id)}/archive`,
    { method: "POST", body: "{}" },
  );
}

export function deleteExperimentRun(id: string, runId: string): Promise<{ deleted: string[] }> {
  return sendJson<{ deleted: string[] }>(
    `/api/experiments/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE", body: "{}" },
  );
}

export function fetchCharts(id: string, run?: string): Promise<ChartPayload> {
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<ChartPayload>(`/api/experiments/${encodeURIComponent(id)}/charts${q}`);
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

export function fetchHomeWork(
  id: string,
  run?: string,
  filter?: DemographicFilter,
): Promise<HomeWorkResponse> {
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
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<AgentCrpPayload>(
    `/api/experiments/${encodeURIComponent(id)}/timeline/agents/${uid}/crp${q}`,
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
  degrees?: number[];
}
export interface ChartPayload {
  mode: "comparison" | "synthetic_only";
  run_id: string;
  labels: { synthetic: string; observed?: string };
  metrics: {
    wasserstein: { filter_key: string; filter_label: string; metric_name: string; name?: string; value: number; unit?: string }[];
    jsd: { filter_key?: string; filter_label?: string; metric_name?: string; name?: string; value: number }[];
    cpc: { filter_key: string; filter_label: string; resolution: number; value: number }[];
  };
  ecdf: { groups: FilteredBlockMap<EcdfBlock>[] };
  mobility_laws: { groups: FilteredBlockMap<LawBlock>[] } | null;
  activity: { groups: FilteredActivityBlock[] } | null;
  micro_activity_usage: { groups: FilteredSingleBlock<MicroActivityUsageBlock>[] } | null;
  profiles: ProfilesBlock | null;
  motifs: { groups: FilteredSingleBlock<MotifsBlock>[] } | null;
  stvd: { groups: FilteredSingleBlock<StvdBlock>[] } | null;
  social_network: SocialNetworkBlock | null;
  warnings: string[];
}

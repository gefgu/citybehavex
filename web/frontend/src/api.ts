// Thin fetch wrapper. Every backend response is `{ data: ... }`; we return `data`.

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
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
  observed_path: string | null;
  observed_exists: boolean;
  params: Record<string, unknown>;
  runs: Run[];
}

export function fetchExperiments(withSummary = false): Promise<Experiment[]> {
  return getJson<Experiment[]>(`/api/experiments?with_summary=${withSummary}`);
}

export function fetchExperiment(id: string): Promise<Experiment> {
  return getJson<Experiment>(`/api/experiments/${encodeURIComponent(id)}`);
}

export function fetchCharts(id: string, run?: string): Promise<ChartPayload> {
  const q = run ? `?run=${encodeURIComponent(run)}` : "";
  return getJson<ChartPayload>(`/api/experiments/${encodeURIComponent(id)}/charts${q}`);
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
    matrix: number[][];
    limit: number;
  };
  daily_activity_difference: {
    categories: string[];
    n_bins: number;
    labels: string[];
    matrix: number[][];
    limit: number;
  } | null;
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
  series: BarSeries[];
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
export interface ChartPayload {
  run_id: string;
  labels: { synthetic: string; observed: string };
  metrics: {
    wasserstein: { name: string; value: number; unit: string }[];
    jsd: { name: string; value: number }[];
    cpc: { resolution: number; value: number }[];
  };
  ecdf: Record<string, EcdfBlock>;
  mobility_laws: Record<string, LawBlock> | null;
  activity: ActivityBlock | null;
  profiles: ProfilesBlock | null;
  motifs: MotifsBlock | null;
  stvd: StvdBlock | null;
  warnings: string[];
}

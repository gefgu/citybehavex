import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { MapContainer, Rectangle, TileLayer, useMap } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  archiveExperiment,
  deleteExperimentRun,
  fetchExperiments,
  updateExperiment,
  type BBox,
  type Experiment,
  type ExperimentUpdate,
} from "../api";

type EditForm = {
  label: string;
  agents: string;
  days: string;
  start_date: string;
  granularity_minutes: string;
  car_speed_kmh: string;
  simulation_output: string;
  observed_path: string;
  profiles_enabled: boolean;
  profiles_output: string;
  bbox_min_lat: string;
  bbox_min_lng: string;
  bbox_max_lat: string;
  bbox_max_lng: string;
};

type PanelMode = "view" | "edit";

function fmtDate(s?: string) {
  return s ? s.slice(0, 10) : "-";
}

function fmtRunId(runId: string) {
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$/.exec(runId);
  if (!match) return runId;
  const [, year, month, day, hour, minute, second] = match;
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function paramString(value: unknown, fallback = "-") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

function bboxFromParams(params: Record<string, unknown>): BBox | null {
  const bbox = params.bbox;
  if (!bbox || typeof bbox !== "object") return null;
  const candidate = bbox as Record<string, unknown>;
  const min_lat = Number(candidate.min_lat);
  const min_lng = Number(candidate.min_lng);
  const max_lat = Number(candidate.max_lat);
  const max_lng = Number(candidate.max_lng);
  if (![min_lat, min_lng, max_lat, max_lng].every(Number.isFinite)) return null;
  return { min_lat, min_lng, max_lat, max_lng };
}

function formFromExperiment(exp: Experiment): EditForm {
  const bbox = bboxFromParams(exp.params);
  return {
    label: exp.label,
    agents: paramString(exp.params.agents, ""),
    days: paramString(exp.params.days, ""),
    start_date: paramString(exp.params.start_date, ""),
    granularity_minutes: paramString(exp.params.granularity_minutes, ""),
    car_speed_kmh: paramString(exp.params.car_speed_kmh, ""),
    simulation_output: exp.simulation_output ?? "",
    observed_path: exp.observed_path ?? "",
    profiles_enabled: Boolean(exp.profiles_enabled),
    profiles_output: exp.profiles_output ?? exp.profiles_path ?? "",
    bbox_min_lat: bbox ? String(bbox.min_lat) : "",
    bbox_min_lng: bbox ? String(bbox.min_lng) : "",
    bbox_max_lat: bbox ? String(bbox.max_lat) : "",
    bbox_max_lng: bbox ? String(bbox.max_lng) : "",
  };
}

function payloadFromForm(form: EditForm): ExperimentUpdate {
  const payload: ExperimentUpdate = {
    label: form.label.trim(),
    agents: Number(form.agents),
    days: Number(form.days),
    start_date: form.start_date.trim() || null,
    granularity_minutes: Number(form.granularity_minutes),
    car_speed_kmh: Number(form.car_speed_kmh),
    simulation_output: form.simulation_output.trim(),
    observed_path: form.observed_path.trim() || null,
    profiles_enabled: form.profiles_enabled,
    profiles_output: form.profiles_output.trim(),
  };
  const bbox = {
    min_lat: Number(form.bbox_min_lat),
    min_lng: Number(form.bbox_min_lng),
    max_lat: Number(form.bbox_max_lat),
    max_lng: Number(form.bbox_max_lng),
  };
  if (Object.values(bbox).every(Number.isFinite)) {
    payload.bbox = bbox;
  }
  return payload;
}

function chunkExperiments(experiments: Experiment[], size: number) {
  const rows: Experiment[][] = [];
  for (let i = 0; i < experiments.length; i += size) {
    rows.push(experiments.slice(i, i + size));
  }
  return rows;
}

export function Experiments() {
  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [panelMode, setPanelMode] = useState<PanelMode>("view");
  const [form, setForm] = useState<EditForm | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  async function reload(nextOpenId = openId, nextPanelMode = panelMode) {
    const data = await fetchExperiments(true);
    setExperiments(data);
    if (nextOpenId && data.some((exp) => exp.id === nextOpenId)) {
      const openExp = data.find((exp) => exp.id === nextOpenId);
      setOpenId(nextOpenId);
      setPanelMode(nextPanelMode);
      setForm(openExp ? formFromExperiment(openExp) : null);
    } else {
      setOpenId(null);
      setPanelMode("view");
      setForm(null);
    }
  }

  useEffect(() => {
    reload(null).catch((e) => setError(String(e)));
  }, []);

  const rows = useMemo(() => chunkExperiments(experiments ?? [], 3), [experiments]);
  const openExperiment = experiments?.find((exp) => exp.id === openId) ?? null;

  function toggleExperiment(exp: Experiment, mode: PanelMode) {
    setActionError(null);
    if (openId === exp.id && panelMode === mode) {
      setOpenId(null);
      setForm(null);
      return;
    }
    setOpenId(exp.id);
    setPanelMode(mode);
    setForm(formFromExperiment(exp));
  }

  async function handleSave(event: FormEvent) {
    event.preventDefault();
    if (!openExperiment || !form) return;

    setBusyAction("save");
    setActionError(null);
    try {
      await updateExperiment(openExperiment.id, payloadFromForm(form));
      await reload(openExperiment.id, "view");
    } catch (e) {
      setActionError(String(e));
    } finally {
      setBusyAction(null);
    }
  }

  async function handleArchive(exp: Experiment) {
    if (!window.confirm(`Archive experiment "${exp.label}"? The config will move out of active discovery.`)) {
      return;
    }
    setBusyAction("archive");
    setActionError(null);
    try {
      await archiveExperiment(exp.id);
      await reload(null);
    } catch (e) {
      setActionError(String(e));
    } finally {
      setBusyAction(null);
    }
  }

  async function handleDeleteRun(exp: Experiment, runId: string) {
    if (!window.confirm(`Delete run "${fmtRunId(runId)}" and its generated sibling files?`)) {
      return;
    }
    setBusyAction(`delete-run:${runId}`);
    setActionError(null);
    try {
      await deleteExperimentRun(exp.id, runId);
      await reload(exp.id, panelMode);
    } catch (e) {
      setActionError(String(e));
    } finally {
      setBusyAction(null);
    }
  }

  if (error) return <div className="state">Failed to load experiments: {error}</div>;
  if (!experiments) return <div className="state">Loading experiments...</div>;

  return (
    <>
      <h1 style={{ margin: "48px 0 8px" }}>Experiments</h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        Discovered from <code>configs/*.yaml</code>. Each run is a timestamped simulation output.
      </p>

      <div className="experiment-grid" style={{ marginTop: 24 }}>
        {rows.map((row) => {
          const expanded = row.find((exp) => exp.id === openId) ?? null;
          return (
            <div className="experiment-grid-row" key={row.map((exp) => exp.id).join(":")}>
              {row.map((exp) => (
                <ExperimentCard
                  experiment={exp}
                  isOpen={openId === exp.id}
                  key={exp.id}
                  mode={openId === exp.id ? panelMode : null}
                  onEdit={() => toggleExperiment(exp, "edit")}
                  onView={() => toggleExperiment(exp, "view")}
                />
              ))}
              {expanded && openExperiment && panelMode === "view" && (
                <ExperimentViewPanel
                  actionError={actionError}
                  busyAction={busyAction}
                  experiment={openExperiment}
                  onDeleteRun={(runId) => handleDeleteRun(openExperiment, runId)}
                />
              )}
              {expanded && openExperiment && form && panelMode === "edit" && (
                <ExperimentPanel
                  actionError={actionError}
                  busyAction={busyAction}
                  experiment={openExperiment}
                  form={form}
                  onArchive={() => handleArchive(openExperiment)}
                  onClose={() => {
                    setOpenId(null);
                    setForm(null);
                  }}
                  onFormChange={setForm}
                  onSave={handleSave}
                />
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}

function ExperimentCard({
  experiment,
  isOpen,
  mode,
  onEdit,
  onView,
}: {
  experiment: Experiment;
  isOpen: boolean;
  mode: PanelMode | null;
  onEdit: () => void;
  onView: () => void;
}) {
  return (
    <article
      className={`card exp-card ${isOpen ? "is-open" : ""}`}
      role="button"
      tabIndex={0}
      onClick={onView}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onView();
        }
      }}
    >
      <div className="exp-head">
        <h3>{experiment.label}</h3>
        <span className="badge">{experiment.id}</span>
      </div>
      <div className="param-list">
        <span>
          <b>{paramString(experiment.params.agents)}</b> agents
        </span>
        <span>
          <b>{paramString(experiment.params.days)}</b> days
        </span>
        <span>
          granularity <b>{paramString(experiment.params.granularity_minutes)}m</b>
        </span>
        <span>{experiment.runs.length.toLocaleString()} runs</span>
        <span>
          Real-Data comparision configured: {experiment.observed_exists ? "true" : "false"}
        </span>
      </div>
      <button
        className="btn btn-secondary exp-toggle"
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onEdit();
        }}
      >
        {isOpen && mode === "edit" ? "Close" : "Edit"}
      </button>
    </article>
  );
}

function ExperimentViewPanel({
  actionError,
  busyAction,
  experiment,
  onDeleteRun,
}: {
  actionError: string | null;
  busyAction: string | null;
  experiment: Experiment;
  onDeleteRun: (runId: string) => void;
}) {
  return (
    <section className="card experiment-panel">
      <div className="experiment-panel-head">
        <div>
          <h2>{experiment.label}</h2>
          <p>
            <code>{experiment.config}</code>
          </p>
        </div>
        <span className="badge">{experiment.id}</span>
      </div>

      {actionError && <div className="inline-error">{actionError}</div>}

      <div className="experiment-view-summary">
        <div className="param-list">
          <span>
            <b>{paramString(experiment.params.agents)}</b> agents
          </span>
          <span>
            <b>{paramString(experiment.params.days)}</b> days
          </span>
          <span>
            granularity <b>{paramString(experiment.params.granularity_minutes)}m</b>
          </span>
          <span>
            car speed <b>{paramString(experiment.params.car_speed_kmh)} km/h</b>
          </span>
          <span>
            Real-Data comparision configured: {experiment.observed_exists ? "true" : "false"}
          </span>
          <span>profiles: {experiment.profiles_enabled ? "enabled" : "disabled"}</span>
        </div>
      </div>

      <ExperimentRuns
        busyAction={busyAction}
        experiment={experiment}
        onDeleteRun={onDeleteRun}
      />
    </section>
  );
}

function ExperimentPanel({
  actionError,
  busyAction,
  experiment,
  form,
  onArchive,
  onClose,
  onFormChange,
  onSave,
}: {
  actionError: string | null;
  busyAction: string | null;
  experiment: Experiment;
  form: EditForm;
  onArchive: () => void;
  onClose: () => void;
  onFormChange: (form: EditForm) => void;
  onSave: (event: FormEvent) => void;
}) {
  function updateField<K extends keyof EditForm>(key: K, value: EditForm[K]) {
    onFormChange({ ...form, [key]: value });
  }

  return (
    <section className="card experiment-panel">
      <div className="experiment-panel-head">
        <div>
          <h2>{experiment.label}</h2>
          <p>
            <code>{experiment.config}</code>
          </p>
        </div>
        <div className="experiment-panel-actions">
          <button
            className="btn btn-secondary"
            disabled={busyAction !== null}
            type="button"
            onClick={onClose}
          >
            Close
          </button>
          <button
            className="btn btn-danger"
            disabled={busyAction !== null}
            type="button"
            onClick={onArchive}
          >
            {busyAction === "archive" ? "Archiving..." : "Archive"}
          </button>
        </div>
      </div>

      {actionError && <div className="inline-error">{actionError}</div>}

      <div className="experiment-panel-body">
        <form className="experiment-edit-form" onSubmit={onSave}>
          <div className="warnings experiment-edit-note">
            More detailed configs can be edited in the <code>.yaml</code> file.
          </div>
          <label>
            Label
            <input value={form.label} onChange={(e) => updateField("label", e.target.value)} />
          </label>
          <label>
            Agents
            <input
              min="1"
              type="number"
              value={form.agents}
              onChange={(e) => updateField("agents", e.target.value)}
            />
          </label>
          <label>
            Days
            <input
              min="1"
              type="number"
              value={form.days}
              onChange={(e) => updateField("days", e.target.value)}
            />
          </label>
          <label>
            Start date
            <input value={form.start_date} onChange={(e) => updateField("start_date", e.target.value)} />
          </label>
          <label>
            Granularity minutes
            <input
              min="1"
              type="number"
              value={form.granularity_minutes}
              onChange={(e) => updateField("granularity_minutes", e.target.value)}
            />
          </label>
          <label>
            Car speed km/h
            <input
              min="0.1"
              step="0.1"
              type="number"
              value={form.car_speed_kmh}
              onChange={(e) => updateField("car_speed_kmh", e.target.value)}
            />
          </label>
          <label>
            Simulation output
            <input
              value={form.simulation_output}
              onChange={(e) => updateField("simulation_output", e.target.value)}
            />
          </label>
          <label>
            Observed path
            <input value={form.observed_path} onChange={(e) => updateField("observed_path", e.target.value)} />
          </label>
          <label className="checkbox-field">
            <input
              checked={form.profiles_enabled}
              type="checkbox"
              onChange={(e) => updateField("profiles_enabled", e.target.checked)}
            />
            Profiles enabled
          </label>
          <label>
            Profiles output
            <input
              value={form.profiles_output}
              onChange={(e) => updateField("profiles_output", e.target.value)}
            />
          </label>
          <div className="form-actions">
            <button className="btn btn-primary" disabled={busyAction !== null} type="submit">
              {busyAction === "save" ? "Saving..." : "Save"}
            </button>
            <button className="btn btn-secondary pretend-run-button" type="button">
              Run
            </button>
          </div>
        </form>

        <BBoxEditor form={form} onChange={updateField} />
      </div>
    </section>
  );
}

function FitBBoxBounds({ bounds }: { bounds: LatLngBoundsExpression }) {
  const map = useMap();

  useEffect(() => {
    map.invalidateSize();
    map.fitBounds(bounds, { animate: false, padding: [18, 18] });
  }, [bounds, map]);

  return null;
}

function BBoxEditor({
  form,
  onChange,
}: {
  form: EditForm;
  onChange: <K extends keyof EditForm>(key: K, value: EditForm[K]) => void;
}) {
  const bbox = {
    min_lat: Number(form.bbox_min_lat),
    min_lng: Number(form.bbox_min_lng),
    max_lat: Number(form.bbox_max_lat),
    max_lng: Number(form.bbox_max_lng),
  };
  const hasBounds =
    Object.values(bbox).every(Number.isFinite) &&
    bbox.min_lat < bbox.max_lat &&
    bbox.min_lng < bbox.max_lng;
  const bounds: LatLngBoundsExpression = hasBounds
    ? [
        [bbox.min_lat, bbox.min_lng],
        [bbox.max_lat, bbox.max_lng],
      ]
    : [
        [48.5, 1.6],
        [49.16, 2.975],
      ];
  const center: [number, number] = hasBounds
    ? [(bbox.min_lat + bbox.max_lat) / 2, (bbox.min_lng + bbox.max_lng) / 2]
    : [48.8566, 2.3522];

  return (
    <div className="experiment-bbox-editor">
      <div className="experiment-runs-head">
        <h3>Bbox</h3>
        <span className="muted-small">Edit the bounding box used by the config.</span>
      </div>
      <MapContainer
        bounds={bounds}
        center={center}
        className="experiment-bbox-map"
        scrollWheelZoom
        zoom={10}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap &copy; CARTO'
        />
        <FitBBoxBounds bounds={bounds} />
        {hasBounds && (
          <Rectangle
            bounds={bounds}
            pathOptions={{ color: "#aa2d00", fillOpacity: 0.08, weight: 2 }}
          />
        )}
      </MapContainer>
      <div className="bbox-fields">
        <label>
          Min lat
          <input
            value={form.bbox_min_lat}
            onChange={(e) => onChange("bbox_min_lat", e.target.value)}
          />
        </label>
        <label>
          Min lng
          <input
            value={form.bbox_min_lng}
            onChange={(e) => onChange("bbox_min_lng", e.target.value)}
          />
        </label>
        <label>
          Max lat
          <input
            value={form.bbox_max_lat}
            onChange={(e) => onChange("bbox_max_lat", e.target.value)}
          />
        </label>
        <label>
          Max lng
          <input
            value={form.bbox_max_lng}
            onChange={(e) => onChange("bbox_max_lng", e.target.value)}
          />
        </label>
      </div>
    </div>
  );
}

function ExperimentRuns({
  busyAction,
  experiment,
  onDeleteRun,
}: {
  busyAction: string | null;
  experiment: Experiment;
  onDeleteRun: (runId: string) => void;
}) {
  const runnable = experiment.observed_exists && experiment.runs.length > 0;

  return (
    <div className="experiment-runs">
      <h3>Runs</h3>
      {experiment.runs.length === 0 && <p className="muted-small">No simulation runs found yet.</p>}
      {experiment.runs.map((run) => (
        <div className="run-row" key={run.run_id}>
          <div className="run-main">
            <span className="run-id">{fmtRunId(run.run_id)}</span>
            <span className="run-meta">
              {run.summary
                ? `${run.summary.rows.toLocaleString()} rows · ${
                    run.summary.uids?.toLocaleString() ?? "?"
                  } users · ${fmtDate(run.summary.date_start)} to ${fmtDate(run.summary.date_end)}`
                : run.summary_error ?? ""}
            </span>
          </div>
          <div className="run-actions">
            {runnable && (
              <Link to={`/experiments/${experiment.id}/charts?run=${run.run_id}`} className="btn btn-secondary">
                Charts
              </Link>
            )}
            {run.summary && run.summary.rows > 0 && (
              <Link to={`/experiments/${experiment.id}/timeline?run=${run.run_id}`} className="btn btn-secondary">
                Timeline
              </Link>
            )}
            <button
              className="btn btn-danger"
              disabled={busyAction !== null}
              type="button"
              onClick={() => onDeleteRun(run.run_id)}
            >
              {busyAction === `delete-run:${run.run_id}` ? "Deleting..." : "Delete"}
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

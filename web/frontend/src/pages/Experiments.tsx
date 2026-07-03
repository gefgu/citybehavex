import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  archiveExperiment,
  deleteExperimentRun,
  fetchExperiments,
  updateExperiment,
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
};

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

function formFromExperiment(exp: Experiment): EditForm {
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
  };
}

function payloadFromForm(form: EditForm): ExperimentUpdate {
  return {
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
  const [form, setForm] = useState<EditForm | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  async function reload(nextOpenId = openId) {
    const data = await fetchExperiments(true);
    setExperiments(data);
    if (nextOpenId && data.some((exp) => exp.id === nextOpenId)) {
      const openExp = data.find((exp) => exp.id === nextOpenId);
      setOpenId(nextOpenId);
      setForm(openExp ? formFromExperiment(openExp) : null);
    } else {
      setOpenId(null);
      setForm(null);
    }
  }

  useEffect(() => {
    reload(null).catch((e) => setError(String(e)));
  }, []);

  const rows = useMemo(() => chunkExperiments(experiments ?? [], 3), [experiments]);
  const openExperiment = experiments?.find((exp) => exp.id === openId) ?? null;

  function toggleExperiment(exp: Experiment) {
    setActionError(null);
    if (openId === exp.id) {
      setOpenId(null);
      setForm(null);
      return;
    }
    setOpenId(exp.id);
    setForm(formFromExperiment(exp));
  }

  async function handleSave(event: FormEvent) {
    event.preventDefault();
    if (!openExperiment || !form) return;

    setBusyAction("save");
    setActionError(null);
    try {
      await updateExperiment(openExperiment.id, payloadFromForm(form));
      await reload(openExperiment.id);
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
      await reload(exp.id);
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
                  onToggle={() => toggleExperiment(exp)}
                />
              ))}
              {expanded && openExperiment && form && (
                <ExperimentPanel
                  actionError={actionError}
                  busyAction={busyAction}
                  experiment={openExperiment}
                  form={form}
                  onArchive={() => handleArchive(openExperiment)}
                  onDeleteRun={(runId) => handleDeleteRun(openExperiment, runId)}
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
  onToggle,
}: {
  experiment: Experiment;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <article className={`card exp-card ${isOpen ? "is-open" : ""}`}>
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
        <span>observed: {experiment.observed_exists ? "yes" : "missing"}</span>
      </div>
      <button className="btn btn-secondary exp-toggle" type="button" onClick={onToggle}>
        {isOpen ? "Close" : "Edit"}
      </button>
    </article>
  );
}

function ExperimentPanel({
  actionError,
  busyAction,
  experiment,
  form,
  onArchive,
  onDeleteRun,
  onFormChange,
  onSave,
}: {
  actionError: string | null;
  busyAction: string | null;
  experiment: Experiment;
  form: EditForm;
  onArchive: () => void;
  onDeleteRun: (runId: string) => void;
  onFormChange: (form: EditForm) => void;
  onSave: (event: FormEvent) => void;
}) {
  const runnable = experiment.observed_exists && experiment.runs.length > 0;

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
        <button
          className="btn btn-danger"
          disabled={busyAction !== null}
          type="button"
          onClick={onArchive}
        >
          {busyAction === "archive" ? "Archiving..." : "Archive"}
        </button>
      </div>

      {actionError && <div className="inline-error">{actionError}</div>}

      <div className="experiment-panel-body">
        <form className="experiment-edit-form" onSubmit={onSave}>
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
          </div>
        </form>

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
      </div>
    </section>
  );
}

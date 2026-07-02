import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchExperiments, type Experiment } from "../api";

function fmtDate(s?: string) {
  return s ? s.slice(0, 10) : "—";
}

export function Experiments() {
  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchExperiments(true).then(setExperiments).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="state">Failed to load experiments: {error}</div>;
  if (!experiments) return <div className="state">Loading experiments…</div>;

  return (
    <>
      <h1 style={{ margin: "48px 0 8px" }}>Experiments</h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        Discovered from <code>configs/*.yaml</code>. Each run is a timestamped
        simulation output.
      </p>

      <div className="card-grid" style={{ marginTop: 24 }}>
        {experiments.map((exp) => {
          const runnable = exp.observed_exists && exp.runs.length > 0;
          return (
            <div className="card exp-card" key={exp.id}>
              <div className="exp-head">
                <h3>{exp.label}</h3>
                <span className="badge">{exp.id}</span>
              </div>
              <div className="param-list">
                <span>
                  <b>{String(exp.params.agents)}</b> agents
                </span>
                <span>
                  <b>{String(exp.params.days)}</b> days
                </span>
                <span>
                  granularity <b>{String(exp.params.granularity_minutes)}m</b>
                </span>
                <span>observed: {exp.observed_exists ? "yes" : "missing"}</span>
              </div>

              {exp.runs.length === 0 && (
                <p style={{ color: "var(--muted)", fontSize: 13 }}>No simulation runs found yet.</p>
              )}

              {exp.runs.map((run) => (
                <div className="run-row" key={run.run_id}>
                  <span className="run-id">{run.run_id}</span>
                  <span className="run-meta">
                    {run.summary
                      ? `${run.summary.rows.toLocaleString()} rows · ${
                          run.summary.uids?.toLocaleString() ?? "?"
                        } users · ${fmtDate(run.summary.date_start)}→${fmtDate(run.summary.date_end)}`
                      : run.summary_error ?? ""}
                  </span>
                  <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                    {runnable && (
                      <Link
                        to={`/experiments/${exp.id}/charts?run=${run.run_id}`}
                        className="btn btn-secondary"
                        style={{ padding: "4px 14px", fontSize: 13 }}
                      >
                        View charts
                      </Link>
                    )}
                    {run.summary && run.summary.rows > 0 && (
                      <Link
                        to={`/experiments/${exp.id}/timeline?run=${run.run_id}`}
                        className="btn btn-secondary"
                        style={{ padding: "4px 14px", fontSize: 13 }}
                      >
                        View timeline
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </>
  );
}

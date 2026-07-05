import { useEffect, useMemo, useState } from "react";
import { fetchTimelineAgentSocial, type AgentSocialPayload } from "../api";

function fmtNumber(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

function fmtParam(value: string | number | boolean | null | undefined): string {
  if (value == null) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
}

export function SocialSummaryPanel({
  expId,
  uid,
  runId,
}: {
  expId: string;
  uid: number;
  runId?: string;
}) {
  const [data, setData] = useState<AgentSocialPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setData(null);
    setError(null);
    fetchTimelineAgentSocial(expId, uid, runId).then(setData).catch((e) => setError(String(e)));
  }, [expId, uid, runId]);

  const topFriends = useMemo(
    () =>
      data
        ? [...data.friends].sort((a, b) => b.social_strength - a.social_strength).slice(0, 20)
        : [],
    [data],
  );

  let summary = "Loading";
  if (error) summary = "Unavailable";
  else if (data) {
    summary = `${data.parameters.degree} friends · strength ${fmtNumber(data.parameters.total_social_strength)}`;
  }

  return (
    <div className={`timeline-detail-panel collapsible-panel${open ? " is-open" : ""}`}>
      <button
        className="collapsible-panel-trigger"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span>
          <span className="section-header">Social network</span>
          <span className="collapsible-panel-summary">{summary}</span>
        </span>
        <span className="collapsible-panel-chevron" aria-hidden="true">⌄</span>
      </button>

      {open && (
        <div className="collapsible-panel-body">
          {error ? (
            <div className="timeline-detail-empty">Failed to load social network: {error}</div>
          ) : !data ? (
            <div className="timeline-detail-empty">Loading social network...</div>
          ) : (
            <>
              {data.warnings.length > 0 && <div className="warnings">{data.warnings.join("; ")}</div>}

              <div className="section-header crp-section-header">Social parameters</div>
              <div className="crp-stats-row social-stats-row">
                <div className="crp-stat">
                  <span className="crp-stat-label">degree</span>
                  <span className="crp-stat-value">{data.parameters.degree}</span>
                </div>
                <div className="crp-stat">
                  <span className="crp-stat-label">strength</span>
                  <span className="crp-stat-value">{fmtNumber(data.parameters.total_social_strength)}</span>
                </div>
                <div className="crp-stat">
                  <span className="crp-stat-label">k</span>
                  <span className="crp-stat-value">{fmtParam(data.parameters.social_graph_k)}</span>
                </div>
              </div>

              <table className="agent-table timeline-detail-table social-param-table">
                <tbody>
                  <tr>
                    <td>layout</td>
                    <td>{fmtParam(data.parameters.layout)}</td>
                  </tr>
                  <tr>
                    <td>kind</td>
                    <td>{fmtParam(data.parameters.kind)}</td>
                  </tr>
                  <tr>
                    <td>rho / gamma / alpha</td>
                    <td>
                      {fmtParam(data.parameters.rho)} / {fmtParam(data.parameters.gamma)} /{" "}
                      {fmtParam(data.parameters.alpha)}
                    </td>
                  </tr>
                  <tr>
                    <td>update / window</td>
                    <td>
                      {fmtParam(data.parameters.dt_update_mob_sim_hours)}h /{" "}
                      {fmtParam(data.parameters.indipendency_window_hours)}h
                    </td>
                  </tr>
                </tbody>
              </table>

              <div className="section-header crp-section-header">Friends</div>
              {topFriends.length === 0 ? (
                <div className="timeline-detail-empty">No social friends available for this agent.</div>
              ) : (
                <div className="agent-scroll social-friends-scroll">
                  <table className="agent-table timeline-detail-table">
                    <thead>
                      <tr>
                        <th>friend</th>
                        <th>strength</th>
                        <th>similarity</th>
                        <th>encounters</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topFriends.map((friend) => (
                        <tr key={friend.uid}>
                          <td>
                            {friend.name ?? `Agent ${friend.uid}`}
                            {friend.reciprocated ? <span className="social-reciprocal"> ↔</span> : null}
                          </td>
                          <td>{fmtNumber(friend.social_strength)}</td>
                          <td>{fmtNumber(friend.embedding_similarity)}</td>
                          <td>{friend.encounter_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

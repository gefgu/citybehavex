import { useEffect, useState } from "react";
import { fetchTimelineAgent, type AgentProfilePayload } from "../api";

function fmtDateTime(s: string): string {
  return s.replace("T", " ");
}

export function AgentSidebar({
  uid,
  expId,
  runId,
  onClose,
}: {
  uid: number;
  expId: string;
  runId?: string;
  onClose: () => void;
}) {
  const [data, setData] = useState<AgentProfilePayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
    fetchTimelineAgent(expId, uid, runId).then(setData).catch((e) => setError(String(e)));
  }, [expId, uid, runId]);

  return (
    <div className="timeline-sidebar">
      <div className="timeline-sidebar-header">
        <h3>{data?.profile?.name ?? `Agent ${uid}`}</h3>
        <button className="btn btn-secondary" style={{ padding: "2px 10px" }} onClick={onClose}>
          ×
        </button>
      </div>

      {error && <div className="state">Failed to load agent: {error}</div>}
      {!error && !data && <div className="state">Loading agent…</div>}

      {data && (
        <>
          {data.warnings.length > 0 && (
            <div className="warnings">{data.warnings.join("; ")}</div>
          )}

          {data.profile && (
            <table className="agent-table">
              <tbody>
                <tr>
                  <td>uid</td>
                  <td>{data.profile.uid}</td>
                </tr>
                <tr>
                  <td>gender</td>
                  <td>{data.profile.gender}</td>
                </tr>
                <tr>
                  <td>age</td>
                  <td>{data.profile.age}</td>
                </tr>
                <tr>
                  <td>education</td>
                  <td>{data.profile.education}</td>
                </tr>
                <tr>
                  <td>health</td>
                  <td>{data.profile.health}/5</td>
                </tr>
                <tr>
                  <td>household</td>
                  <td>{data.profile.household}</td>
                </tr>
                <tr>
                  <td>job</td>
                  <td>{data.profile.job}</td>
                </tr>
                <tr>
                  <td>has car</td>
                  <td>{data.profile.has_car ? "yes" : "no"}</td>
                </tr>
                <tr>
                  <td>has bike</td>
                  <td>{data.profile.has_bike ? "yes" : "no"}</td>
                </tr>
              </tbody>
            </table>
          )}

          {data.narrative && <p className="agent-narrative">{data.narrative}</p>}

          {data.trips.length > 0 && (
            <>
              <div className="section-header">Trip history</div>
              <div className="agent-scroll">
                <table className="agent-table">
                  <thead>
                    <tr>
                      <th>arrival</th>
                      <th>purpose</th>
                      <th>dwell (min)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.trips.map((t, i) => (
                      <tr key={i}>
                        <td>{fmtDateTime(t.arrival)}</td>
                        <td>{t.purpose}</td>
                        <td>{t.dwell_minutes.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {data.encounters.length > 0 && (
            <>
              <div className="section-header">Recent encounters</div>
              <div className="agent-scroll">
                <table className="agent-table">
                  <thead>
                    <tr>
                      <th>agent</th>
                      <th>time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.encounters.map((e, i) => (
                      <tr key={i}>
                        <td>{e.contact_uid}</td>
                        <td>{fmtDateTime(e.ts)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

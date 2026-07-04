import { useEffect, useState } from "react";
import { fetchTimelineAgentCrp, type AgentCrpPayload } from "../api";

type DayType = "weekday" | "weekend";

export function CrpSummaryPanel({
  expId,
  uid,
  runId,
}: {
  expId: string;
  uid: number;
  runId?: string;
}) {
  const [data, setData] = useState<AgentCrpPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dayType, setDayType] = useState<DayType>("weekday");

  useEffect(() => {
    setData(null);
    setError(null);
    fetchTimelineAgentCrp(expId, uid, runId).then(setData).catch((e) => setError(String(e)));
  }, [expId, uid, runId]);

  if (error) {
    return (
      <div className="timeline-detail-panel timeline-detail-empty">
        Failed to load diary selection: {error}
      </div>
    );
  }
  if (!data) {
    return <div className="timeline-detail-panel timeline-detail-empty">Loading diary selection…</div>;
  }
  if (data.T_a == null || data.alpha_a == null || data.diaries.length === 0) {
    return (
      <div className="timeline-detail-panel timeline-detail-empty">
        No ddCRP diary selection data available for this run.
      </div>
    );
  }

  const T_a = data.T_a;
  const alpha_a = data.alpha_a;
  const isWeekend = dayType === "weekend";
  const bank = data.diaries.filter((d) => d.is_weekend === isWeekend);

  const topUsed = [...bank].sort((a, b) => b.usage_count - a.usage_count).slice(0, 5);
  const maxUsage = Math.max(1, ...topUsed.map((d) => d.usage_count));

  const probs = bank
    .map((d) => {
      const count = d.usage_count > 0 ? d.usage_count : alpha_a;
      return { ...d, weight: count * Math.exp(d.sim / T_a) };
    })
    .filter((d) => Number.isFinite(d.weight));
  const totalWeight = probs.reduce((sum, d) => sum + d.weight, 0) || 1;
  const topProbs = probs
    .map((d) => ({ ...d, prob: d.weight / totalWeight }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 10);

  return (
    <div className="timeline-detail-panel">
      <div className="timeline-detail-header">
        <div className="section-header">Diary selection (ddCRP)</div>
        <div className="crp-panel-toggle">
          <button
            className={dayType === "weekday" ? "is-active" : ""}
            onClick={() => setDayType("weekday")}
          >
            Weekday
          </button>
          <button
            className={dayType === "weekend" ? "is-active" : ""}
            onClick={() => setDayType("weekend")}
          >
            Weekend
          </button>
        </div>
      </div>

      <div className="section-header crp-section-header">Agent parameters</div>
      <div className="crp-stats-row">
        <div className="crp-stat">
          <span className="crp-stat-label">T</span>
          <span className="crp-stat-value">{T_a.toFixed(2)}</span>
        </div>
        <div className="crp-stat">
          <span className="crp-stat-label">α</span>
          <span className="crp-stat-value">{alpha_a.toFixed(2)}</span>
        </div>
      </div>

      {bank.length === 0 ? (
        <div className="timeline-detail-empty">No {dayType} diaries in this bank.</div>
      ) : (
        <>
          <div className="section-header crp-section-header">Top diaries ({dayType})</div>
          <table className="agent-table timeline-detail-table">
            <thead>
              <tr>
                <th>diary</th>
                <th>days used</th>
                <th>similarity</th>
                <th>usage</th>
              </tr>
            </thead>
            <tbody>
              {topUsed.map((d) => (
                <tr key={d.diary_id}>
                  <td>{d.diary_id}</td>
                  <td>{d.usage_count}</td>
                  <td>{d.sim.toFixed(2)}</td>
                  <td className="crp-bar-cell">
                    <div className="crp-bar-track">
                      <div
                        className="crp-bar-fill"
                        style={{ width: `${(d.usage_count / maxUsage) * 100}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="section-header crp-section-header">Next-day probabilities ({dayType})</div>
          <div>
            {topProbs.map((d, i) => (
              <div key={d.diary_id} className={`crp-prob-row${i === 0 ? " is-top" : ""}`}>
                <span className="crp-prob-label">{d.diary_id}</span>
                <div className="crp-prob-track">
                  <div className="crp-prob-fill" style={{ width: `${d.prob * 100}%` }} />
                </div>
                <span className="crp-prob-value">{(d.prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

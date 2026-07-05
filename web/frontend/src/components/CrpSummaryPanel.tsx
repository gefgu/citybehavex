import { useEffect, useMemo, useState } from "react";
import { fetchTimelineAgentCrp, type AgentCrpPayload } from "../api";

// Preferred display order for the built-in calendar day types; any other day
// type (e.g. a config-declared special day like "emergency") sorts after them.
const DAY_TYPE_ORDER = ["weekday", "weekend"];

function dayTypeLabel(dayType: string): string {
  return dayType.charAt(0).toUpperCase() + dayType.slice(1);
}

function sortDayTypes(dayTypes: string[]): string[] {
  return [...dayTypes].sort((a, b) => {
    const ia = DAY_TYPE_ORDER.indexOf(a);
    const ib = DAY_TYPE_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

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
  const [dayType, setDayType] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setData(null);
    setError(null);
    setDayType(null);
    fetchTimelineAgentCrp(expId, uid, runId).then(setData).catch((e) => setError(String(e)));
  }, [expId, uid, runId]);

  const dayTypes = useMemo(
    () => (data ? sortDayTypes(Array.from(new Set(data.diaries.map((d) => d.day_type)))) : []),
    [data],
  );

  let summary = "Loading";
  if (error) summary = "Unavailable";
  else if (data?.T_a != null && data.alpha_a != null && data.diaries.length > 0) {
    summary = `T ${data.T_a.toFixed(2)} · α ${data.alpha_a.toFixed(2)} · ${data.diaries.length} diaries`;
  } else if (data) summary = "No data";

  const T_a = data?.T_a ?? null;
  const alpha_a = data?.alpha_a ?? null;
  const activeDayType = dayType && dayTypes.includes(dayType) ? dayType : dayTypes[0];
  const bank = data?.diaries.filter((d) => d.day_type === activeDayType) ?? [];

  const topUsed = [...bank].sort((a, b) => b.usage_count - a.usage_count).slice(0, 5);
  const maxUsage = Math.max(1, ...topUsed.map((d) => d.usage_count));

  const probs =
    T_a == null || alpha_a == null
      ? []
      : bank
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
    <div className={`timeline-detail-panel collapsible-panel${open ? " is-open" : ""}`}>
      <button
        className="collapsible-panel-trigger"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span>
          <span className="section-header">Diary selection (ddCRP)</span>
          <span className="collapsible-panel-summary">{summary}</span>
        </span>
        <span className="collapsible-panel-chevron" aria-hidden="true">⌄</span>
      </button>

      {open && (
        <div className="collapsible-panel-body">
          {error ? (
            <div className="timeline-detail-empty">Failed to load diary selection: {error}</div>
          ) : !data ? (
            <div className="timeline-detail-empty">Loading diary selection...</div>
          ) : T_a == null || alpha_a == null || data.diaries.length === 0 ? (
            <div className="timeline-detail-empty">No ddCRP diary selection data available for this run.</div>
          ) : (
            <>
              {data.warnings.length > 0 && <div className="warnings">{data.warnings.join("; ")}</div>}
              <div className="timeline-detail-header">
                <div className="section-header crp-section-header">Agent parameters</div>
                <div className="crp-panel-toggle">
                  {dayTypes.map((dt) => (
                    <button
                      key={dt}
                      className={activeDayType === dt ? "is-active" : ""}
                      onClick={() => setDayType(dt)}
                    >
                      {dayTypeLabel(dt)}
                    </button>
                  ))}
                </div>
              </div>
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
                <div className="timeline-detail-empty">No {activeDayType} diaries in this bank.</div>
              ) : (
                <>
                  <div className="section-header crp-section-header">Top diaries ({activeDayType})</div>
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

                  <div className="section-header crp-section-header">Next-day probabilities ({activeDayType})</div>
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
            </>
          )}
        </div>
      )}
    </div>
  );
}

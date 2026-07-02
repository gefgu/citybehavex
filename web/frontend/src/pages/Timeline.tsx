import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { fetchTimelineMeta, type TimelineMeta } from "../api";
import { TimelineMap } from "../components/TimelineMap";
import { AgentSidebar } from "../components/AgentSidebar";

export function Timeline() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const run = params.get("run") ?? undefined;
  const [meta, setMeta] = useState<TimelineMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedUid, setSelectedUid] = useState<number | null>(null);

  useEffect(() => {
    setMeta(null);
    setError(null);
    setSelectedUid(null);
    fetchTimelineMeta(id, run).then(setMeta).catch((e) => setError(String(e)));
  }, [id, run]);

  if (error) return <div className="state">Failed to load timeline: {error}</div>;
  if (!meta)
    return <div className="state">Preparing timeline… (first load can take a while for large cities)</div>;
  if (!meta.bbox || !meta.date_start || !meta.date_end)
    return <div className="state">This run has no usable trajectory data for a timeline.</div>;

  return (
    <>
      <h1 style={{ margin: "48px 0 4px" }}>Timeline</h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        <Link to="/experiments">experiments</Link> / {id} · run <code>{meta.run_id}</code> ·{" "}
        {meta.agents_total?.toLocaleString() ?? "?"} agents
      </p>

      <div className="timeline-layout">
        <TimelineMap meta={meta} expId={id} runId={run} onSelectAgent={setSelectedUid} />
        {selectedUid !== null && (
          <AgentSidebar
            uid={selectedUid}
            expId={id}
            runId={run}
            onClose={() => setSelectedUid(null)}
          />
        )}
      </div>

      <div style={{ height: 48 }} />
    </>
  );
}

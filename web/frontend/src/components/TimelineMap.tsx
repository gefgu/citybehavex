import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import type { FeatureCollection, Point } from "geojson";
import { fetchTimelineLegs, type TimelineMeta, type TimelineSegment } from "../api";
import { DEFAULT_PURPOSE_COLOR, PURPOSE_COLOR } from "../charts/theme";

// How far ahead of the playback clock we prefetch leg data, how close to the
// prefetched edge we trigger the next fetch, and how long behind the clock we
// keep old segments around before dropping them (bounds memory during long
// playback sessions).
const LOOKAHEAD_MS = 20 * 60 * 1000;
const REFETCH_MARGIN_MS = 5 * 60 * 1000;
const RETENTION_MS = 10 * 60 * 1000;
const DEFAULT_MAX_AGENTS = 2000;
const SPEED_OPTIONS = [1, 10, 60, 300];
// Cap how often we recompute positions and re-upload the GeoJSON source to
// Mapbox. Agents move at city scale — 60fps buffer re-uploads are wasted GPU
// work a human can't perceive, and repeatedly hammering the GPU like that is
// exactly the kind of load that trips WebGL context loss on some driver
// combinations (e.g. NVIDIA + DMABUF on Linux Firefox, a known upstream bug).
const RENDER_INTERVAL_MS = 1000 / 15;

interface Waypoint {
  t: number;
  lat: number;
  lng: number;
}

interface Seg {
  kind: "dwell" | "leg";
  t_start: number;
  t_end: number;
  o_lat: number;
  o_lng: number;
  d_lat: number;
  d_lng: number;
  purpose: string;
  mode: "stay" | "car" | "walk" | "bike" | "rail";
  waypoints?: Waypoint[];
}

type AgentFeatureCollection = FeatureCollection<Point, { uid: number; purpose: string; mode: string }>;

const EMPTY_FC: AgentFeatureCollection = { type: "FeatureCollection", features: [] };
const MODE_LABEL: Record<string, string> = {
  stay: "Stay",
  car: "Car",
  walk: "Walk",
  bike: "Bike",
  rail: "Rail",
};

function makeModeIcon(mode: string): ImageData {
  const size = 32;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d canvas unavailable");
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#000000";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 2;
  ctx.beginPath();
  if (mode === "walk") {
    ctx.moveTo(16, 5);
    ctx.lineTo(28, 27);
    ctx.lineTo(4, 27);
    ctx.closePath();
  } else if (mode === "bike") {
    for (let i = 0; i < 5; i++) {
      const a = -Math.PI / 2 + (i * Math.PI * 2) / 5;
      const x = 16 + Math.cos(a) * 12;
      const y = 16 + Math.sin(a) * 12;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
  } else if (mode === "rail") {
    ctx.moveTo(16, 4);
    ctx.lineTo(28, 16);
    ctx.lineTo(16, 28);
    ctx.lineTo(4, 16);
    ctx.closePath();
  } else if (mode === "car") {
    ctx.rect(5, 9, 22, 14);
  } else {
    ctx.arc(16, 16, 11, 0, Math.PI * 2);
  }
  ctx.fill();
  return ctx.getImageData(0, 0, size, size);
}

// Sim timestamps are naive (no timezone, not tied to any real place) — parsed
// and formatted as local wall-clock strings throughout, consistently in both
// directions, so only relative ms values ever matter.
function parseSimTime(s: string): number {
  return new Date(s.includes("T") ? s : s.replace(" ", "T")).getTime();
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function toRequestIso(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(
    d.getMinutes(),
  )}:${pad2(d.getSeconds())}`;
}

function formatClock(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(
    d.getMinutes(),
  )}:${pad2(d.getSeconds())}`;
}

// Finds the two waypoints bracketing `t` (or the closest pair at either end
// if `t` falls outside the path's own time range). Waypoint counts per leg
// are small (capped server-side), so a linear scan is plenty.
function bracketWaypoints(waypoints: Waypoint[], t: number): [Waypoint, Waypoint] {
  for (let i = 0; i < waypoints.length - 1; i++) {
    if (t <= waypoints[i + 1].t) return [waypoints[i], waypoints[i + 1]];
  }
  return [waypoints[waypoints.length - 2], waypoints[waypoints.length - 1]];
}

function findActiveSegment(segs: Seg[], t: number): Seg | undefined {
  for (let i = segs.length - 1; i >= 0; i--) {
    const s = segs[i];
    if (t >= s.t_start && t <= s.t_end) return s;
  }
  return undefined;
}

function mergeSegments(target: Map<number, Seg[]>, raw: TimelineSegment[]) {
  const byUid = new Map<number, Seg[]>();
  for (const s of raw) {
    const seg: Seg = {
      kind: s.kind,
      t_start: parseSimTime(s.t_start),
      t_end: parseSimTime(s.t_end),
      o_lat: s.o_lat,
      o_lng: s.o_lng,
      d_lat: s.d_lat,
      d_lng: s.d_lng,
      purpose: s.purpose,
      mode: s.mode ?? (s.kind === "dwell" ? "stay" : "car"),
      waypoints: s.waypoints?.map((w) => ({ t: parseSimTime(w.t), lat: w.lat, lng: w.lng })),
    };
    const list = byUid.get(s.uid) ?? [];
    list.push(seg);
    byUid.set(s.uid, list);
  }
  for (const [uid, newSegs] of byUid) {
    const merged = [...(target.get(uid) ?? []), ...newSegs].sort((a, b) => a.t_start - b.t_start);
    const deduped: Seg[] = [];
    for (const seg of merged) {
      const prev = deduped[deduped.length - 1];
      if (prev && prev.t_start === seg.t_start && prev.t_end === seg.t_end) continue;
      deduped.push(seg);
    }
    target.set(uid, deduped);
  }
}

function pruneSegments(target: Map<number, Seg[]>, olderThanMs: number) {
  for (const [uid, segs] of target) {
    const kept = segs.filter((s) => s.t_end >= olderThanMs);
    if (kept.length === 0) target.delete(uid);
    else target.set(uid, kept);
  }
}

// Renders the agent map for a timeline run: agents are Mapbox circle-layer
// features whose position is recomputed every animation frame (dwelling =
// fixed at the stop's coordinates, traveling = interpolated along the leg's
// road-routing waypoints when present, else linearly between the previous
// stop and this one over the leg's arrival/duration window) and pushed via
// `setData`, never through React state — a per-frame re-render would not
// keep up with thousands of agents.
export function TimelineMap({
  meta,
  expId,
  runId,
  onSelectAgent,
}: {
  meta: TimelineMeta;
  expId: string;
  runId?: string;
  onSelectAgent: (uid: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);
  const segmentsByUid = useRef<Map<number, Seg[]>>(new Map());
  const fetchSeqRef = useRef(0);
  const fetchingRef = useRef(false);
  const rafRef = useRef(0);
  const lastUiSyncWallMs = useRef(0);
  const lastRenderWallMs = useRef(0);

  const dateStartMs = meta.date_start ? parseSimTime(meta.date_start) : 0;
  const dateEndMs = meta.date_end ? parseSimTime(meta.date_end) : dateStartMs;

  const clockRef = useRef({
    simTimeMs: dateStartMs,
    playing: false,
    speedMultiplier: 60,
    lastFrameWallMs: 0,
  });
  const loadedUntilMs = useRef(dateStartMs);

  const [uiClockMs, setUiClockMs] = useState(dateStartMs);
  const [playing, setPlaying] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(60);
  const [viewHint, setViewHint] = useState<{ count: number; truncated: boolean }>({
    count: 0,
    truncated: false,
  });
  const [glLost, setGlLost] = useState(false);

  const token = import.meta.env.VITE_MAPBOX_TOKEN;

  useEffect(() => {
    if (!token || !containerRef.current || !meta.bbox) return;
    mapboxgl.accessToken = token;

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: "mapbox://styles/mapbox/light-v11",
      bounds: [
        [meta.bbox.min_lng, meta.bbox.min_lat],
        [meta.bbox.max_lng, meta.bbox.max_lat],
      ],
      fitBoundsOptions: { padding: 24 },
    });
    mapRef.current = map;

    // A GPU/driver-level failure, not a data/logic bug: when it fires, nothing
    // — not base tiles, not agent dots — can be painted to the canvas, no
    // matter how correct the fetched data is. Surface it instead of leaving a
    // silently blank map.
    map.on("webglcontextlost", () => setGlLost(true));
    map.on("webglcontextrestored", () => setGlLost(false));

    async function fetchWindow(sinceMs: number, untilMs: number) {
      // Guards against the periodic prefetch check (run every render tick)
      // firing a new overlapping request on every frame while a previous one
      // is still in flight — without this, a slow/backed-up response could
      // trigger dozens of duplicate concurrent fetches per second.
      if (fetchingRef.current) return;
      const m = mapRef.current;
      if (!m) return;
      const bounds = m.getBounds();
      if (!bounds) return;
      fetchingRef.current = true;
      const seq = ++fetchSeqRef.current;
      try {
        const payload = await fetchTimelineLegs(expId, {
          run: runId,
          since: toRequestIso(sinceMs),
          until: toRequestIso(untilMs),
          bbox: [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()],
          maxAgents: DEFAULT_MAX_AGENTS,
        });
        if (seq !== fetchSeqRef.current) return; // superseded (e.g. a seek happened meanwhile)
        mergeSegments(segmentsByUid.current, payload.segments);
        pruneSegments(segmentsByUid.current, clockRef.current.simTimeMs - RETENTION_MS);
        loadedUntilMs.current = Math.max(loadedUntilMs.current, untilMs);
        setViewHint({ count: payload.agent_count, truncated: payload.truncated });
      } catch {
        // transient viewport/time-window fetch failure — the next tick or
        // moveend retries with fresh params, nothing to surface as an error
      } finally {
        fetchingRef.current = false;
      }
    }

    let moveendTimer: ReturnType<typeof setTimeout> | undefined;
    const onMoveEnd = () => {
      if (moveendTimer) clearTimeout(moveendTimer);
      moveendTimer = setTimeout(() => {
        const until = Math.max(loadedUntilMs.current, clockRef.current.simTimeMs + LOOKAHEAD_MS);
        fetchWindow(clockRef.current.simTimeMs, until);
      }, 400);
    };

    const onLoad = () => {
      for (const mode of Object.keys(MODE_LABEL)) {
        if (!map.hasImage(`mode-${mode}`)) {
          map.addImage(`mode-${mode}`, makeModeIcon(mode), { sdf: true, pixelRatio: 2 });
        }
      }
      map.addSource("agents", { type: "geojson", data: EMPTY_FC });
      map.addLayer({
        id: "agents-symbols",
        type: "symbol",
        source: "agents",
        layout: {
          "icon-image": ["concat", "mode-", ["get", "mode"]],
          "icon-size": 0.55,
          "icon-allow-overlap": true,
          "icon-ignore-placement": true,
        },
        paint: {
          "icon-halo-color": "#ffffff",
          "icon-halo-width": 1,
          "icon-color": [
            "match",
            ["get", "purpose"],
            ...Object.entries(PURPOSE_COLOR).flatMap(([k, v]) => [k, v]),
            DEFAULT_PURPOSE_COLOR,
          ] as unknown as string,
        },
      });
      map.on("click", "agents-symbols", (e) => {
        const f = e.features?.[0];
        const uid = f?.properties?.uid;
        if (uid !== undefined && uid !== null) onSelectAgent(Number(uid));
      });
      map.on("mouseenter", "agents-symbols", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "agents-symbols", () => {
        map.getCanvas().style.cursor = "";
      });
      map.on("moveend", onMoveEnd);

      fetchWindow(clockRef.current.simTimeMs, clockRef.current.simTimeMs + LOOKAHEAD_MS);

      const tick = (nowWallMs: number) => {
        const c = clockRef.current;
        if (c.lastFrameWallMs === 0) c.lastFrameWallMs = nowWallMs;
        if (c.playing) {
          c.simTimeMs += (nowWallMs - c.lastFrameWallMs) * c.speedMultiplier;
          if (dateEndMs && c.simTimeMs >= dateEndMs) {
            c.simTimeMs = dateEndMs;
            c.playing = false;
            setPlaying(false);
          }
        }
        c.lastFrameWallMs = nowWallMs;

        // Re-uploading the GeoJSON source is real GPU work (buffer rebuild,
        // tessellation) — cap it well below the display's frame rate. Always
        // render once immediately (lastRenderWallMs.current === 0, also reset
        // by onSeek) so a fresh mount or a seek while paused isn't stuck
        // showing stale/no dots; otherwise only re-render on a cadence while
        // actually playing, since positions are static while paused.
        const dueForRender =
          lastRenderWallMs.current === 0 ||
          (c.playing && nowWallMs - lastRenderWallMs.current >= RENDER_INTERVAL_MS);
        if (!dueForRender) {
          rafRef.current = requestAnimationFrame(tick);
          return;
        }
        lastRenderWallMs.current = nowWallMs;

        const features: AgentFeatureCollection["features"] = [];
        for (const [uid, segs] of segmentsByUid.current) {
          const seg = findActiveSegment(segs, c.simTimeMs);
          if (!seg) continue;
          let lat: number;
          let lng: number;
          if (seg.kind === "dwell") {
            lat = seg.d_lat;
            lng = seg.d_lng;
          } else if (seg.waypoints && seg.waypoints.length >= 2) {
            const [a, b] = bracketWaypoints(seg.waypoints, c.simTimeMs);
            const span = b.t - a.t;
            const frac = span > 0 ? (c.simTimeMs - a.t) / span : 1;
            lat = a.lat + (b.lat - a.lat) * frac;
            lng = a.lng + (b.lng - a.lng) * frac;
          } else {
            const span = seg.t_end - seg.t_start;
            const frac = span > 0 ? (c.simTimeMs - seg.t_start) / span : 1;
            lat = seg.o_lat + (seg.d_lat - seg.o_lat) * frac;
            lng = seg.o_lng + (seg.d_lng - seg.o_lng) * frac;
          }
          features.push({
            type: "Feature",
            geometry: { type: "Point", coordinates: [lng, lat] },
            properties: { uid, purpose: seg.purpose, mode: seg.mode },
          });
        }
        const source = mapRef.current?.getSource("agents") as mapboxgl.GeoJSONSource | undefined;
        source?.setData({ type: "FeatureCollection", features });

        if (c.simTimeMs >= loadedUntilMs.current - REFETCH_MARGIN_MS) {
          fetchWindow(loadedUntilMs.current, loadedUntilMs.current + LOOKAHEAD_MS);
        }
        if (nowWallMs - lastUiSyncWallMs.current > 250) {
          setUiClockMs(c.simTimeMs);
          lastUiSyncWallMs.current = nowWallMs;
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    };
    map.on("load", onLoad);

    return () => {
      if (moveendTimer) clearTimeout(moveendTimer);
      cancelAnimationFrame(rafRef.current);
      map.remove();
      mapRef.current = null;
    };
    // meta/expId/runId are fixed props for the lifetime of a mounted Timeline
    // page (a new id/run navigates to a fresh page instance), so this setup
    // effect intentionally runs once per mount rather than reacting to them.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  function togglePlay() {
    clockRef.current.playing = !clockRef.current.playing;
    clockRef.current.lastFrameWallMs = 0;
    setPlaying(clockRef.current.playing);
  }

  function setSpeed(mult: number) {
    clockRef.current.speedMultiplier = mult;
    setSpeedMultiplier(mult);
  }

  function onSeek(ms: number) {
    clockRef.current.simTimeMs = ms;
    clockRef.current.lastFrameWallMs = 0;
    segmentsByUid.current.clear();
    loadedUntilMs.current = ms;
    setUiClockMs(ms);
    // Force the render-throttled tick loop to repaint on its very next frame
    // even if playback is paused, so a seek is never stuck showing stale dots.
    lastRenderWallMs.current = 0;
    const m = mapRef.current;
    const bounds = m?.getBounds();
    if (!bounds) return;
    const seq = ++fetchSeqRef.current;
    fetchTimelineLegs(expId, {
      run: runId,
      since: toRequestIso(ms),
      until: toRequestIso(ms + LOOKAHEAD_MS),
      bbox: [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()],
      maxAgents: DEFAULT_MAX_AGENTS,
    })
      .then((payload) => {
        if (seq !== fetchSeqRef.current) return;
        mergeSegments(segmentsByUid.current, payload.segments);
        loadedUntilMs.current = ms + LOOKAHEAD_MS;
        setViewHint({ count: payload.agent_count, truncated: payload.truncated });
      })
      .catch(() => {});
  }

  if (!token) {
    return (
      <div className="state">
        Set <code>VITE_MAPBOX_TOKEN</code> in <code>web/frontend/.env.local</code> to use the timeline
        view — see <code>web/README.md</code>.
      </div>
    );
  }

  return (
    <div className="timeline-main">
      <div className="timeline-controls">
        <button className={`btn ${playing ? "btn-primary" : "btn-secondary"}`} onClick={togglePlay}>
          {playing ? "Pause" : "Play"}
        </button>
        {SPEED_OPTIONS.map((s) => (
          <button
            key={s}
            className={`btn ${s === speedMultiplier ? "btn-primary" : "btn-secondary"}`}
            style={{ padding: "4px 10px", fontSize: 13 }}
            onClick={() => setSpeed(s)}
          >
            {s}x
          </button>
        ))}
        <span className="timeline-clock">{formatClock(uiClockMs)}</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>
            {viewHint.count} agent{viewHint.count === 1 ? "" : "s"} in view
            {viewHint.truncated ? " (capped)" : ""}
          </span>
          <div className="timeline-legend">
            {Object.entries(PURPOSE_COLOR).map(([purpose, color]) => (
              <span key={purpose} className="timeline-legend-item">
                <i style={{ background: color }} />
                {purpose}
              </span>
            ))}
          </div>
          <div className="timeline-legend">
            {Object.entries(MODE_LABEL).map(([mode, label]) => (
              <span key={mode} className="timeline-legend-item">
                <i className={`timeline-mode-swatch timeline-mode-${mode}`} />
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>
      <input
        className="timeline-scrub"
        type="range"
        min={dateStartMs}
        max={dateEndMs}
        step={60000}
        value={uiClockMs}
        onChange={(e) => onSeek(Number(e.target.value))}
      />
      <div className="timeline-map-wrap">
        <div ref={containerRef} className="timeline-map" />
        {glLost && (
          <div className="timeline-gl-lost">
            WebGL context lost — this is a browser/GPU rendering failure, not a data problem (the agent
            data itself loads correctly). Try reloading the page, a different browser, or checking that
            hardware acceleration / WebGL is enabled and working (e.g. Firefox's <code>about:support</code>{" "}
            → Graphics → check for "WebGL2 Driver Renderer" without a blocklist warning).
          </div>
        )}
      </div>
    </div>
  );
}

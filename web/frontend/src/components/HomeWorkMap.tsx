import { useMemo, useState } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import type { Layer } from "leaflet";
import type { Feature } from "geojson";
import "leaflet/dist/leaflet.css";
import type { HomeWorkMapBlock, HomeWorkPanel } from "../api";

function Panel({ title, panel }: { title: string; panel: HomeWorkPanel }) {
  const resolutions = Object.keys(panel.layers).sort();
  const [res, setRes] = useState(resolutions[resolutions.length - 1] ?? resolutions[0]);
  const center: [number, number] = panel.center ? [panel.center[1], panel.center[0]] : [0, 0];
  const geojson = useMemo(() => panel.layers[res], [panel, res]);

  const style = (feature?: Feature) => {
    const color = (feature?.properties as { color?: string })?.color ?? "#cccccc";
    return { color: "#555", weight: 0.4, fillColor: color, fillOpacity: 0.7 };
  };

  const onEach = (feature: Feature, layer: Layer) => {
    const p = feature.properties as { area?: string; agent_count?: number; agent_pct?: number };
    layer.bindTooltip(
      `cell ${p.area}<br/>${p.agent_count} agents (${p.agent_pct?.toFixed(1)}%)`,
    );
  };

  return (
    <div className="hw-panel">
      <h5>{title}</h5>
      <div className="stvd-controls">
        <span style={{ fontSize: 13, color: "var(--muted)" }}>H3 resolution</span>
        {resolutions.map((r) => (
          <button
            key={r}
            className={`btn ${r === res ? "btn-primary" : "btn-secondary"}`}
            style={{ padding: "4px 14px", fontSize: 13 }}
            onClick={() => setRes(r)}
          >
            {r}
          </button>
        ))}
      </div>
      <MapContainer center={center} zoom={10} className="hw-map" scrollWheelZoom>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution="&copy; OpenStreetMap &copy; CARTO"
        />
        {geojson && <GeoJSON key={res} data={geojson as never} style={style} onEachFeature={onEach} />}
      </MapContainer>
      <div className="hw-legend">
        {panel.colors.map((c, i) => (
          <span key={i} style={{ background: c }} />
        ))}
      </div>
      <span style={{ fontSize: 12, color: "var(--muted)" }}>fewer agents → more agents</span>
    </div>
  );
}

export function HomeWorkMap({
  block,
  syntheticLabel,
  observedLabel,
}: {
  block: HomeWorkMapBlock;
  syntheticLabel: string;
  observedLabel?: string;
}) {
  return (
    <div className="hw-panels">
      <Panel title={syntheticLabel} panel={block.synthetic} />
      {block.real && <Panel title={observedLabel ?? "observed"} panel={block.real} />}
    </div>
  );
}

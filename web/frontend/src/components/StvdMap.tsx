import { useMemo, useState } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import type { Layer } from "leaflet";
import type { Feature } from "geojson";
import "leaflet/dist/leaflet.css";
import type { StvdBlock } from "../api";

// STVD H3 choropleth: bivariate (volume diff × peak shift) colour is precomputed
// per feature in the backend, so we just paint each polygon with its colour.
export function StvdMap({ block }: { block: StvdBlock }) {
  const resolutions = Object.keys(block.layers).sort();
  const defaultResolution = resolutions.includes("7")
    ? "7"
    : (resolutions[resolutions.length - 1] ?? resolutions[0]);
  const [res, setRes] = useState(defaultResolution);
  const center: [number, number] = block.center
    ? [block.center[1], block.center[0]]
    : [0, 0];

  const geojson = useMemo(() => block.layers[res], [block, res]);

  const style = (feature?: Feature) => {
    const color = (feature?.properties as { color?: string })?.color ?? "#cccccc";
    return { color: "#555", weight: 0.4, fillColor: color, fillOpacity: 0.6 };
  };

  const onEach = (feature: Feature, layer: Layer) => {
    const p = feature.properties as {
      area?: string;
      volume_diff_pct?: number;
      peak_shift_hours?: number;
    };
    layer.bindTooltip(
      `cell ${p.area}<br/>volume Δ ${p.volume_diff_pct?.toFixed(1)}%<br/>peak shift ${p.peak_shift_hours?.toFixed(1)} h`,
    );
  };

  return (
    <div>
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
        <div style={{ marginLeft: "auto", display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>synthetic lower → higher volume</span>
          <div className="stvd-legend">
            {block.colors.flat().map((c, i) => (
              <span key={i} style={{ background: c }} />
            ))}
          </div>
        </div>
      </div>
      <MapContainer center={center} zoom={10} className="stvd-map" scrollWheelZoom>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap &copy; CARTO'
        />
        <GeoJSON key={res} data={geojson as never} style={style} onEachFeature={onEach} />
      </MapContainer>
    </div>
  );
}

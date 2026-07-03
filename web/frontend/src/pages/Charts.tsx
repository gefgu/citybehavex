import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { EChartsOption } from "echarts";
import { fetchCharts, type ChartPayload } from "../api";
import { EChart } from "../charts/EChart";
import { SocialNetworkGraph } from "../components/SocialNetworkGraph";
import { StvdMap } from "../components/StvdMap";
import {
  dailyActivityOption,
  ecdfOption,
  lawOption,
  motifOption,
  profileBoxOption,
  profileScatterOption,
  purposeOption,
  transitionOption,
} from "../charts/builders";

function ChartCard({
  title,
  option,
  wide = false,
  subtitle,
}: {
  title: string;
  option: EChartsOption;
  wide?: boolean;
  subtitle?: string;
}) {
  return (
    <div className={`chart-card${wide ? " wide" : ""}`}>
      <h4>{title}</h4>
      <EChart option={option} />
      {subtitle && <div className="fit-params">{subtitle}</div>}
    </div>
  );
}

const ECDF_TITLES: Record<string, string> = {
  jump_lengths: "Jump length",
  visits_per_user: "Visits per user",
  radius_of_gyration: "Radius of gyration",
  dwell_time: "Dwell time",
  trip_duration: "Trip duration",
};

export function Charts() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const run = params.get("run") ?? undefined;
  const [payload, setPayload] = useState<ChartPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPayload(null);
    setError(null);
    fetchCharts(id, run).then(setPayload).catch((e) => setError(String(e)));
  }, [id, run]);

  const fitSubtitle = useMemo(
    () => (fits: { label: string; params: Record<string, number> }[]) =>
      fits
        .map(
          (f) =>
            `${f.label}: ${Object.entries(f.params)
              .map(([k, v]) => `${k}=${Number(v).toPrecision(3)}`)
              .join(", ")}`,
        )
        .join("  ·  "),
    [],
  );

  if (error) return <div className="state">Failed to load charts: {error}</div>;
  if (!payload)
    return <div className="state">Building comparison… (first load can take a while)</div>;

  const { metrics } = payload;

  return (
    <>
      <h1 style={{ margin: "48px 0 4px" }}>
        {payload.labels.observed} <span style={{ color: "var(--muted)" }}>vs synthetic</span>
      </h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        <Link to="/experiments">experiments</Link> / {id} · run{" "}
        <code>{payload.run_id}</code>
      </p>

      {payload.warnings.length > 0 && (
        <div className="warnings">
          {payload.warnings.length} section(s) skipped: {payload.warnings.join("; ")}
        </div>
      )}

      {/* metrics */}
      <div className="section-header">Metrics</div>
      <div className="metric-tables">
        <div>
          <h4>Wasserstein distances</h4>
          <table className="metrics">
            <tbody>
              {metrics.wasserstein.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td className="value">{m.value.toFixed(4)}</td>
                  <td className="unit">{m.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h4>Jensen–Shannon divergences</h4>
          <table className="metrics">
            <tbody>
              {metrics.jsd.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td className="value">{m.value.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h4>Common Part of Commuters</h4>
          <table className="metrics">
            <tbody>
              {metrics.cpc.map((m) => (
                <tr key={m.resolution}>
                  <td>H3 {m.resolution}</td>
                  <td className="value">{m.value.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ECDFs */}
      <div className="section-header">Distribution comparisons</div>
      <div className="chart-grid">
        {Object.entries(payload.ecdf).map(([key, block]) => (
          <ChartCard key={key} title={`${ECDF_TITLES[key] ?? key} ECDF`} option={ecdfOption(block)} />
        ))}
      </div>

      {/* mobility laws */}
      {payload.mobility_laws && (
        <>
          <div className="section-header">Mobility laws</div>
          <div className="chart-grid">
            {Object.entries(payload.mobility_laws).map(([key, block]) => (
              <ChartCard
                key={key}
                title={block.title}
                option={lawOption(block)}
                subtitle={fitSubtitle(block.fits)}
              />
            ))}
          </div>
        </>
      )}

      {/* activity */}
      {payload.activity && (
        <>
          <div className="section-header">Activity comparison</div>
          <div className="chart-grid">
            <ChartCard title="Visit purpose comparison" option={purposeOption(payload.activity.purpose)} wide />
            <ChartCard
              title="Activity transition difference"
              option={transitionOption(payload.activity.transition_difference)}
            />
            {payload.activity.daily_activity_difference && (
              <ChartCard
                title="Daily activity difference"
                option={dailyActivityOption(payload.activity.daily_activity_difference)}
              />
            )}
          </div>
        </>
      )}

      {/* profiles */}
      {payload.profiles && (
        <>
          <div className="section-header">Mobility profiles</div>
          <div className="chart-grid">
            <ChartCard
              title="Intermittency vs degree of return"
              option={profileScatterOption(payload.profiles)}
              wide
            />
            {payload.profiles.metrics.map((metric) => (
              <ChartCard
                key={metric}
                title={metric[0].toUpperCase() + metric.slice(1)}
                option={profileBoxOption(payload.profiles!, metric)}
              />
            ))}
          </div>
        </>
      )}

      {/* motifs */}
      {payload.motifs && (
        <>
          <div className="section-header">Daily motifs</div>
          <div className="chart-grid">
            <ChartCard title="Motif literature comparison" option={motifOption(payload.motifs)} wide />
          </div>
        </>
      )}

      {/* STVD */}
      {payload.stvd && (
        <>
          <div className="section-header">Spatial-temporal volume difference</div>
          <StvdMap block={payload.stvd} />
        </>
      )}

      <div className="section-header">Social network</div>
      {payload.social_network ? (
        <SocialNetworkGraph block={payload.social_network} />
      ) : (
        <div className="network-empty">
          No social network sidecar found for this run. Re-run the simulation with the latest code,
          then refresh the chart payload.
        </div>
      )}

      <div style={{ height: 96 }} />
    </>
  );
}

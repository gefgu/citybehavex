import type { AgentEncounter, AgentTrip } from "../api";

export type TimelineDetailSelection =
  | {
      kind: "trip";
      agentUid: number;
      agentName: string | null;
      trip: AgentTrip;
    }
  | {
      kind: "encounter";
      agentUid: number;
      agentName: string | null;
      encounter: AgentEncounter;
    };

function fmtDateTime(s: string | null): string {
  return s ? s.replace("T", " ") : "not available";
}

function fmtMinutes(value: number | null | undefined): string {
  return value == null ? "not available" : `${value.toFixed(0)} min`;
}

function fmtCoords(lat: number | null | undefined, lng: number | null | undefined): string {
  if (lat == null || lng == null) return "not available";
  return `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
}

function fmtPurpose(purpose: string | null | undefined, category?: string | null): string {
  if (!purpose) return "not available";
  return category ? `${purpose} · ${category}` : purpose;
}

function fmtActivity(name: string | null, id: number | null): string {
  if (name) return id == null ? name : `${name} (#${id})`;
  return id == null ? "not available" : `activity #${id}`;
}

export function TimelineDetailPanel({ selection }: { selection: TimelineDetailSelection | null }) {
  if (!selection) {
    return (
      <div className="timeline-detail-panel timeline-detail-empty">
        Select a trip or encounter to inspect activity timing.
      </div>
    );
  }

  if (selection.kind === "trip") {
    const { trip } = selection;
    return (
      <div className="timeline-detail-panel">
        <div className="timeline-detail-header">
          <div>
            <div className="section-header">Trip activity detail</div>
            <h3>{selection.agentName ?? `Agent ${selection.agentUid}`}</h3>
          </div>
          <span className="timeline-detail-pill">agent {selection.agentUid}</span>
        </div>

        <p className="agent-narrative">
          {trip.activity_description ?? "No micro-activity description is available for this stop."}
        </p>

        <table className="agent-table timeline-detail-table">
          <tbody>
            <tr>
              <td>arrival</td>
              <td>{fmtDateTime(trip.arrival)}</td>
            </tr>
            <tr>
              <td>departure</td>
              <td>{fmtDateTime(trip.departure)}</td>
            </tr>
            <tr>
              <td>purpose</td>
              <td>{fmtPurpose(trip.purpose, trip.category)}</td>
            </tr>
            <tr>
              <td>micro-activity</td>
              <td>{fmtActivity(trip.activity_name, trip.activity)}</td>
            </tr>
            <tr>
              <td>dwell time</td>
              <td>{fmtMinutes(trip.dwell_minutes)}</td>
            </tr>
            <tr>
              <td>trip time</td>
              <td>{fmtMinutes(trip.trip_duration_minutes)}</td>
            </tr>
            <tr>
              <td>location</td>
              <td>{fmtCoords(trip.lat, trip.lng)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    );
  }

  const { encounter } = selection;
  const contactName = encounter.contact_profile?.name ?? `Agent ${encounter.contact_uid}`;

  return (
    <div className="timeline-detail-panel">
      <div className="timeline-detail-header">
        <div>
          <div className="section-header">Social encounter detail</div>
          <h3>{contactName}</h3>
        </div>
        <span className="timeline-detail-pill">contact {encounter.contact_uid}</span>
      </div>

      {encounter.contact_narrative && <p className="agent-narrative">{encounter.contact_narrative}</p>}
      {encounter.location_warning && <div className="warnings">{encounter.location_warning}</div>}

      <table className="agent-table timeline-detail-table">
        <tbody>
          <tr>
            <td>encounter time</td>
            <td>{fmtDateTime(encounter.ts)}</td>
          </tr>
          <tr>
            <td>tile</td>
            <td>{encounter.tile}</td>
          </tr>
          <tr>
            <td>location</td>
            <td>{fmtCoords(encounter.lat, encounter.lng)}</td>
          </tr>
          <tr>
            <td>stop window</td>
            <td>
              {fmtDateTime(encounter.stop_arrival)} to {fmtDateTime(encounter.stop_departure)}
            </td>
          </tr>
          <tr>
            <td>purpose</td>
            <td>{fmtPurpose(encounter.purpose, encounter.category)}</td>
          </tr>
          <tr>
            <td>micro-activity</td>
            <td>{fmtActivity(encounter.activity_name, encounter.activity)}</td>
          </tr>
          <tr>
            <td>dwell time</td>
            <td>{fmtMinutes(encounter.dwell_minutes)}</td>
          </tr>
        </tbody>
      </table>

      {encounter.activity_description && (
        <p className="agent-narrative">{encounter.activity_description}</p>
      )}
    </div>
  );
}

import { useState } from "react";
import { summarizeAnomaly } from "../api";

export default function AnomalyPanel({ anomalies, onUpdated }) {
  const [loadingId, setLoadingId] = useState(null);

  async function handleSummarize(id) {
    setLoadingId(id);
    try {
      await summarizeAnomaly(id);
      onUpdated?.();
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingId(null);
    }
  }

  if (!anomalies.length) {
    return <div className="empty-state">No anomalies detected.</div>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Track ID</th>
          <th>Reasons</th>
          <th>Incident summary</th>
          <th>Time</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {anomalies.map((a) => (
          <tr key={a.id}>
            <td>#{a.track_id}</td>
            <td>
              {a.reasons.map((r) => (
                <span key={r} className="badge anomaly" style={{ marginRight: 4 }}>
                  {r}
                </span>
              ))}
            </td>
            <td style={{ maxWidth: 320 }}>{a.incident_summary || "—"}</td>
            <td>{new Date(a.timestamp).toLocaleTimeString()}</td>
            <td>
              {!a.incident_summary && (
                <button
                  className="btn"
                  onClick={() => handleSummarize(a.id)}
                  disabled={loadingId === a.id}
                >
                  {loadingId === a.id ? "Summarizing…" : "Summarize"}
                </button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

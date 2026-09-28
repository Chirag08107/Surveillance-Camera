import { useEffect, useState, useCallback } from "react";
import {
  getLatestDetections,
  getZoneEvents,
  getAnomalies,
  getIdentities,
  connectLiveSocket,
} from "./api";
import DetectionsTable from "./components/DetectionsTable";
import ZoneEventsPanel from "./components/ZoneEventsPanel";
import AnomalyPanel from "./components/AnomalyPanel";
import IdentitiesPanel from "./components/IdentitiesPanel";
import AnalyticsPanel from "./components/AnalyticsPanel";

const TABS = [
  { id: "live", label: "Live Feed" },
  { id: "events", label: "Zone Events" },
  { id: "anomalies", label: "Anomalies" },
  { id: "identities", label: "Identities" },
  { id: "analytics", label: "Analytics" },
];

export default function App() {
  const [tab, setTab] = useState("live");
  const [connected, setConnected] = useState(false);
  const [detections, setDetections] = useState([]);
  const [zoneEvents, setZoneEvents] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [identities, setIdentities] = useState([]);

  const refreshAll = useCallback(async () => {
    try {
      const [d, z, a, i] = await Promise.all([
        getLatestDetections(),
        getZoneEvents(),
        getAnomalies(),
        getIdentities(),
      ]);
      setDetections(d);
      setZoneEvents(z);
      setAnomalies(a);
      setIdentities(i);
    } catch (e) {
      // Backend not reachable yet - dashboard still renders, just empty.
      console.warn("Backend unreachable:", e.message);
    }
  }, []);

  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 5000); // periodic backstop refresh
    return () => clearInterval(interval);
  }, [refreshAll]);

  useEffect(() => {
    const close = connectLiveSocket((msg) => {
      if (msg.type === "frame") {
        if (msg.detections?.length) {
          setDetections((prev) => {
            const byId = new Map(prev.map((d) => [d.track_id, d]));
            msg.detections.forEach((d) => byId.set(d.track_id, d));
            return Array.from(byId.values());
          });
        }
        if (msg.zone_events?.length) {
          setZoneEvents((prev) => [
            ...msg.zone_events.map((e, idx) => ({ ...e, id: `live-${Date.now()}-${idx}`, timestamp: msg.timestamp })),
            ...prev,
          ].slice(0, 200));
        }
        if (msg.anomaly_events?.length) {
          setAnomalies((prev) => [
            ...msg.anomaly_events.map((a, idx) => ({ ...a, id: `live-${Date.now()}-${idx}`, timestamp: msg.timestamp, incident_summary: null })),
            ...prev,
          ].slice(0, 200));
        }
      }
    }, setConnected);

    return close;
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Sentinel</h1>
        <div className="subtitle">Behavioral Intelligence</div>

        {TABS.map((t) => (
          <div
            key={t.id}
            className={`nav-item ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </div>
        ))}
      </aside>

      <main className="main">
        <div className="topbar">
          <h2 style={{ margin: 0, fontWeight: 500, fontSize: 18 }}>
            {TABS.find((t) => t.id === tab)?.label}
          </h2>
          <div className="conn">
            <span className={`status-dot ${connected ? "online" : "offline"}`} />
            {connected ? "Live feed connected" : "Live feed disconnected — retrying"}
          </div>
        </div>

        <div className="grid">
          <div className="card">
            <h3>Active tracks</h3>
            <div className="metric">{detections.length}</div>
          </div>
          <div className="card">
            <h3>In restricted zone</h3>
            <div className="metric">{detections.filter((d) => d.in_restricted_zone).length}</div>
          </div>
          <div className="card">
            <h3>Open anomalies</h3>
            <div className={`metric ${anomalies.length ? "alert" : ""}`}>{anomalies.length}</div>
          </div>
          <div className="card">
            <h3>Enrolled identities</h3>
            <div className="metric">{identities.length}</div>
          </div>
        </div>

        {tab === "live" && (
          <div className="panel">
            <h2>Currently Tracked</h2>
            <DetectionsTable detections={detections} />
          </div>
        )}

        {tab === "events" && (
          <div className="panel">
            <h2>Restricted Zone Entries / Exits</h2>
            <ZoneEventsPanel events={zoneEvents} />
          </div>
        )}

        {tab === "anomalies" && (
          <div className="panel">
            <h2>Flagged Behavior</h2>
            <AnomalyPanel anomalies={anomalies} onUpdated={refreshAll} />
          </div>
        )}

        {tab === "identities" && (
          <div className="panel">
            <h2>Enrolled Identities (Consent-Based)</h2>
            <IdentitiesPanel identities={identities} onUpdated={refreshAll} />
          </div>
        )}

        {tab === "analytics" && (
          <div className="panel">
            <h2>Behavior Distribution</h2>
            <AnalyticsPanel detections={detections} />
          </div>
        )}
      </main>
    </div>
  );
}

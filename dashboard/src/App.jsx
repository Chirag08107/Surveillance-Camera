
import { useCallback, useEffect, useState } from "react";

import {
  getLatestDetections,
  getZoneEvents,
  getAnomalies,
  connectLiveSocket,
} from "./api";

import UploadVideo from "./components/UploadVideo";
import LiveMonitoring from "./components/LiveMonitoring";
import AnomalyPanel from "./components/AnomalyPanel";
import AnalyticsPanel from "./components/AnalyticsPanel";

const TABS = [
  { id: "upload", label: "Upload Video", icon: "↑" },
  { id: "live", label: "Live Monitoring", icon: "◉" },
  { id: "anomalies", label: "Anomalies", icon: "⚠" },
  { id: "analytics", label: "Analytics", icon: "▥" },
];

export default function App() {
  const [tab, setTab] = useState("upload");
  const [connected, setConnected] = useState(false);
  const [detections, setDetections] = useState([]);
  const [zoneEvents, setZoneEvents] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [session, setSession] = useState(null);

  // A session remains visible after processing finishes.
  const [monitoring, setMonitoring] = useState(false);
  const [processing, setProcessing] = useState(false);

  const refreshAll = useCallback(async () => {
    try {
      const [d, z, a] = await Promise.all([
        getLatestDetections(),
        getZoneEvents(),
        getAnomalies(),
      ]);

      setDetections(Array.isArray(d) ? d : []);
      setZoneEvents(Array.isArray(z) ? z : []);
      setAnomalies(Array.isArray(a) ? a : []);
    } catch (error) {
      console.warn(
        "Unable to refresh dashboard:",
        error.message
      );
    }
  }, []);

  // Load database history when no uploaded session is active.
  useEffect(() => {
    if (monitoring) return undefined;

    refreshAll();

    const interval = setInterval(refreshAll, 5000);

    return () => clearInterval(interval);
  }, [refreshAll, monitoring]);

  useEffect(() => {
    const closeSocket = connectLiveSocket((message) => {
      if (message?.type !== "frame") return;

      if (
        Array.isArray(message.detections) &&
        message.detections.length
      ) {
        setDetections((previous) => {
          const byTrack = new Map(
            previous.map((item) => [
              String(item.track_id),
              item,
            ])
          );

          message.detections.forEach((item) => {
            if (
              item.track_id === undefined ||
              item.track_id === null
            ) {
              return;
            }

            const key = String(item.track_id);

            byTrack.set(key, {
              ...byTrack.get(key),
              ...item,
              _receivedAt: Date.now(),
            });
          });

          const now = Date.now();

          return Array.from(byTrack.values()).filter(
            (item) =>
              !item._receivedAt ||
              now - item._receivedAt < 30000
          );
        });
      }

      // Zone events remain available internally for Live Monitoring.
      if (
        Array.isArray(message.zone_events) &&
        message.zone_events.length
      ) {
        setZoneEvents((previous) => [
          ...message.zone_events.map((event, index) => ({
            ...event,
            id: `live-zone-${Date.now()}-${index}`,
            timestamp: event.timestamp || message.timestamp,
            _sessionEvent: true,
          })),
          ...previous,
        ].slice(0, 200));
      }

      if (
        Array.isArray(message.anomaly_events) &&
        message.anomaly_events.length
      ) {
        setAnomalies((previous) => [
          ...message.anomaly_events.map((event, index) => ({
            ...event,
            id: `live-anomaly-${Date.now()}-${index}`,
            timestamp: event.timestamp || message.timestamp,
            incident_summary: null,
            _sessionEvent: true,
          })),
          ...previous,
        ].slice(0, 200));
      }
    }, setConnected);

    return closeSocket;
  }, []);

  const people = detections.filter(
    (item) => item.class_name?.toLowerCase() === "person"
  );

  const zoneWarnings = people.filter(
    (item) => item.in_restricted_zone
  ).length;

  const flaggedPeople = people.filter(
    (item) => item.behavior?.toLowerCase() === "anomaly"
  ).length;

  function handleUploadComplete(uploadedSession) {
    // Start the new session with no previous-session results.
    setSession(uploadedSession);
    setDetections([]);
    setZoneEvents([]);
    setAnomalies([]);
    setMonitoring(true);
    setProcessing(true);
    setTab("live");
  }

  const handleProcessingStopped = useCallback(() => {
    setProcessing(false);
  }, []);

  const handleSessionClosed = useCallback(() => {
    setProcessing(false);
    setMonitoring(false);
    setSession(null);
    setDetections([]);
    setZoneEvents([]);
    setAnomalies([]);
    setTab("upload");
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <span />
            <span />
            <span />
          </div>

          <div>
            <div className="brand-name">SENTINEL</div>
            <div className="brand-subtitle">
              Behavioral Intelligence
            </div>
          </div>
        </div>

        <div className="system-status">
          <span
            className={`system-dot ${
              connected ? "online" : "offline"
            }`}
          />

          <div>
            <div className="system-status-title">
              {connected ? "SYSTEM ONLINE" : "CONNECTING"}
            </div>

            <div className="system-status-subtitle">
              {connected
                ? "Live telemetry connected"
                : "Connecting to backend"}
            </div>
          </div>
        </div>

        <nav className="navigation">
          <div className="nav-label">SURVEILLANCE</div>

          {TABS.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${
                tab === item.id ? "active" : ""
              }`}
              onClick={() => setTab(item.id)}
              type="button"
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>

              {item.id === "anomalies" &&
                anomalies.length > 0 && (
                  <span className="nav-badge">
                    {anomalies.length}
                  </span>
                )}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-footer-line" />
          <div className="sidebar-footer-text">
            SENTINEL ENGINE
          </div>
          <div className="sidebar-footer-version">
            AI Surveillance Platform · v1.0
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="page-kicker">
              SURVEILLANCE CONTROL
            </div>
            <h1>
              {TABS.find((item) => item.id === tab)?.label}
            </h1>
          </div>

          <div className="connection-indicator">
            <span
              className={`connection-dot ${
                connected ? "online" : "offline"
              }`}
            />
            {connected
              ? "Live feed connected"
              : "Live feed disconnected"}
          </div>
        </header>

        <section className="metrics-grid">
          <MetricCard
            label="TRACKED PEOPLE"
            value={monitoring ? people.length : 0}
            description="People in the current monitoring view"
            tone="blue"
            icon="◉"
          />

          <MetricCard
            label="ZONE WARNINGS"
            value={monitoring ? zoneWarnings : 0}
            description="People currently inside a restricted zone"
            tone="yellow"
            icon="⚠"
          />

          <MetricCard
            label="ANOMALY FLAGS"
            value={monitoring ? flaggedPeople : 0}
            description="Tracks flagged by the behavior classifier"
            tone="red"
            icon="!"
          />

          <MetricCard
            label="SESSION"
            value={
              processing
                ? "LIVE"
                : monitoring
                  ? "COMPLETE"
                  : "IDLE"
            }
            description={
              session?.video_name || "Upload a video to begin"
            }
            tone="purple"
            icon="▣"
          />
        </section>

        {tab === "upload" && (
          <UploadVideo
            onUploadComplete={handleUploadComplete}
          />
        )}

        {tab === "live" && (
          <LiveMonitoring
            detections={monitoring ? detections : []}
            zoneEvents={monitoring ? zoneEvents : []}
            anomalies={monitoring ? anomalies : []}
            session={session}
            monitoring={monitoring}
            processing={processing}
            onStopped={handleProcessingStopped}
            onCloseSession={handleSessionClosed}
          />
        )}

        {tab === "anomalies" && (
          <section className="page-panel">
            <PanelHeading
              kicker="INCIDENT REVIEW"
              title="Detected Anomalies"
            />
            <AnomalyPanel
              anomalies={anomalies}
              onUpdated={refreshAll}
            />
          </section>
        )}

        {tab === "analytics" && (
          <section className="page-panel">
            <PanelHeading
              kicker="SYSTEM INSIGHTS"
              title="Behavior Analytics"
            />
            <AnalyticsPanel
              detections={detections}
              anomalies={anomalies}
            />
          </section>
        )}
      </main>
    </div>
  );
}

function MetricCard({
  label,
  value,
  description,
  tone,
  icon,
}) {
  return (
    <div className={`metric-card ${tone}-card`}>
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        <span className={`metric-icon ${tone}`}>
          {icon}
        </span>
      </div>

      <div className="metric-value">{value}</div>
      <div className="metric-description">
        {description}
      </div>
    </div>
  );
}

function PanelHeading({ kicker, title }) {
  return (
    <div className="panel-heading">
      <div>
        <div className="panel-kicker">{kicker}</div>
        <h2>{title}</h2>
      </div>
    </div>
  );
}

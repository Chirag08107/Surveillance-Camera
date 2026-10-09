import { useEffect, useMemo, useState } from "react";
import { getMonitoringStatus, getVideoUrl, stopMonitoring } from "../api";

function formatTime(timestamp) {
  if (!timestamp) return "--:--:--";
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime())
    ? "--:--:--"
    : date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
}

function getRiskState(person) {
  if (person.behavior === "Anomaly") {
    return {
      label: "ANOMALY FLAG",
      tone: "danger",
      description: "Behavior classifier flagged this track for human review",
    };
  }

  if (person.in_restricted_zone) {
    return {
      label: "WARNING",
      tone: "warning",
      description: "Person is inside the configured restricted zone",
    };
  }

  return {
    label: "NORMAL",
    tone: "normal",
    description: "No current warning condition reported",
  };
}

export default function LiveMonitoring({
  detections = [],
  zoneEvents = [],
  anomalies = [],
  session,
  monitoring,
  processing,
  onStopped,
  onCloseSession,
}) {
  const [stopping, setStopping] = useState(false);
  const [message, setMessage] = useState("");

  const people = useMemo(
    () =>
      detections
        .filter((item) => item.class_name?.toLowerCase() === "person")
        .sort((a, b) => {
          const priority = (person) => {
            if (person.behavior === "Anomaly") return 2;
            if (person.in_restricted_zone) return 1;
            return 0;
          };
          return priority(b) - priority(a);
        }),
    [detections]
  );

  const warnings = people.filter((person) => person.in_restricted_zone).length;
  const flagged = people.filter((person) => person.behavior === "Anomaly").length;

  const recentEvents = useMemo(() => {
    const zone = zoneEvents.map((event, index) => ({
      id: `zone-${event.id ?? `${event.track_id}-${index}`}`,
      timestamp: event.timestamp,
      track_id: event.track_id,
      kind: "ZONE EVENT",
      tone: "warning",
      description: event.event_type || "Zone event recorded",
    }));

    const anomaly = anomalies.map((event, index) => ({
      id: `anomaly-${event.id ?? `${event.track_id}-${index}`}`,
      timestamp: event.timestamp,
      track_id: event.track_id,
      kind: "ANOMALY FLAG",
      tone: "danger",
      description: Array.isArray(event.reasons)
        ? event.reasons.join(", ")
        : "Behavioral alert requires review",
    }));

    return [...zone, ...anomaly]
      .sort(
        (a, b) =>
          new Date(b.timestamp || 0).getTime() -
          new Date(a.timestamp || 0).getTime()
      )
      .slice(0, 8);
  }, [zoneEvents, anomalies]);

  // Poll the backend until the pipeline completes. Keep the results visible
  // when it finishes; only change the processing indicator and stop button.
  useEffect(() => {
    if (!session?.session_id || !processing) return undefined;

    let active = true;

    async function checkStatus() {
      try {
        const status = await getMonitoringStatus();
        if (!active) return;

        if (!status.running) {
          setMessage(
            "Analysis finished or the pipeline stopped. Results from this session remain visible."
          );
          onStopped?.();
        } else {
          setMessage("");
        }
      } catch {
        if (active) {
          setMessage("Unable to retrieve monitoring status from the backend.");
        }
      }
    }

    checkStatus();
    const timer = setInterval(checkStatus, 3000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [session?.session_id, processing, onStopped]);

  async function handleStop() {
    setStopping(true);
    setMessage("");

    try {
      await stopMonitoring();
      onStopped?.();
      setMessage("Monitoring has been stopped. Current results are retained.");
    } catch (error) {
      setMessage(error.message || "Could not stop monitoring.");
    } finally {
      setStopping(false);
    }
  }

  function handleCloseSession() {
    onCloseSession?.();
    setMessage("");
  }

  const videoUrl = getVideoUrl(session?.video_url);

  return (
    <div className="monitoring-page">
      <section className="monitoring-control-card">
        <div className="source-section">
          <div className="panel-kicker">CURRENT SESSION</div>
          <div className="source-title-row">
            <div>
              <h2>{session?.video_name || "No video selected"}</h2>
              <p>
                {processing
                  ? "Uploaded video · AI pipeline processing"
                  : monitoring
                    ? "Analysis complete · Session results retained"
                    : "Upload a video to begin a new session"}
              </p>
            </div>
            <div className={`monitoring-state ${processing ? "running" : "idle"}`}>
              <span />
              {processing ? "MONITORING" : monitoring ? "COMPLETE" : "STANDBY"}
            </div>
          </div>
        </div>

        <div className="control-section">
          {processing && (
            <button
              className="stop-action"
              onClick={handleStop}
              disabled={stopping}
              type="button"
            >
              {stopping ? "STOPPING..." : "■ STOP MONITORING"}
            </button>
          )}
          {monitoring && !processing && (
            <button
              className="stop-action"
              onClick={handleCloseSession}
              type="button"
            >
              CLOSE SESSION
            </button>
          )}
        </div>

        {message && (
          <div className="status-message" role="status">
            <span className="status-message-dot" />
            {message}
          </div>
        )}
      </section>

      <section className="monitoring-layout">
        <div className="video-panel">
          <div className="video-header">
            <div>
              <div className="panel-kicker">UPLOADED SURVEILLANCE FEED</div>
              <h2>{session?.video_name || "Video player"}</h2>
            </div>
            <div className={`recording-indicator ${processing ? "live" : ""}`}>
              <span />
              {processing ? "PROCESSING" : monitoring ? "ANALYSIS COMPLETE" : "STANDBY"}
            </div>
          </div>

          <div className="video-container">
            {videoUrl ? (
              <video
                key={session?.session_id || videoUrl}
                className="surveillance-video"
                src={videoUrl}
                controls
                autoPlay
                muted
                playsInline
                onError={() =>
                  setMessage(
                    "The video could not be played. Check that the upload and video endpoint are available."
                  )
                }
              />
            ) : (
              <div className="video-placeholder">
                <div className="camera-placeholder-icon">◉</div>
                <div className="video-placeholder-title">No surveillance video</div>
                <div className="video-placeholder-text">
                  Open Upload Video and choose a file to start analysis.
                </div>
              </div>
            )}
          </div>

          <div className="video-footnote">
            The Python pipeline analyzes the uploaded file independently of browser playback.
            Bounding-box overlays are not synchronized with this player yet.
          </div>
        </div>

        <div className="intelligence-panel">
          <div className="video-header">
            <div>
              <div className="panel-kicker">BEHAVIORAL INTELLIGENCE</div>
              <h2>Live Assessment</h2>
            </div>
            <div className="intelligence-pulse">
              <span />
              {processing ? "ANALYZING" : monitoring ? "COMPLETE" : "READY"}
            </div>
          </div>

          <div className="risk-legend">
            <span><i className="legend-dot normal" /> Normal</span>
            <span><i className="legend-dot warning" /> Warning</span>
            <span><i className="legend-dot danger" /> Anomaly flag</span>
          </div>

          <div className="track-list">
            {!monitoring ? (
              <div className="empty-tracks">
                <div className="empty-icon">◎</div>
                <div>Monitoring is not active</div>
                <span>Upload a video to begin a new session.</span>
              </div>
            ) : people.length === 0 ? (
              <div className="empty-tracks">
                <div className="empty-icon">◎</div>
                <div>{processing ? "Waiting for person detections" : "No person results recorded"}</div>
                <span>
                  {processing
                    ? "Results will appear here as the pipeline sends detections."
                    : "The pipeline finished without sending person detections for this session."}
                </span>
              </div>
            ) : (
              people.slice(0, 10).map((person) => {
                const risk = getRiskState(person);
                return (
                  <article
                    key={person.track_id}
                    className={`track-card ${risk.tone}`}
                  >
                    <div className="track-main">
                      <div className={`track-avatar ${risk.tone}`}>
                        #{person.track_id}
                      </div>
                      <div className="track-information">
                        <div className="track-name">
                          Person · Track #{person.track_id}
                        </div>
                        <div className="track-meta">
                          {person.direction || "Direction unavailable"} ·{" "}
                          {Number(person.velocity || 0).toFixed(1)} px/s
                        </div>
                      </div>
                      <div className={`track-state ${risk.tone}`}>
                        <span />
                        {risk.label}
                      </div>
                    </div>
                    <div className={`track-description ${risk.tone}`}>
                      {risk.description}
                    </div>
                  </article>
                );
              })
            )}
          </div>
        </div>
      </section>

      <section className="monitoring-bottom-grid">
        <div className="events-panel">
          <div className="panel-heading compact">
            <div>
              <div className="panel-kicker">ACTIVITY LOG</div>
              <h2>Recent Security Events</h2>
            </div>
            <span className="event-count">{monitoring ? recentEvents.length : 0} EVENTS</span>
          </div>

          {!monitoring || recentEvents.length === 0 ? (
            <div className="empty-event-state">
              {monitoring
                ? "No security events were recorded in this session."
                : "Start a monitoring session to see its events here."}
            </div>
          ) : (
            <div className="event-list">
              {recentEvents.map((event) => (
                <div className="event-row" key={event.id}>
                  <div className={`event-severity ${event.tone}`}><span /></div>
                  <div className="event-time">{formatTime(event.timestamp)}</div>
                  <div className="event-type">{event.kind}</div>
                  <div className="event-description">{event.description}</div>
                  <div className="event-track">#{event.track_id}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="session-panel">
          <div className="panel-heading compact">
            <div>
              <div className="panel-kicker">SESSION OVERVIEW</div>
              <h2>Current Analysis</h2>
            </div>
          </div>

          <div className="session-status-large">
            <div className={`session-status-orb ${processing ? "active" : ""}`}><span /></div>
            <div>
              <div className="session-status-title">
                {processing
                  ? "Processing uploaded video"
                  : monitoring
                    ? "Analysis complete"
                    : "Ready for monitoring"}
              </div>
              <div className="session-status-description">
                {session?.video_name || "No video uploaded yet"}
              </div>
            </div>
          </div>

          <div className="session-stats">
            <div><span>PEOPLE</span><strong>{monitoring ? people.length : 0}</strong></div>
            <div><span>WARNINGS</span><strong>{monitoring ? warnings : 0}</strong></div>
            <div>
              <span>ANOMALY FLAGS</span>
              <strong className="danger-text">{monitoring ? flagged : 0}</strong>
            </div>
          </div>

          <div className="session-disclaimer">
            Anomaly flags are model predictions for human review, not proof of wrongdoing.
            Track IDs identify tracker instances, not verified real-world identities.
          </div>
        </div>
      </section>
    </div>
  );
}

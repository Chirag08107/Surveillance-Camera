const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/live";

export async function getLatestDetections() {
  const res = await fetch(`${API_URL}/api/detections/latest`);
  if (!res.ok) throw new Error("Failed to fetch detections");
  return res.json();
}

export async function getZoneEvents() {
  const res = await fetch(`${API_URL}/api/events/zone`);
  if (!res.ok) throw new Error("Failed to fetch zone events");
  return res.json();
}

export async function getAnomalies() {
  const res = await fetch(`${API_URL}/api/events/anomalies`);
  if (!res.ok) throw new Error("Failed to fetch anomalies");
  return res.json();
}

export async function summarizeAnomaly(id) {
  const res = await fetch(`${API_URL}/api/events/anomalies/${id}/summarize`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to summarize anomaly");
  return res.json();
}

export async function getIdentities() {
  const res = await fetch(`${API_URL}/api/identities/`);
  if (!res.ok) throw new Error("Failed to fetch identities");
  return res.json();
}

export function connectLiveSocket(onMessage, onStatusChange) {
  let socket;
  let retryTimer;

  function open() {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => onStatusChange?.(true);
    socket.onclose = () => {
      onStatusChange?.(false);
      retryTimer = setTimeout(open, 3000);
    };
    socket.onerror = () => socket.close();
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch {
        // ignore malformed frames
      }
    };
  }

  open();

  return () => {
    clearTimeout(retryTimer);
    socket?.close();
  };
}

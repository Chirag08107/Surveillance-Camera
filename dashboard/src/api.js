
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/live";

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      // Keep the HTTP error message if the response is not JSON.
    }

    throw new Error(message);
  }

  return response.json();
}

export function getLatestDetections() {
  return request("/api/detections/latest");
}

export function getZoneEvents() {
  return request("/api/events/zone");
}

export function getAnomalies() {
  return request("/api/events/anomalies");
}

export function summarizeAnomaly(id) {
  return request(`/api/events/anomalies/${id}/summarize`, {
    method: "POST",
  });
}

export function getIdentities() {
  return request("/api/identities/");
}

export function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/api/monitoring/upload", {
    method: "POST",
    body: formData,
  });
}

export function getMonitoringStatus() {
  return request("/api/monitoring/status");
}

export function stopMonitoring() {
  return request("/api/monitoring/stop", {
    method: "POST",
  });
}

export function getVideoUrl(path) {
  if (!path) return "";

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  return `${API_URL}${path.startsWith("/") ? "" : "/"}${path}`;
}

export function connectLiveSocket(onMessage, onStatusChange) {
  let socket = null;
  let retryTimer = null;
  let closedByClient = false;

  function open() {
    if (closedByClient) return;

    socket = new WebSocket(WS_URL);

    socket.onopen = () => onStatusChange?.(true);

    socket.onclose = () => {
      onStatusChange?.(false);

      if (!closedByClient) {
        retryTimer = setTimeout(open, 3000);
      }
    };

    socket.onerror = () => socket?.close();

    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch {
        // Ignore malformed messages.
      }
    };
  }

  open();

  return () => {
    closedByClient = true;
    clearTimeout(retryTimer);
    socket?.close();
  };
}

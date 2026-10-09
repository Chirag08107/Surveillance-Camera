
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
} from "recharts";

export default function AnalyticsPanel({
  detections = [],
}) {
  const latestByTrack = new Map();

  detections.forEach((detection) => {
    if (
      detection.class_name?.toLowerCase() !== "person" ||
      detection.track_id === undefined ||
      detection.track_id === null
    ) {
      return;
    }

    latestByTrack.set(Number(detection.track_id), detection);
  });

  let realCount = 0;
  let anomalyCount = 0;
  let warningCount = 0;
  let unknownCount = 0;

  latestByTrack.forEach((person) => {
    const behavior = person.behavior?.toLowerCase();

    if (behavior === "anomaly") {
      anomalyCount += 1;
    } else if (person.in_restricted_zone) {
      warningCount += 1;
    } else if (["real", "normal"].includes(behavior)) {
      realCount += 1;
    } else {
      unknownCount += 1;
    }
  });

  const suspiciousCount = anomalyCount + warningCount;

  let videoCategory = "Unknown";

  if (suspiciousCount > realCount) {
    videoCategory = "Anomaly";
  } else if (realCount > suspiciousCount) {
    videoCategory = "Real";
  }

  const data = [
    {
      behavior: "Real",
      count: realCount,
    },
    {
      behavior: "Warning",
      count: warningCount,
    },
    {
      behavior: "Anomaly",
      count: anomalyCount,
    },
  ];

  if (latestByTrack.size === 0) {
    return (
      <div className="empty-state">
        No person behavior data available for this session.
      </div>
    );
  }

  return (
    <div>
      <div
        style={{
          marginBottom: 16,
          padding: "14px 16px",
          border: "1px solid #1f1f1f",
          borderRadius: 10,
          background: "#0a0a0a",
        }}
      >
        <div
          style={{
            color: "#8a8a8a",
            fontSize: 11,
            letterSpacing: 1.2,
            marginBottom: 6,
          }}
        >
          FINAL VIDEO CLASSIFICATION
        </div>

        <div
          style={{
            color:
              videoCategory === "Anomaly"
                ? "#ff4d5a"
                : videoCategory === "Real"
                  ? "#19d3a2"
                  : "#e8b45b",
            fontSize: 20,
            fontWeight: 700,
          }}
        >
          {videoCategory}
        </div>

        <div
          style={{
            color: "#8a8a8a",
            fontSize: 12,
            marginTop: 5,
          }}
        >
          Real: {realCount} · Warning: {warningCount} ·
          {" "}Anomaly: {anomalyCount}
        </div>

        <div
          style={{
            color: "#8a8a8a",
            fontSize: 12,
            marginTop: 4,
          }}
        >
          Combined suspicious tracks: {suspiciousCount}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1f1f1f"
          />

          <XAxis
            dataKey="behavior"
            stroke="#8a8a8a"
            fontSize={12}
          />

          <YAxis
            stroke="#8a8a8a"
            fontSize={12}
            allowDecimals={false}
          />

          <Tooltip
            contentStyle={{
              background: "#0a0a0a",
              border: "1px solid #1f1f1f",
              color: "#e8e8e8",
            }}
          />

          <Bar
            dataKey="count"
            fill="#ff3b3b"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

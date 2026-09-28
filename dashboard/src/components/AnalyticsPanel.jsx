import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";

export default function AnalyticsPanel({ detections }) {
  const counts = {};
  detections.forEach((d) => {
    const label = d.behavior || "Unknown";
    counts[label] = (counts[label] || 0) + 1;
  });

  const data = Object.entries(counts).map(([behavior, count]) => ({ behavior, count }));

  if (!data.length) {
    return <div className="empty-state">No behavior data yet.</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
        <XAxis dataKey="behavior" stroke="#8a8a8a" fontSize={12} />
        <YAxis stroke="#8a8a8a" fontSize={12} allowDecimals={false} />
        <Tooltip
          contentStyle={{ background: "#0a0a0a", border: "1px solid #1f1f1f", color: "#e8e8e8" }}
        />
        <Bar dataKey="count" fill="#ff3b3b" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

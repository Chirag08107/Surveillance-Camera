export default function DetectionsTable({ detections }) {
  if (!detections.length) {
    return <div className="empty-state">No active tracks right now.</div>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Track ID</th>
          <th>Class</th>
          <th>Behavior</th>
          <th>Direction</th>
          <th>Velocity</th>
          <th>Zone</th>
        </tr>
      </thead>
      <tbody>
        {detections.map((d) => (
          <tr key={d.track_id}>
            <td>#{d.track_id}</td>
            <td>{d.class_name}</td>
            <td>{d.behavior || "—"}</td>
            <td>{d.direction || "—"}</td>
            <td>{d.velocity ? d.velocity.toFixed(1) : "0.0"} px/s</td>
            <td>
              {d.in_restricted_zone ? (
                <span className="badge zone">In zone</span>
              ) : (
                <span className="badge normal">Clear</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

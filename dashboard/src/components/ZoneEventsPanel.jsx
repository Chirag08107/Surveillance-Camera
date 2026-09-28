export default function ZoneEventsPanel({ events }) {
  if (!events.length) {
    return <div className="empty-state">No zone entries/exits recorded yet.</div>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Track ID</th>
          <th>Event</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e) => (
          <tr key={e.id}>
            <td>#{e.track_id}</td>
            <td>
              <span className={`badge ${e.event_type === "enter" ? "zone" : "normal"}`}>
                {e.event_type === "enter" ? "Entered zone" : "Left zone"}
              </span>
            </td>
            <td>{new Date(e.timestamp).toLocaleTimeString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

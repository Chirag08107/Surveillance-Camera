import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function IdentitiesPanel({ identities, onUpdated }) {
  const [personId, setPersonId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [photo, setPhoto] = useState(null);
  const [status, setStatus] = useState("");

  async function handleEnroll(e) {
    e.preventDefault();
    if (!photo || !personId || !displayName) {
      setStatus("Fill in ID, name, and a consented photo first.");
      return;
    }

    setStatus("Enrolling…");
    const form = new FormData();
    form.append("photo", photo);

    try {
      const res = await fetch(
        `${API_URL}/api/identities/enroll?person_id=${encodeURIComponent(
          personId
        )}&display_name=${encodeURIComponent(displayName)}`,
        { method: "POST", body: form }
      );
      if (!res.ok) throw new Error(await res.text());
      setStatus("Enrolled.");
      setPersonId("");
      setDisplayName("");
      setPhoto(null);
      onUpdated?.();
    } catch (err) {
      setStatus(`Failed: ${err.message}`);
    }
  }

  async function handleDelete(personId) {
    await fetch(`${API_URL}/api/identities/${personId}`, { method: "DELETE" });
    onUpdated?.();
  }

  return (
    <>
      <p style={{ color: "var(--text-dim)", fontSize: 13, marginTop: -6, marginBottom: 16 }}>
        Enrollment is opt-in only — only people who consent and are enrolled here can ever be
        matched. Unmatched faces are never stored.
      </p>

      <form onSubmit={handleEnroll} style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        <input
          placeholder="Person ID"
          value={personId}
          onChange={(e) => setPersonId(e.target.value)}
          style={inputStyle}
        />
        <input
          placeholder="Display name"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          style={inputStyle}
        />
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setPhoto(e.target.files[0])}
          style={{ color: "var(--text-dim)", fontSize: 13 }}
        />
        <button className="btn" type="submit">Enroll</button>
      </form>

      {status && <p style={{ color: "var(--text-dim)", fontSize: 12 }}>{status}</p>}

      {identities.length === 0 ? (
        <div className="empty-state">No enrolled identities.</div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Person ID</th>
              <th>Name</th>
              <th>Consented at</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {identities.map((i) => (
              <tr key={i.person_id}>
                <td>{i.person_id}</td>
                <td>{i.display_name}</td>
                <td>{new Date(i.consented_at).toLocaleDateString()}</td>
                <td>
                  <button className="btn" onClick={() => handleDelete(i.person_id)}>Forget</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

const inputStyle = {
  background: "#0f0f0f",
  border: "1px solid var(--panel-border)",
  color: "var(--text)",
  borderRadius: 6,
  padding: "8px 10px",
  fontSize: 13,
};

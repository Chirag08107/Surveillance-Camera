
import { useRef, useState } from "react";
import { uploadVideo } from "../api";

const MAX_SIZE = 500 * 1024 * 1024;

const ALLOWED_TYPES = [
  "video/mp4",
  "video/avi",
  "video/quicktime",
  "video/x-matroska",
  "video/webm",
  "video/x-msvideo",
];

export default function UploadVideo({ onUploadComplete }) {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function selectFile(selected) {
    setError("");

    if (!selected) return;

    const extension = selected.name.split(".").pop()?.toLowerCase();

    if (!["mp4", "avi", "mov", "mkv", "webm"].includes(extension)) {
      setError("Choose an MP4, AVI, MOV, MKV, or WEBM video.");
      return;
    }

    if (selected.size === 0) {
      setError("This file is empty. Choose another video.");
      return;
    }

    if (selected.size > MAX_SIZE) {
      setError("The maximum upload size is 500 MB.");
      return;
    }

    if (
      selected.type &&
      !ALLOWED_TYPES.includes(selected.type) &&
      !selected.type.startsWith("video/")
    ) {
      setError("The selected file does not appear to be a video.");
      return;
    }

    setFile(selected);
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file || loading) return;

    setLoading(true);
    setError("");

    try {
      const result = await uploadVideo(file);

      if (!result.video_url || !result.session_id) {
        throw new Error("The backend returned an incomplete session response.");
      }

      onUploadComplete(result);
    } catch (uploadError) {
      setError(uploadError.message || "Video upload failed.");
      setLoading(false);
    }
  }

  return (
    <div className="upload-page">
      <section className="upload-intro">
        <div className="page-kicker">AI-POWERED SURVEILLANCE</div>
        <h2>Turn video into actionable intelligence.</h2>
        <p>
          Upload a surveillance clip to track people, analyze pose and movement,
          and flag behavioral patterns that may need review.
        </p>
      </section>

      <form onSubmit={handleSubmit} className="upload-card">
        <div className="upload-card-heading">
          <div>
            <div className="panel-kicker">VIDEO INPUT</div>
            <h3>Start a monitoring session</h3>
          </div>

          <div className="upload-icon">
            <span>↑</span>
          </div>
        </div>

        <div
          className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            selectFile(event.dataTransfer.files?.[0]);
          }}
        >
          <input
            ref={inputRef}
            className="hidden-file-input"
            type="file"
            accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
            onChange={(event) => selectFile(event.target.files?.[0])}
          />

          <div className="dropzone-icon">
            {file ? "✓" : "↑"}
          </div>

          {file ? (
            <>
              <div className="dropzone-title">{file.name}</div>
              <div className="dropzone-description">
                {(file.size / (1024 * 1024)).toFixed(2)} MB · Ready for analysis
              </div>
            </>
          ) : (
            <>
              <div className="dropzone-title">Drop your video here</div>
              <div className="dropzone-description">
                or select a file from your device
              </div>
            </>
          )}

          <button
            type="button"
            className="secondary-action"
            onClick={() => inputRef.current?.click()}
            disabled={loading}
          >
            {file ? "Choose another video" : "Browse files"}
          </button>

          <div className="file-requirements">
            MP4, AVI, MOV, MKV, WEBM · Maximum 500 MB
          </div>
        </div>

        {error && (
          <div className="error-message" role="alert">
            <span>!</span>
            {error}
          </div>
        )}

        <div className="upload-info-grid">
          <div>
            <span className="info-number">01</span>
            <div>
              <strong>Detect and track</strong>
              <p>Identify people and maintain track IDs.</p>
            </div>
          </div>

          <div>
            <span className="info-number">02</span>
            <div>
              <strong>Analyze movement</strong>
              <p>Extract pose and temporal features.</p>
            </div>
          </div>

          <div>
            <span className="info-number">03</span>
            <div>
              <strong>Review alerts</strong>
              <p>Inspect anomaly flags and zone events.</p>
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="primary-action upload-submit"
          disabled={!file || loading}
        >
          {loading ? (
            <>
              <span className="loading-spinner" />
              UPLOADING AND STARTING ANALYSIS...
            </>
          ) : (
            <>
              <span>▶</span>
              START ANALYSIS
            </>
          )}
        </button>

        <p className="upload-disclaimer">
          An anomaly flag indicates a model prediction that requires review. It
          is not proof that a person has committed wrongdoing.
        </p>
      </form>
    </div>
  );
}

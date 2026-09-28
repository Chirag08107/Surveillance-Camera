"""
STEP 12 - FACE RECOGNITION + PRIVACY CONTROLS

Per the architecture: "Authorized/consented identity matching with
appropriate privacy safeguards." This is deliberately NOT a covert
"identify anyone who walks past the camera" system. It only ever
matches against people who were explicitly enrolled with consent.

Privacy safeguards built in:
  - Enrollment is opt-in only, via `enroll()` - nothing is matched
    against a face that wasn't explicitly added.
  - Only a numeric face embedding is stored, never the raw enrollment
    photo (the image is used in-memory to compute the embedding, then
    discarded).
  - `redact_unmatched=True` (default) means faces that don't match any
    enrolled identity are reported only as "unidentified" - no
    embedding or image is retained for them.
  - `delete(person_id)` supports the right to be forgotten.
  - Matching returns a confidence score; callers should treat anything
    below `match_threshold` as "no match", not a low-confidence guess.

Requires: pip install face_recognition (needs cmake + dlib - see
requirements.txt for platform notes).
"""

import pickle
import os
import numpy as np

try:
    import face_recognition
except ImportError:
    face_recognition = None  # see requirements.txt - dlib build is optional/platform-dependent


class FaceIdentityStore:
    def __init__(self, store_path: str = "identity/enrolled_faces.pkl", match_threshold: float = 0.6):
        if face_recognition is None:
            raise RuntimeError(
                "face_recognition is not installed. See requirements.txt - "
                "this feature is optional and requires a dlib build."
            )

        self.store_path = store_path
        self.match_threshold = match_threshold
        self.embeddings: dict[str, np.ndarray] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, "rb") as f:
                self.embeddings = pickle.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def enroll(self, person_id: str, image_bgr: np.ndarray) -> bool:
        """
        Explicit opt-in enrollment. `image_bgr` should be a clear, single-face
        photo taken with the person's consent. Only the embedding is stored;
        the image itself is never written to disk by this method.
        """

        rgb = image_bgr[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb)

        if len(face_locations) != 1:
            return False  # require exactly one clear face for enrollment

        encoding = face_recognition.face_encodings(rgb, face_locations)[0]
        self.embeddings[person_id] = encoding
        self._save()
        return True

    def delete(self, person_id: str) -> bool:
        """Right-to-be-forgotten: permanently removes an enrolled identity."""
        if person_id in self.embeddings:
            del self.embeddings[person_id]
            self._save()
            return True
        return False

    def match_faces_in_frame(self, frame_bgr: np.ndarray, redact_unmatched: bool = True):
        """
        Returns a list of {"box": (top, right, bottom, left), "person_id": str | None, "confidence": float}
        for every face detected. If redact_unmatched is True (default), unmatched
        faces get person_id=None and confidence=0.0 - nothing about them is stored.
        """

        rgb = frame_bgr[:, :, ::-1]
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        known_ids = list(self.embeddings.keys())
        known_vectors = np.array([self.embeddings[pid] for pid in known_ids]) if known_ids else None

        results = []

        for box, encoding in zip(locations, encodings):
            person_id = None
            confidence = 0.0

            if known_vectors is not None:
                distances = np.linalg.norm(known_vectors - encoding, axis=1)
                best_idx = int(np.argmin(distances))
                best_distance = distances[best_idx]

                # face_recognition distances: lower is more similar; convert to a 0-1 confidence
                similarity = max(0.0, 1.0 - best_distance)

                if similarity >= self.match_threshold:
                    person_id = known_ids[best_idx]
                    confidence = similarity

            if person_id is None and redact_unmatched:
                results.append({"box": box, "person_id": None, "confidence": 0.0})
            else:
                results.append({"box": box, "person_id": person_id, "confidence": confidence})

        return results

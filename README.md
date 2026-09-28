# AI Surveillance & Behavioral Intelligence Platform

Steps 1–7 (video input, YOLO detection, tracking, restricted-zone
detection, pose estimation, rule-based behavior, motion features) are
your original `main.py` / `pose.py` / `behaviour.py` / `test.py` —
**unchanged**. Everything below is additive: Steps 8–17.

```
video → YOLO detect/track → pose → behaviour.py (rules)          [Steps 1-7, original]
                                  → features/ (vectors + sequences) [Step 8]
                                  → dataset/  (labeling tool)       [Step 9]
                                  → model/    (LSTM classifier)     [Step 10]
                                  → anomaly/  (outlier + loitering) [Step 11]
                                  → identity/ (consent-based faces) [Step 12]
                                  → anpr/     (plate OCR)           [Step 13]
pipeline.py wires all of the above together and posts results to →
backend/  (FastAPI + Postgres, Step 14) → dashboard/ (React, Step 15)
                                          → llm_summary.py (Step 16)
```

## 1. Why there's no `venv/` in the zip, and what's genuinely "done"

A virtualenv is machine-specific (OS, Python version, GPU/CPU) — it's
never something you ship in a zip, you create it locally. Below is
exactly how.

Also, three honest limits worth knowing up front:

- **The neural network isn't trained.** There's no labeled dataset in
  the world for your specific cameras/scenes. `model/train.py` is a
  complete, working training pipeline — but it needs YOU to run
  `dataset/dataset_generator.py` against labeled clips first (a
  walking clip, a falling clip, etc). Until you do, `pipeline.py`
  automatically falls back to the original rule-based Standing/Sitting
  behaviour from `behaviour.py` — nothing breaks, it's just not a
  learned model yet.
- **Face recognition is consent-based only**, per your architecture
  note ("Authorized/consented identity matching with appropriate
  privacy safeguards"). It will only ever match people you explicitly
  enrolled through the dashboard's Identities tab. It does not do
  general-purpose face ID on strangers.
- **This can't be "deployed" from inside this chat.** There's no
  environment here that can run a persistent GPU process reading a
  live camera feed. Section 4 below gives you a real path to deploy
  the backend + dashboard to an actual server, and Section 3 to run
  everything locally first.

## 2. Local setup (venv)

```bash
cd Ai-project-1
python3 -m venv venv

# activate it:
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Notes on `requirements.txt`:
- `face_recognition`/`dlib` need a C++ build toolchain. If you don't
  need Step 12, delete those two lines and skip the Identities tab.
- `torch` installs CPU-only by default on most systems. For GPU
  training, install the CUDA build from https://pytorch.org/get-started/locally/
  instead, *before* running `pip install -r requirements.txt` (pip
  will then see torch already satisfied).

Copy the env template and fill in real values:
```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY if you want Step 16 incident summaries
```

## 3. Running it locally, in order

**a. Start Postgres + the backend** (easiest via Docker):
```bash
docker compose up -d db      # just the database
uvicorn backend.main:app --reload --port 8000   # backend, from your venv
```
Or run everything (db + backend) in Docker: `docker compose up --build`.

**b. Start the dashboard:**
```bash
cd dashboard
cp .env.example .env
npm install
npm run dev
# open http://localhost:5173
```

**c. Run the integrated pipeline** against your test video:
```bash
python pipeline.py --video videos/test.mp4
# or, on a machine with no display (a server):
python pipeline.py --video videos/test.mp4 --headless
```
Detections, zone events, and anomalies now stream live into the
dashboard, and persist in Postgres for history.

Your original standalone scripts still work exactly as before and are
untouched:
```bash
python main.py     # original steps 1-7 demo, unchanged
python test.py     # original pose keypoint dump, unchanged
```

## 4. Training the behavior classifier (Steps 9-10) once you have footage

```bash
python -m dataset.dataset_generator \
    --video videos/walking_clip.mp4 --label Walking \
    --video videos/falling_clip.mp4 --label Falling \
    --video videos/loitering_clip.mp4 --label Loitering \
    --out dataset/behavior_dataset.npz

python -m model.train --data dataset/behavior_dataset.npz --epochs 40
```
This writes `model/checkpoints/behavior_lstm.pt` — `pipeline.py` picks
it up automatically on the next run (it checks that path at startup).
More clips per class = a real model instead of the rule-based fallback.

## 5. Deploying for real

**Backend + Postgres** — pick one host that can run a long-lived
Docker container (this repo's `Dockerfile`/`docker-compose.yml` work
as-is on all of these):
- Render / Railway: connect the repo, point it at `Dockerfile`, add a
  managed Postgres add-on, set `DATABASE_URL` and `ANTHROPIC_API_KEY`
  as environment variables.
- Your own VPS (e.g. a GPU box for the pipeline): `git clone`, then
  `docker compose up --build -d`.

**The pipeline process** (`pipeline.py`) needs real camera/video access
and ideally a GPU — it's not something a serverless platform runs well.
Run it on-site or on a GPU VM, pointed at your deployed backend:
```bash
python pipeline.py --video rtsp://your-camera-url --backend-url https://your-backend.example.com --headless
```

**Dashboard** — it's a static site once built:
```bash
cd dashboard
# set VITE_API_URL / VITE_WS_URL in .env to your deployed backend's URL (wss:// in production)
npm run build
# deploy the resulting dashboard/dist/ folder to Vercel, Netlify, Cloudflare Pages, or any static host
```

## 6. What's left for a fully "final integrated system" (Step 17)

Everything is wired and runs end-to-end today with the rule-based
fallback. To get to production-grade:
- Collect and label enough footage per behavior class to train a
  classifier that actually beats the fallback.
- Add track-timeout cleanup (drop `SequenceBuilder`/`AnomalyDetector`
  state for tracks that leave frame — noted with a TODO comment in
  `pipeline.py`).
- Tighten CORS in `backend/main.py` to your real dashboard origin
  before going to production.
- Add auth in front of the dashboard/API — right now anyone who can
  reach the URL can view detections and enroll identities.

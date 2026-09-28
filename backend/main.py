"""
STEP 14 - FASTAPI + POSTGRESQL BACKEND (entrypoint)

Run with:
    uvicorn backend.main:app --reload --port 8000

Then run the actual CV pipeline separately (it pushes into this backend
over HTTP + the /ws/live socket):
    python pipeline.py --video videos/test.mp4
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routers import detections, events, identities, stream, ingest

app = FastAPI(title="Surveillance & Behavioral Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your dashboard's real origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detections.router)
app.include_router(events.router)
app.include_router(identities.router)
app.include_router(stream.router)
app.include_router(ingest.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}

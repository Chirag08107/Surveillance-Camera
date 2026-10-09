
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routers import (
    detections,
    events,
    identities,
    ingest,
    monitoring,
    stream,
)

app = FastAPI(title="Sentinel Surveillance & Behavioral Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detections.router)
app.include_router(events.router)
app.include_router(identities.router)
app.include_router(stream.router)
app.include_router(ingest.router)
app.include_router(monitoring.router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "sentinel-backend",
    }

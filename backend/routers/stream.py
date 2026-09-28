"""
Live feed for the dashboard. pipeline.py pushes each frame's results into
`broadcaster`; every connected dashboard client receives them over this
WebSocket as JSON. This is deliberately decoupled from the DB writes
(events.py/detections.py read from Postgres for history) - the socket is
for "what's happening right now" with minimal latency.
"""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class Broadcaster:
    def __init__(self):
        self.connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)

    async def broadcast(self, payload: dict):
        message = json.dumps(payload, default=str)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


broadcaster = Broadcaster()


@router.websocket("/ws/live")
async def live_feed(ws: WebSocket):
    await broadcaster.connect(ws)
    try:
        while True:
            # keep the connection open; the server only ever pushes
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        broadcaster.disconnect(ws)

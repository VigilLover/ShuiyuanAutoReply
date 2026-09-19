"""Read-only forum monitor snapshot and its live SSE event stream."""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..support import state_store

router = APIRouter()


@router.get("/api/forum/monitor")
async def forum_monitor(request: Request):
    return await state_store(request).forum_monitor()


@router.get("/api/forum/events/stream")
async def forum_events(request: Request, after: int = 0):
    try:
        cursor = max(0, after, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的事件游标")
    store = state_store(request)

    async def stream():
        nonlocal cursor
        heartbeat = asyncio.get_running_loop().time()
        while not await request.is_disconnected():
            batch = await store.forum_events_after(cursor)
            for event in batch:
                cursor = event["event_id"]
                yield f"id: {cursor}\nevent: forum.event\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            now = asyncio.get_running_loop().time()
            if now - heartbeat >= 15:
                yield ": heartbeat\n\n"
                heartbeat = now
            if len(batch) < 200:
                await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

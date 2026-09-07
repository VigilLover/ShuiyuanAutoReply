"""Enforce request size before multipart parsing, including chunked requests."""

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse


class RequestLimits:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from shuiyuan_auto_reply.application.scheduling import get_scheduler
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

        if scope["method"] == "POST" and (
            "/messages/stream" in scope["path"] or scope["path"] == "/api/chat"
        ):
            scheduler = get_scheduler()
            if scheduler.waiting >= scheduler.queue_limit:
                return await JSONResponse(
                    {"detail": "Reply queue is full"}, status_code=429
                )(scope, receive, send)
        limit = get_deployment().section("media")["max_turn_bytes"] + 1024 * 1024
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = limit + 1
        if declared > limit:
            return await JSONResponse({"detail": "Request too large"}, status_code=413)(
                scope, receive, send
            )
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > limit:
                raise HTTPException(413, "Request too large")
            return message

        if scope["method"] == "POST" and "/messages/stream" in scope["path"]:
            from shuiyuan_auto_reply.application.scheduling import BusyError

            # Reserve a slot before reading uploads; waiting requests do not hold
            # decoded pictures or complete multipart bodies in application memory.
            try:
                async with get_scheduler().admission(scope["path"]):
                    await self.app(scope, bounded_receive, send)
            except BusyError:
                await JSONResponse({"detail": "Reply queue is full"}, status_code=429)(
                    scope, receive, send
                )
        else:
            await self.app(scope, bounded_receive, send)

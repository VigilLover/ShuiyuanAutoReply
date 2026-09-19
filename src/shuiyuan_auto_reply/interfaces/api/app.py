"""Composition root for the management API: assembles routers onto one FastAPI app.

Route handlers live in :mod:`shuiyuan_auto_reply.interfaces.api.routes`, grouped
by concern (legacy chat contract, forum monitor, conversations, artifacts,
runtime profiles, model config library, tool/MCP status, static frontend).
Shared helpers (state-store lookup, profile defaults, the session registry) are
in :mod:`shuiyuan_auto_reply.interfaces.api.support`.
"""

from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shuiyuan_auto_reply.application.scheduling import BusyError
from shuiyuan_auto_reply.bootstrap import ApplicationContainer

from .limits import RequestLimits
from .routes import (
    artifacts,
    compat,
    conversations,
    forum_monitor,
    model_configs,
    settings_profiles,
    static,
    tools,
)
from .support import SessionRegistry

ContainerFactory = Callable[[], Awaitable[ApplicationContainer]]


def create_app(container_factory: ContainerFactory | None = None) -> FastAPI:
    factory = container_factory or ApplicationContainer.for_api

    @asynccontextmanager
    async def lifespan(current_app: FastAPI):
        load_dotenv()
        container = await factory()
        current_app.state.container = container
        current_app.state.sessions = SessionRegistry()
        try:
            yield
        finally:
            await container.aclose()

    api = FastAPI(title="ShuiyuanAutoReply 对话后端", lifespan=lifespan)

    @api.exception_handler(BusyError)
    async def busy_handler(request, exc):
        return JSONResponse(
            status_code=429, content={"detail": str(exc)}, headers={"Retry-After": "5"}
        )

    api.add_middleware(RequestLimits)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )

    for module in (
        compat,
        forum_monitor,
        conversations,
        artifacts,
        settings_profiles,
        model_configs,
        tools,
    ):
        api.include_router(module.router)
    static.mount_static(api)

    return api


app = create_app()

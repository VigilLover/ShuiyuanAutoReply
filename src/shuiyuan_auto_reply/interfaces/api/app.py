"""Existing HTTP contract backed by the shared BotService."""

import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shuiyuan_auto_reply.bootstrap import ApplicationContainer
from shuiyuan_auto_reply.domain import (
    ActorRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ReplyRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionData:
    token: str


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    def authenticate(self, session_id: str, token: str) -> None:
        current = self._sessions.get(session_id)
        if current is None:
            self._sessions[session_id] = SessionData(token)
        elif current.token != token:
            raise PermissionError("invalid token")

    def authorize_removal(self, session_id: str, token: str) -> bool:
        current = self._sessions.get(session_id)
        if current is None:
            return False
        if current.token != token:
            raise PermissionError("invalid token")
        return True

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)


class ChatRequest(BaseModel):
    session_id: str
    token: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ClearRequest(BaseModel):
    session_id: str
    token: str


class ClearResponse(BaseModel):
    status: str
    message: str


ContainerFactory = Callable[[], Awaitable[ApplicationContainer]]


def _conversation(session_id: str) -> ConversationRef:
    return ConversationRef(Channel.API, session_id, "wolf_lumine", "wolf_lumine")


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

    api = FastAPI(title="小南瓜 (OpenRouter) 对话后端", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(payload: ChatRequest, request: Request):
        if not payload.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")
        sessions: SessionRegistry = request.app.state.sessions
        try:
            sessions.authenticate(payload.session_id, payload.token)
        except PermissionError:
            raise HTTPException(
                status_code=403, detail="身份验证失败：Token 错误或已过期。"
            ) from None

        reply_request = ReplyRequest(
            request_id=str(uuid.uuid4()),
            conversation=_conversation(payload.session_id),
            actor=ActorRef(
                Channel.API, payload.session_id, "NULL", None
            ),
            content=payload.message,
            dispatch_mode=DispatchMode.CHAT_ONLY,
        )
        try:
            result = await request.app.state.container.bot_service.reply(reply_request)
        except Exception as exc:
            logger.exception("处理消息时发生错误")
            raise HTTPException(
                status_code=500, detail=f"内部服务器错误: {str(exc)}"
            ) from exc
        return ChatResponse(session_id=payload.session_id, reply=result.text)

    @api.post("/api/clear", response_model=ClearResponse)
    async def clear_endpoint(payload: ClearRequest, request: Request):
        sessions: SessionRegistry = request.app.state.sessions
        try:
            existed = sessions.authorize_removal(payload.session_id, payload.token)
        except PermissionError:
            raise HTTPException(
                status_code=403, detail="身份验证失败：Token 错误，无法清理他人历史。"
            ) from None
        if not existed:
            return ClearResponse(status="success", message="会话已处于清理状态")
        try:
            await request.app.state.container.bot_service.clear_conversation(
                _conversation(payload.session_id)
            )
        except Exception as exc:
            logger.exception("清除底层模型历史时发生错误")
            raise HTTPException(
                status_code=500, detail=f"清理模型历史失败: {str(exc)}"
            ) from exc
        sessions.discard(payload.session_id)
        return ClearResponse(status="success", message="已成功清除会话和历史记录")

    @api.get("/api/health")
    async def health_check(request: Request):
        return {
            "status": "ok",
            "active_sessions_count": len(request.app.state.sessions),
        }

    return api


app = create_app()

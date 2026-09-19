"""Legacy chat/clear contract plus health, liveness and bootstrap endpoints."""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from shuiyuan_auto_reply.application.scheduling import BusyError
from shuiyuan_auto_reply.bootstrap import AppSettings
from shuiyuan_auto_reply.domain import (
    UNKNOWN_REPLY_TEXT,
    ActorRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ReplyRequest,
)

from ..support import SessionRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


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


def _conversation(session_id: str) -> ConversationRef:
    return ConversationRef(Channel.API, session_id, "wolf_lumine", "wolf_lumine")


@router.post("/api/chat", response_model=ChatResponse)
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
        actor=ActorRef(Channel.API, payload.session_id, "NULL", None),
        content=payload.message,
        dispatch_mode=DispatchMode.CHAT_ONLY,
    )
    try:
        result = await request.app.state.container.bot_service.reply(reply_request)
    except BusyError:
        raise
    except Exception as exc:
        logger.exception("处理消息时发生错误")
        raise HTTPException(status_code=500, detail=UNKNOWN_REPLY_TEXT) from exc
    return ChatResponse(session_id=payload.session_id, reply=result.text)


@router.post("/api/clear", response_model=ClearResponse)
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


@router.get("/api/health")
async def health_check(request: Request):
    return {
        "status": "ok",
        "active_sessions_count": len(request.app.state.sessions),
    }


@router.get("/api/live")
async def liveness():
    return {"status": "ok"}


@router.get("/api/runtime-health")
async def readiness(request: Request):
    from ..health import runtime_health
    from ..support import state_store

    return await runtime_health(state_store(request))


@router.get("/api/bootstrap")
async def bootstrap():
    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

    runtime = get_deployment().section("runtime")
    providers = AppSettings().providers
    return {
        "app": "ShuiyuanAutoReply",
        "channels": ["web", "forum"],
        "web_enabled": True,
        # Deployment-level knobs shown read-only in the settings page.
        "runtime": {
            key: runtime[key]
            for key in (
                "concurrency",
                "image_concurrency",
                "timeout",
                "model_call_timeout",
                "final_reserve_seconds",
                "context_token_budget",
                "model_limit",
                "query_limit",
                "no_progress_batches",
            )
        },
        "reasoning": {
            "investigate": providers.deepseek_reasoning_effort,
            "final": providers.deepseek_final_reasoning_effort,
            "request_timeout": providers.deepseek_request_timeout,
        },
    }

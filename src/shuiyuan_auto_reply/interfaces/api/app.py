"""Existing HTTP contract backed by the shared BotService."""

import logging
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shuiyuan_auto_reply.bootstrap import AppSettings, ApplicationContainer
from shuiyuan_auto_reply.domain import (
    ActorRef,
    AttachmentRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ReplyRequest,
)
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.features.mention.deepseek_vision import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_TURN,
    DeepSeekFilesClient,
    VisionMediaError,
    save_uploaded_image,
)

logger = logging.getLogger(__name__)
DEEPSEEK_VISION_MODEL = "deepseek-v4-flash-vision-exp"


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


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationRenameRequest(BaseModel):
    title: str


class ConversationMessageRequest(BaseModel):
    message: str


class ProfileDraftRequest(BaseModel):
    provider: str
    model: str | None = None
    fallback_model: str | None = None
    system_prompt: str
    enabled_tools: list[str] | None = None
    disabled_mcp_tools: list[str] = Field(default_factory=list)
    api_key: str | None = None


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

    api = FastAPI(title="ShuiyuanAutoReply 对话后端", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
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

    @api.get("/api/bootstrap")
    async def bootstrap():
        return {
            "app": "ShuiyuanAutoReply",
            "channels": ["web", "forum"],
            "web_enabled": True,
        }

    def _store(request: Request):
        store = getattr(request.app.state.container, "state_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="本地状态库未启用")
        return store

    @api.get("/api/conversations")
    async def list_conversations(request: Request, channel: str | None = None, search: str | None = None, limit: int = 100, offset: int = 0):
        records = await _store(request).list_conversations(channel=channel, search=search, limit=limit, offset=offset)
        return [record.__dict__ if hasattr(record, "__dict__") else {
            name: getattr(record, name) for name in record.__dataclass_fields__
        } for record in records]

    @api.post("/api/conversations")
    async def create_conversation(payload: ConversationCreateRequest, request: Request):
        external_id = str(uuid.uuid4())
        ref = ConversationRef(Channel.WEB, external_id, "wolf_lumine", "wolf_lumine")
        record = await _store(request).ensure_conversation(ref, title=payload.title or "新对话")
        return {name: getattr(record, name) for name in record.__dataclass_fields__}

    async def _conversation_record(request: Request, conversation_id: str):
        record = await _store(request).get_conversation(conversation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return record

    def _ref_from_record(record) -> ConversationRef:
        return ConversationRef(Channel(record.channel), record.external_id, record.bot_id, record.persona_id)

    @api.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, request: Request):
        store = _store(request)
        record = await _conversation_record(request, conversation_id)
        messages = await store.list_messages(conversation_id)
        events = await store.list_events_for_conversation(conversation_id)
        serialized_messages = []
        for message in messages:
            attachments = []
            display_content = message.content
            for artifact_id in message.attachments:
                artifact = await store.get_artifact(artifact_id)
                if artifact and artifact.available:
                    display_content = display_content.replace(
                        f"artifact://{artifact.id}", f"/api/artifacts/{artifact.id}"
                    )
                    attachments.append({
                        "artifact_id": artifact.id,
                        "url": f"/api/artifacts/{artifact.id}",
                        "mime_type": artifact.mime_type,
                        "filename": artifact.filename,
                        "width": artifact.width,
                        "height": artifact.height,
                        "source_kind": artifact.source_kind,
                        "source_url": artifact.source_url,
                    })
            serialized_messages.append({
                "id": message.id,
                "role": message.role,
                "content": display_content,
                "status": message.status,
                "run_id": message.run_id,
                "attachments": attachments,
                "created_at": message.created_at,
                "epoch": message.epoch,
            })
        return {
            "conversation": {name: getattr(record, name) for name in record.__dataclass_fields__},
            "messages": serialized_messages,
            "events": [{
                "id": event.id, "run_id": event.run_id, "type": event.event_type,
                "payload": event.payload, "created_at": event.created_at,
            } for event in events],
        }

    @api.patch("/api/conversations/{conversation_id}")
    async def rename_conversation(conversation_id: str, payload: ConversationRenameRequest, request: Request):
        record = await _conversation_record(request, conversation_id)
        if record.channel != Channel.WEB.value:
            raise HTTPException(status_code=403, detail="论坛会话标题不可修改")
        if not payload.title.strip():
            raise HTTPException(status_code=400, detail="标题不能为空")
        await _store(request).update_title(conversation_id, payload.title, custom=True)
        return {"status": "ok"}

    @api.post("/api/conversations/{conversation_id}/messages/stream")
    async def stream_message(conversation_id: str, request: Request):
        record = await _conversation_record(request, conversation_id)
        if record.channel != Channel.WEB.value:
            raise HTTPException(status_code=403, detail="论坛会话为只读")

        content_type = request.headers.get("content-type", "").lower()
        uploads = []
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            message = str(form.get("message") or "")
            uploads = [item for item in form.getlist("images") if hasattr(item, "read")]
        else:
            try:
                payload = ConversationMessageRequest.model_validate(await request.json())
            except Exception as exc:
                raise HTTPException(status_code=422, detail="消息请求格式无效") from exc
            message = payload.message

        if len(uploads) > MAX_IMAGES_PER_TURN:
            raise HTTPException(status_code=400, detail="每条消息最多上传 20 张图片")
        pending_uploads: list[tuple[bytes, str | None]] = []
        for upload in uploads:
            data = await upload.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail="单张图片不能超过 20MB")
            pending_uploads.append((data, getattr(upload, "filename", None)))
        if not message.strip() and not pending_uploads:
            raise HTTPException(status_code=400, detail="消息或图片不能为空")

        input_attachments: list[AttachmentRef] = []
        try:
            for data, filename in pending_uploads:
                artifact = await save_uploaded_image(
                    _store(request),
                    conversation_id=conversation_id,
                    data=data,
                    filename=filename,
                )
                input_attachments.append(
                    AttachmentRef(
                        artifact.uri,
                        artifact.mime_type,
                        artifact.artifact_id,
                        artifact.source_kind,
                        artifact.source_url,
                        artifact.filename,
                        artifact.width,
                        artifact.height,
                    )
                )
        except VisionMediaError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def events():
            def encode(event: str, data: dict[str, Any]) -> str:
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

            request_id = str(uuid.uuid4())
            reply_request = ReplyRequest(
                request_id=request_id,
                conversation=_ref_from_record(record),
                actor=ActorRef(Channel.WEB, record.external_id, "web-user", None),
                content=message,
                dispatch_mode=DispatchMode.AUTO,
                attachments=tuple(input_attachments),
            )
            reply_task = asyncio.create_task(
                request.app.state.container.bot_service.reply(reply_request)
            )
            last_event_id = 0
            try:
                while not reply_task.done():
                    run_events = await _store(request).list_events_for_request(request_id)
                    for event in run_events:
                        if event.id <= last_event_id:
                            continue
                        last_event_id = event.id
                        yield encode(event.event_type, {
                            "run_id": event.run_id,
                            "event_id": event.id,
                            "created_at": event.created_at,
                            **event.payload,
                        })
                    await asyncio.sleep(0.1)
                result = await reply_task
                run_events = await _store(request).list_events_for_request(request_id)
                for event in run_events:
                    if event.id <= last_event_id:
                        continue
                    last_event_id = event.id
                    yield encode(event.event_type, {
                        "run_id": event.run_id,
                        "event_id": event.id,
                        "created_at": event.created_at,
                        **event.payload,
                    })
                yield encode("message.completed", {
                    "text": result.text.replace("artifact://", "/api/artifacts/"),
                    "attachments": [
                        {
                            "artifact_id": a.name,
                            "url": f"/api/artifacts/{a.name}",
                            "mime_type": a.media_type,
                            "filename": a.filename,
                            "width": a.width,
                            "height": a.height,
                            "source_kind": a.source_kind or "generated",
                            "source_url": a.source_url,
                        }
                        for a in result.attachments if a.name
                    ],
                })
            except Exception as exc:
                logger.exception("网页对话失败")
                yield encode("stream.error", {"error": str(exc)})
            finally:
                if not reply_task.done():
                    reply_task.cancel()
                    await asyncio.gather(reply_task, return_exceptions=True)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @api.post("/api/conversations/{conversation_id}/clear")
    async def clear_managed_conversation(conversation_id: str, request: Request):
        record = await _conversation_record(request, conversation_id)
        await request.app.state.container.bot_service.clear_conversation(_ref_from_record(record))
        return {"status": "ok"}

    @api.delete("/api/conversations/{conversation_id}")
    async def delete_managed_conversation(conversation_id: str, request: Request):
        record = await _conversation_record(request, conversation_id)
        store = _store(request)
        remote_files = await store.list_provider_files_for_conversation(conversation_id)
        deepseek_key = None
        vault = getattr(request.app.state.container, "secret_vault", None)
        if vault is not None:
            scope = "forum" if record.channel == Channel.FORUM.value else "web"
            deepseek_key = await vault.get(f"{scope}:deepseek")
        deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            client = DeepSeekFilesClient(deepseek_key)
            for remote in remote_files:
                if remote["provider"] != "deepseek":
                    continue
                try:
                    await client.delete(remote["file_id"])
                except Exception:
                    logger.warning(
                        "删除 DeepSeek 远端文件失败，将等待其自动过期: %s",
                        remote["file_id"],
                    )
        paths = await store.delete_conversation(conversation_id)
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                logger.exception("删除 Artifact 文件失败: %s", path)
        return {"status": "ok"}

    @api.get("/api/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str, request: Request):
        artifact = await _store(request).get_artifact(artifact_id)
        if artifact is None or not artifact.available:
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(artifact.local_path, media_type=artifact.mime_type, filename=Path(artifact.local_path).name)

    def _profile_defaults(scope: str) -> dict[str, Any]:
        settings = AppSettings().providers
        prompt_scope = PromptScope.WEB if scope == "web" else PromptScope.FORUM
        prompt = FilePromptRepository().load("wolf_lumine", set(), prompt_scope).system_prompt
        return {
            "provider": "deepseek",
            "model": DEEPSEEK_VISION_MODEL,
            "fallback_model": None,
            "system_prompt": prompt,
            "enabled_tools": None,
            "disabled_mcp_tools": [],
        }

    @api.get("/api/settings/profiles")
    async def get_profiles(request: Request):
        store = _store(request)
        profiles = [await store.get_profile(scope, _profile_defaults(scope)) for scope in ("forum", "web")]
        vault = request.app.state.container.secret_vault
        env_names = {"openrouter": "OPENROUTER_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "tongyi": "DASHSCOPE_API_KEY", "mimo": "MIMO_API_KEY"}
        for profile in profiles:
            for value in (profile["draft"], profile["active"]):
                value["provider"] = "deepseek"
                value["model"] = DEEPSEEK_VISION_MODEL
                value["fallback_model"] = None
            provider = "deepseek"
            metadata = await vault.metadata(f"{profile['scope']}:{provider}") if vault else {"configured": False}
            if metadata.get("configured"):
                metadata["source"] = "ui"
            else:
                environment_value = os.getenv(env_names[provider])
                metadata.update({
                    "configured": bool(environment_value),
                    "source": "environment" if environment_value else None,
                    "last_four": environment_value[-4:] if environment_value else None,
                })
            profile["secret"] = metadata
        return profiles

    @api.put("/api/settings/profiles/{scope}/draft")
    async def save_profile(scope: str, payload: ProfileDraftRequest, request: Request):
        if scope not in {"forum", "web"}:
            raise HTTPException(status_code=404, detail="未知应用")
        if payload.provider != "deepseek":
            raise HTTPException(status_code=400, detail="视觉流程固定使用 DeepSeek")
        if payload.model not in {None, DEEPSEEK_VISION_MODEL}:
            raise HTTPException(status_code=400, detail="模型固定为 deepseek-v4-flash-vision-exp")
        value = payload.model_dump(exclude={"api_key"})
        value["model"] = DEEPSEEK_VISION_MODEL
        value["fallback_model"] = None
        await _store(request).get_profile(scope, _profile_defaults(scope))
        await _store(request).save_profile_draft(scope, value)
        if payload.api_key:
            await request.app.state.container.secret_vault.set(f"{scope}:{payload.provider}", payload.api_key)
        return {"status": "saved"}

    @api.post("/api/settings/profiles/{scope}/validate")
    async def validate_profile(scope: str, request: Request):
        profile = await _store(request).get_profile(scope, _profile_defaults(scope))
        draft = profile["draft"]
        errors = []
        if not draft.get("provider"):
            errors.append("Provider 不能为空")
        if not str(draft.get("system_prompt", "")).strip():
            errors.append("System Prompt 不能为空")
        return {"valid": not errors, "errors": errors}

    @api.post("/api/settings/profiles/{scope}/apply")
    async def apply_profile(scope: str, request: Request):
        validation = await validate_profile(scope, request)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["errors"])
        profile = await _store(request).get_profile(scope, _profile_defaults(scope))
        container = request.app.state.container
        prepared = None
        forum_candidate = None
        prepare_runtime = getattr(container, "prepare_runtime_profile", None)
        if scope == "web" and prepare_runtime is not None:
            try:
                prepared = await prepare_runtime(scope, profile["draft"])
            except Exception as exc:
                logger.exception("候选 Web Runtime 构建失败")
                raise HTTPException(status_code=400, detail=f"Runtime 构建失败: {exc}") from exc
        prepare_forum = getattr(container, "prepare_forum_runtime_profile", None)
        if scope == "forum" and prepare_forum is not None:
            try:
                forum_candidate = await prepare_forum(profile["draft"])
            except Exception as exc:
                logger.exception("候选 Forum Runtime 构建失败")
                raise HTTPException(status_code=400, detail=f"Runtime 构建失败: {exc}") from exc
        try:
            revision = await _store(request).apply_profile(scope)
        except Exception:
            if prepared is not None:
                await prepared[0].aclose()
            if forum_candidate is not None:
                await forum_candidate.aclose()
            raise
        if forum_candidate is not None:
            await forum_candidate.aclose()
        profile = await _store(request).get_profile(scope, _profile_defaults(scope))
        if prepared is not None:
            await container.activate_prepared_runtime(prepared)
        else:
            apply_runtime = getattr(container, "apply_runtime_profile", None)
            if apply_runtime is not None and scope == "web":
                await apply_runtime(scope, profile["active"])
        return {"status": "applied", "active_revision": revision}

    @api.post("/api/settings/profiles/{scope}/provider-test")
    async def provider_test(scope: str, request: Request):
        profile = await _store(request).get_profile(scope, _profile_defaults(scope))
        provider = profile["draft"].get("provider", "deepseek")
        secret = await request.app.state.container.secret_vault.get(f"{scope}:{provider}")
        env_names = {"openrouter": "OPENROUTER_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "tongyi": "DASHSCOPE_API_KEY", "mimo": "MIMO_API_KEY"}
        if not (secret or os.getenv(env_names[provider])):
            return {"ok": False, "message": "缺少 API Key"}
        candidate = None
        handler = None
        try:
            container = request.app.state.container
            if scope == "web" and hasattr(container, "prepare_runtime_profile"):
                handler, _service = await container.prepare_runtime_profile(
                    scope, profile["draft"]
                )
                candidate = getattr(getattr(handler, "_backend", None), "model", None)
            elif scope == "forum" and hasattr(container, "prepare_forum_runtime_profile"):
                candidate = await container.prepare_forum_runtime_profile(profile["draft"])
            if candidate is not None:
                await asyncio.wait_for(
                    candidate.llm.ainvoke("Reply with exactly: OK"), timeout=45
                )
            return {"ok": True, "message": "Provider 连接测试成功"}
        except Exception as exc:
            logger.exception("Provider connection test failed")
            return {"ok": False, "message": f"Provider 连接失败: {str(exc)[:300]}"}
        finally:
            if handler is not None:
                await handler.aclose()
            elif candidate is not None:
                await candidate.aclose()

    @api.post("/api/settings/profiles/{scope}/restore-default")
    async def restore_profile_default(scope: str, request: Request):
        if scope not in {"forum", "web"}:
            raise HTTPException(status_code=404, detail="未知应用")
        defaults = _profile_defaults(scope)
        await _store(request).get_profile(scope, defaults)
        await _store(request).save_profile_draft(scope, defaults)
        return {"status": "restored"}

    @api.get("/api/settings/tools/{scope}")
    async def get_tools(scope: str, request: Request):
        if scope not in {"forum", "web"}:
            raise HTTPException(status_code=404, detail="未知应用")
        catalog = await _store(request).list_tool_catalog(scope)
        if catalog:
            catalog = [item for item in catalog if item.get("source") != "mcp"]
            profile = await _store(request).get_profile(scope, _profile_defaults(scope))
            configured = profile["draft"].get("enabled_tools")
            if configured is not None:
                selected = set(configured)
                for item in catalog:
                    item["enabled"] = item["name"] in selected
            return catalog
        names = ["search_user", "search_user_by_id", "search_posts", "recent_posts", "search_posts_by_time", "get_post", "generate_image", "search_mention_memory", "manage_mention_memory"]
        profile = await _store(request).get_profile(scope, _profile_defaults(scope))
        configured = profile["draft"].get("enabled_tools")
        enabled = set(configured) if configured is not None else set(names)
        return [{"name": name, "enabled": name in enabled, "source": "runtime"} for name in names]

    @api.get("/api/settings/mcp/{scope}")
    async def get_mcp_status(scope: str, request: Request):
        if scope not in {"forum", "web"}:
            raise HTTPException(status_code=404, detail="未知应用")
        container = request.app.state.container
        url = container.settings.providers.mcp_server_url
        profile = await _store(request).get_profile(scope, _profile_defaults(scope))
        disabled = set(profile["draft"].get("disabled_mcp_tools", []))
        if not url:
            return {
                "url": None,
                "configured": False,
                "connected": False,
                "error": "MCP_SERVER_URL 未配置",
                "tools": [],
            }
        try:
            loaded_tools = await asyncio.wait_for(
                MentionChatModel._load_mcp_tools(url), timeout=15
            )
        except Exception as exc:
            logger.warning("MCP status probe failed for %s: %s", url, exc)
            return {
                "url": url,
                "configured": True,
                "connected": False,
                "error": str(exc)[:300],
                "tools": [],
            }
        return {
            "url": url,
            "configured": True,
            "connected": True,
            "error": None,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "enabled": tool.name not in disabled,
                }
                for tool in loaded_tools
            ],
        }

    static_dir = Path(__file__).with_name("static")
    if static_dir.is_dir():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            api.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

        @api.get("/favicon.ico", include_in_schema=False)
        async def frontend_favicon():
            return FileResponse(
                static_dir / "assets" / "favicon.ico",
                media_type="image/x-icon",
            )

        @api.get("/apple-touch-icon.png", include_in_schema=False)
        async def frontend_apple_touch_icon():
            return FileResponse(
                static_dir / "assets" / "apple-touch-icon.png",
                media_type="image/png",
            )

        @api.get("/", include_in_schema=False)
        async def frontend_index():
            return FileResponse(static_dir / "index.html")

        @api.get("/{frontend_path:path}", include_in_schema=False)
        async def frontend_history_fallback(frontend_path: str):
            if frontend_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            return FileResponse(static_dir / "index.html")

    return api


app = create_app()

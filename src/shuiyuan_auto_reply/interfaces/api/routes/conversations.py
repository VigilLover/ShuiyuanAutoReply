"""Conversation CRUD, their event/message timelines, and the web streaming reply."""

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shuiyuan_auto_reply.domain import (
    ActorRef,
    AttachmentRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ReplyRequest,
)
from shuiyuan_auto_reply.features.mention.deepseek_vision import (
    MAX_IMAGES_PER_TURN,
    DeepSeekFilesClient,
    VisionMediaError,
    save_uploaded_image,
)

from ..support import state_store

logger = logging.getLogger(__name__)
router = APIRouter()


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationRenameRequest(BaseModel):
    title: str


class ConversationMessageRequest(BaseModel):
    message: str


async def _conversation_record(request: Request, conversation_id: str):
    record = await state_store(request).get_conversation(conversation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return record


def _ref_from_record(record) -> ConversationRef:
    return ConversationRef(
        Channel(record.channel),
        record.external_id,
        record.bot_id,
        record.persona_id,
    )


@router.get("/api/conversations")
async def list_conversations(
    request: Request,
    channel: str | None = None,
    search: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    store = state_store(request)
    records = await store.list_conversations(
        channel=channel, search=search, limit=limit, offset=offset
    )
    counts = {}
    if channel in {None, "forum"}:
        monitor = await store.forum_monitor()
        counts = {
            item["id"]: {
                "queued_count": item["queued_count"],
                "running_count": item["running_count"],
            }
            for item in monitor["conversations"]
        }
    return [
        {
            **(
                record.__dict__
                if hasattr(record, "__dict__")
                else {
                    name: getattr(record, name) for name in record.__dataclass_fields__
                }
            ),
            **counts.get(record.id, {}),
        }
        for record in records
    ]


@router.post("/api/conversations")
async def create_conversation(payload: ConversationCreateRequest, request: Request):
    external_id = str(uuid.uuid4())
    ref = ConversationRef(Channel.WEB, external_id, "wolf_lumine", "wolf_lumine")
    record = await state_store(request).ensure_conversation(
        ref, title=payload.title or "新对话"
    )
    return {name: getattr(record, name) for name in record.__dataclass_fields__}


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    limit: int | None = None,
    before: str | None = None,
):
    store = state_store(request)
    record = await _conversation_record(request, conversation_id)
    # Without a limit the whole history is returned, as before. With one, only
    # the newest window is read so a long topic opens immediately.
    page = (
        await store.conversation_timeline_page(
            conversation_id, limit=limit, before=before
        )
        if limit is not None
        else None
    )
    if page is None:
        messages = await store.list_messages(conversation_id)
        events = await store.list_events_for_conversation(conversation_id)
        runs = await store.list_forum_runs(conversation_id)
    else:
        messages = page["messages"]
        events = page["events"]
        runs = page["runs"]
    serialized_messages = []
    for message in messages:
        attachments = []
        display_content = message.content
        for artifact_id in message.attachments:
            artifact = await store.get_artifact(artifact_id)
            if artifact and artifact.available:
                artifact_url = f"/api/artifacts/{artifact.id}"
                display_content = display_content.replace(
                    f"artifact://{artifact.id}", artifact_url
                )
                if artifact.forum_short_path:
                    display_content = display_content.replace(
                        artifact.forum_short_path, artifact_url
                    )
                attachments.append(
                    {
                        "artifact_id": artifact.id,
                        "url": f"/api/artifacts/{artifact.id}",
                        "mime_type": artifact.mime_type,
                        "filename": artifact.filename,
                        "width": artifact.width,
                        "height": artifact.height,
                        "source_kind": artifact.source_kind,
                        "source_url": artifact.source_url,
                    }
                )
        serialized_messages.append(
            {
                "id": message.id,
                "role": message.role,
                "content": display_content,
                "status": message.status,
                "run_id": message.run_id,
                "attachments": attachments,
                "created_at": message.created_at,
                "epoch": message.epoch,
            }
        )
    return {
        "conversation": {
            name: getattr(record, name) for name in record.__dataclass_fields__
        },
        "messages": serialized_messages,
        "runs": runs,
        "events": [
            {
                "id": event.id,
                "run_id": event.run_id,
                "type": event.event_type,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in events
        ],
        "has_more": bool(page and page["has_more"]),
        "next_cursor": page["next_cursor"] if page else None,
        "events_has_more": bool(page and page["events_has_more"]),
    }


@router.get("/api/conversations/{conversation_id}/events")
async def conversation_events(
    conversation_id: str,
    request: Request,
    limit: int = 200,
    before: int | None = None,
):
    await _conversation_record(request, conversation_id)
    page = await state_store(request).conversation_events_page(
        conversation_id, limit=limit, before=before
    )
    return {
        "events": [
            {
                "id": event.id,
                "run_id": event.run_id,
                "type": event.event_type,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in page["events"]
        ],
        "has_more": page["has_more"],
        "next_cursor": page["next_cursor"],
    }


@router.patch("/api/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str, payload: ConversationRenameRequest, request: Request
):
    record = await _conversation_record(request, conversation_id)
    if record.channel != Channel.WEB.value:
        raise HTTPException(status_code=403, detail="论坛会话标题不可修改")
    if not payload.title.strip():
        raise HTTPException(status_code=400, detail="标题不能为空")
    await state_store(request).update_title(conversation_id, payload.title, custom=True)
    return {"status": "ok"}


@router.post("/api/conversations/{conversation_id}/messages/stream")
async def stream_message(conversation_id: str, request: Request):
    record = await _conversation_record(request, conversation_id)
    if record.channel != Channel.WEB.value:
        raise HTTPException(status_code=403, detail="论坛会话为只读")

    content_type = request.headers.get("content-type", "").lower()
    uploads = []
    if content_type.startswith("multipart/form-data"):
        form = await request.form(
            max_files=20, max_fields=10, max_part_size=1024 * 1024
        )
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
    if not message.strip() and not uploads:
        raise HTTPException(status_code=400, detail="消息或图片不能为空")
    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

    limits = get_deployment().section("media")
    total_bytes = 0

    input_attachments: list[AttachmentRef] = []
    try:
        for upload in uploads:
            data = await upload.read(limits["max_image_bytes"] + 1)
            total_bytes += len(data)
            if (
                len(data) > limits["max_image_bytes"]
                or total_bytes > limits["max_turn_bytes"]
            ):
                raise HTTPException(
                    status_code=413, detail="图片超过单张或整轮大小限制"
                )
            filename = getattr(upload, "filename", None)
            artifact = await save_uploaded_image(
                state_store(request),
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
                run_events = await state_store(request).list_events_for_request(
                    request_id
                )
                for event in run_events:
                    if event.id <= last_event_id:
                        continue
                    last_event_id = event.id
                    yield encode(
                        event.event_type,
                        {
                            "run_id": event.run_id,
                            "event_id": event.id,
                            "created_at": event.created_at,
                            **event.payload,
                        },
                    )
                await asyncio.sleep(0.1)
            result = await reply_task
            run_events = await state_store(request).list_events_for_request(request_id)
            for event in run_events:
                if event.id <= last_event_id:
                    continue
                last_event_id = event.id
                yield encode(
                    event.event_type,
                    {
                        "run_id": event.run_id,
                        "event_id": event.id,
                        "created_at": event.created_at,
                        **event.payload,
                    },
                )
            yield encode(
                "message.completed",
                {
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
                        for a in result.attachments
                        if a.name
                    ],
                },
            )
        except Exception as exc:
            logger.exception("网页对话失败")
            yield encode("stream.error", {"error": str(exc)})
        finally:
            if not reply_task.done():
                reply_task.cancel()
                await asyncio.gather(reply_task, return_exceptions=True)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/api/conversations/{conversation_id}/clear")
async def clear_managed_conversation(conversation_id: str, request: Request):
    record = await _conversation_record(request, conversation_id)
    await request.app.state.container.bot_service.clear_conversation(
        _ref_from_record(record)
    )
    return {"status": "ok"}


@router.delete("/api/conversations/{conversation_id}")
async def delete_managed_conversation(conversation_id: str, request: Request):
    record = await _conversation_record(request, conversation_id)
    store = state_store(request)
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

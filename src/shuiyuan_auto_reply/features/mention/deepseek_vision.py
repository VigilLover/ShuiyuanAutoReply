"""DeepSeek Vision media ingestion, Files API transport, and tool-output parsing."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import io
import logging
import mimetypes
import re
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import httpx
from openai import AsyncOpenAI
from PIL import Image, UnidentifiedImageError

from shuiyuan_auto_reply.domain import AttachmentRef, VisualMediaArtifact
from shuiyuan_auto_reply.infrastructure.persistence.state import state_directory

from .mention_multimodal import extract_image_urls, normalize_shuiyuan_image_url


SUPPORTED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
MAX_IMAGES_PER_TURN = 20
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REMOTE_FILE_SECONDS = 7 * 24 * 60 * 60

_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*]\(\s*(?P<url>https?://[^)\s]+)", re.IGNORECASE
)
_HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>['\"]?)(?P<url>[^'\"\s>]+)(?P=quote)",
    re.IGNORECASE,
)
_PUBLIC_IMAGE_RE = re.compile(
    r"https?://[^\s<>)\"']+\.(?:jpe?g|png|gif|webp)(?:\?[^\s<>)\"']*)?",
    re.IGNORECASE,
)
_JSON_ESCAPED_SLASH_RE = re.compile(r"\\+(?:/|u002f)", re.IGNORECASE)
_TOOL_PAGE_URL_RE = re.compile(
    r"Contents of\s+(?P<url>https?://[^\s]+?):(?:\r?\n|$)", re.IGNORECASE
)
_IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
}


class VisionMediaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DeepSeekVisionInput:
    source_url: str
    source_kind: str
    content_block: dict[str, Any]
    artifact: VisualMediaArtifact
    description: str = ""


def sniff_image(data: bytes) -> tuple[str, int, int]:
    if len(data) > MAX_IMAGE_BYTES:
        raise VisionMediaError("单张图片不能超过 20MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise VisionMediaError("无法识别图片内容") from exc
    mime_type = Image.MIME.get(image_format)
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise VisionMediaError("仅支持 JPEG、PNG、GIF 和 WebP 图片")
    if width <= 0 or height <= 0 or max(width, height) > 8192:
        raise VisionMediaError("图片尺寸无效或长边超过 8192 像素")
    return mime_type, width, height


def extract_public_image_urls(value: Any) -> list[str]:
    """Extract explicit public image references without inventing new URLs."""
    texts: list[str] = []
    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            texts.extend(extract_public_image_urls(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(extract_public_image_urls(item))
    else:
        for name in ("raw", "cooked"):
            item = getattr(value, name, None)
            if isinstance(item, str):
                texts.append(item)
        image_urls = getattr(value, "image_urls", None)
        if image_urls:
            texts.extend(str(item) for item in image_urls)

    result: list[str] = []
    seen: set[str] = set()
    for text in texts:
        # JSON APIs commonly escape URL slashes as ``\/`` or ``\u002F``.
        # html.unescape() does not decode those forms, so normalize the text
        # before applying the explicit image URL patterns.
        normalized_text = _JSON_ESCAPED_SLASH_RE.sub("/", text)
        explicit_candidates = [
            *(
                match.group("url")
                for match in _MARKDOWN_IMAGE_RE.finditer(normalized_text)
            ),
            *(
                match.group("url")
                for match in _HTML_IMAGE_RE.finditer(normalized_text)
            ),
        ]
        raw_candidates = [
            match.group(0) for match in _PUBLIC_IMAGE_RE.finditer(normalized_text)
        ]

        # Booru-style APIs usually emit preview/sample/original URLs for every
        # post in that order. Prefer originals so the per-turn vision limit is
        # not consumed by multiple lower-resolution copies of the same image.
        def image_quality(candidate: str) -> int:
            path = urlparse(unescape(candidate)).path.lower()
            if "/thumbnails/" in path or "thumbnail_" in path:
                return 2
            if "/samples/" in path or "sample_" in path:
                return 1
            return 0

        raw_candidates.sort(key=image_quality)
        candidates = explicit_candidates + raw_candidates
        for candidate in candidates:
            url = unescape(candidate.strip().strip("<>").rstrip(".,;:"))
            parsed = urlparse(url)
            overlaps_explicit_url = any(
                existing.startswith(f"{url}/") for existing in result
            )
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or url in seen
                or overlaps_explicit_url
            ):
                continue
            seen.add(url)
            result.append(url)
    return result


def extract_tool_page_url(value: Any) -> str | None:
    """Find the source page URL inside nested MCP text content blocks."""
    if isinstance(value, str):
        match = _TOOL_PAGE_URL_RE.search(value)
        return match.group("url") if match else None
    if isinstance(value, dict):
        for item in value.values():
            if result := extract_tool_page_url(item):
                return result
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if result := extract_tool_page_url(item):
                return result
    return None


def extract_inline_images(value: Any) -> list[tuple[bytes, str, str]]:
    result: list[tuple[bytes, str, str]] = []
    blocks = value if isinstance(value, list) else [value]
    for block in blocks:
        if not isinstance(block, dict) or str(block.get("type", "")).lower() != "image":
            continue
        encoded = block.get("data")
        mime_type = block.get("mimeType") or block.get("mime_type") or "image/png"
        if not isinstance(encoded, str):
            continue
        try:
            result.append((base64.b64decode(encoded, validate=True), str(mime_type), "mcp-image"))
        except (ValueError, TypeError):
            continue
    return result


async def _assert_public_host(host: str) -> None:
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, None, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise VisionMediaError("图片域名无法解析") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise VisionMediaError("禁止访问内网或保留地址")


class DeepSeekFilesClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def upload(
        self, path: str | Path, *, mime_type: str, filename: str | None = None
    ) -> str:
        file_path = Path(path)
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.files.create(
                file=(filename or file_path.name, file_path.read_bytes(), mime_type),
                purpose="user_data",  # type: ignore[arg-type]
                expires_after={
                    "anchor": "created_at",
                    "seconds": REMOTE_FILE_SECONDS,
                },
                timeout=90,
            )
            file_id = response.id
        if not file_id:
            raise VisionMediaError("DeepSeek Files API 未返回 file_id")
        return str(file_id)

    async def delete(self, file_id: str) -> None:
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            await client.files.delete(file_id, timeout=30)


class DeepSeekVisionMediaManager:
    def __init__(self, *, state_store, forum_model, api_key: str) -> None:
        self.state_store = state_store
        self.forum_model = forum_model
        self.files = DeepSeekFilesClient(api_key)
        self.credential_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    async def _register_bytes(
        self,
        data: bytes,
        *,
        conversation_id: str | None,
        source_kind: str,
        source_url: str | None,
        filename: str | None,
    ) -> VisualMediaArtifact:
        mime_type, width, height = sniff_image(data)
        digest = hashlib.sha256(data).hexdigest()
        existing = await self.state_store.find_artifact_by_sha256(
            conversation_id, digest, source_kind
        )
        if existing is not None:
            return self.artifact_from_record(existing)
        extension = SUPPORTED_MIME_TYPES[mime_type]
        artifact_id = str(uuid.uuid4())
        output_dir = state_directory() / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{artifact_id}{extension}"
        path.write_bytes(data)
        try:
            await self.state_store.register_artifact(
                artifact_id=artifact_id,
                local_path=str(path),
                mime_type=mime_type,
                byte_count=len(data),
                width=width,
                height=height,
                conversation_id=conversation_id,
                source_kind=source_kind,
                source_url=source_url,
                filename=filename,
                sha256=digest,
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return VisualMediaArtifact(
            artifact_id=artifact_id,
            mime_type=mime_type,
            local_path=str(path),
            byte_count=len(data),
            source_kind=source_kind,
            source_url=source_url,
            filename=filename,
            width=width,
            height=height,
        )

    @staticmethod
    def artifact_from_record(record) -> VisualMediaArtifact:
        return VisualMediaArtifact(
            artifact_id=record.id,
            mime_type=record.mime_type,
            local_path=record.local_path,
            byte_count=record.byte_count,
            source_kind=record.source_kind,
            source_url=record.source_url,
            filename=record.filename,
            width=record.width,
            height=record.height,
        )

    async def ensure_file_id(self, artifact: VisualMediaArtifact) -> str:
        cached = await self.state_store.get_provider_file(
            artifact.artifact_id, "deepseek", self.credential_fingerprint
        )
        now = datetime.now(UTC)
        if cached:
            expires_at = datetime.fromisoformat(cached["expires_at"])
            if expires_at > now + timedelta(minutes=5):
                return cached["file_id"]
        file_id = await self.files.upload(
            artifact.local_path,
            mime_type=artifact.mime_type,
            filename=artifact.filename,
        )
        expires_at = now + timedelta(seconds=REMOTE_FILE_SECONDS)
        await self.state_store.upsert_provider_file(
            artifact_id=artifact.artifact_id,
            provider="deepseek",
            credential_fingerprint=self.credential_fingerprint,
            file_id=file_id,
            expires_at=expires_at.isoformat(),
        )
        return file_id

    async def prepare_attachment(self, attachment: AttachmentRef) -> DeepSeekVisionInput:
        if not attachment.url.startswith("artifact://"):
            raise VisionMediaError("用户附件必须引用本地 Artifact")
        artifact_id = attachment.url.removeprefix("artifact://")
        record = await self.state_store.get_artifact(artifact_id)
        if record is None or not record.available:
            raise VisionMediaError("用户上传图片不存在")
        artifact = self.artifact_from_record(record)
        file_id = await self.ensure_file_id(artifact)
        return DeepSeekVisionInput(
            source_url=artifact.source_url or artifact.uri,
            source_kind=artifact.source_kind,
            content_block={"type": "file", "file_id": file_id},
            artifact=artifact,
            description=artifact.filename or "用户上传图片",
        )

    async def prepare_forum_url(
        self,
        url: str,
        *,
        conversation_id: str | None,
        source_kind: str,
        description: str = "",
    ) -> DeepSeekVisionInput | None:
        normalized = normalize_shuiyuan_image_url(url)
        if normalized is None:
            return None
        if normalized.startswith("upload://"):
            data = await self.forum_model.download_image(normalized)
        else:
            data = await self.forum_model.download_raw_image(normalized)
        artifact = await self._register_bytes(
            data,
            conversation_id=conversation_id,
            source_kind=source_kind,
            source_url=normalized,
            filename=Path(urlparse(normalized).path).name or None,
        )
        file_id = await self.ensure_file_id(artifact)
        return DeepSeekVisionInput(
            source_url=normalized,
            source_kind=source_kind,
            content_block={"type": "file", "file_id": file_id},
            artifact=artifact,
            description=description,
        )

    async def prepare_inline(
        self,
        data: bytes,
        *,
        conversation_id: str | None,
        source_kind: str,
        filename: str,
    ) -> DeepSeekVisionInput:
        artifact = await self._register_bytes(
            data,
            conversation_id=conversation_id,
            source_kind=source_kind,
            source_url=None,
            filename=filename,
        )
        file_id = await self.ensure_file_id(artifact)
        return DeepSeekVisionInput(
            source_url=artifact.uri,
            source_kind=source_kind,
            content_block={"type": "file", "file_id": file_id},
            artifact=artifact,
            description=filename,
        )

    async def prepare_public_url(
        self,
        url: str,
        *,
        conversation_id: str | None,
        source_kind: str = "web_search",
        description: str = "",
        referer: str | None = None,
    ) -> DeepSeekVisionInput:
        current = url
        response_headers: dict[str, str] = {}
        headers = dict(_IMAGE_REQUEST_HEADERS)
        if referer and referer.startswith(("http://", "https://")):
            headers["Referer"] = referer
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=10), headers=headers
        ) as client:
            for _ in range(5):
                parsed = urlparse(current)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise VisionMediaError("图片 URL 无效")
                await _assert_public_host(parsed.hostname)
                async with client.stream("GET", current, follow_redirects=False) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise VisionMediaError("图片重定向缺少目标地址")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    declared_length = int(response.headers.get("content-length", "0") or 0)
                    if declared_length > MAX_IMAGE_BYTES:
                        raise VisionMediaError("单张图片不能超过 20MB")
                    chunks: list[bytes] = []
                    byte_count = 0
                    async for chunk in response.aiter_bytes():
                        byte_count += len(chunk)
                        if byte_count > MAX_IMAGE_BYTES:
                            raise VisionMediaError("单张图片不能超过 20MB")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    response_headers = dict(response.headers)
                break
            else:
                raise VisionMediaError("图片重定向次数过多")
        artifact = await self._register_bytes(
            data,
            conversation_id=conversation_id,
            source_kind=source_kind,
            source_url=url,
            filename=Path(urlparse(url).path).name or mimetypes.guess_extension(
                response_headers.get("content-type", "").split(";", 1)[0]
            ),
        )
        file_id = await self.ensure_file_id(artifact)
        return DeepSeekVisionInput(
            source_url=url,
            source_kind=source_kind,
            content_block={"type": "file", "file_id": file_id},
            artifact=artifact,
            description=description,
        )

    async def prepare_tool_output(
        self,
        messages: Iterable[Any],
        *,
        conversation_id: str | None,
        existing_urls: set[str],
        limit: int,
    ) -> list[DeepSeekVisionInput]:
        results: list[DeepSeekVisionInput] = []
        for message in messages:
            name = str(getattr(message, "name", "") or "")
            content = getattr(message, "content", "")
            artifact_value = getattr(message, "artifact", None)
            source_kind = (
                "forum_search"
                if name in {"search_posts", "recent_posts", "search_posts_by_time", "get_post"}
                else "web_search"
            )
            for data, _claimed_mime, filename in extract_inline_images(content):
                if len(results) >= limit:
                    return results
                image = await self.prepare_inline(
                    data,
                    conversation_id=conversation_id,
                    source_kind=source_kind,
                    filename=filename,
                )
                if image.source_url not in existing_urls:
                    existing_urls.add(image.source_url)
                    results.append(image)

            combined = [content, artifact_value]
            referer = extract_tool_page_url(combined)
            private_urls: list[str] = []
            for item in combined:
                if isinstance(item, str):
                    private_urls.extend(extract_image_urls(item))
                else:
                    for candidate in self._object_image_urls(item):
                        normalized = normalize_shuiyuan_image_url(candidate)
                        if normalized:
                            private_urls.append(normalized)
            for private_url in private_urls:
                if len(results) >= limit:
                    return results
                if private_url in existing_urls:
                    continue
                try:
                    image = await self.prepare_forum_url(
                        private_url,
                        conversation_id=conversation_id,
                        source_kind="forum_search",
                        description=f"来自 {name or '论坛工具'}",
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to cache forum search image %s from %s: %s",
                        private_url,
                        name or "forum tool",
                        exc,
                    )
                    continue
                if image:
                    existing_urls.add(private_url)
                    results.append(image)

            for public_url in extract_public_image_urls(combined):
                if len(results) >= limit:
                    return results
                if public_url in existing_urls:
                    continue
                try:
                    image = await self.prepare_public_url(
                        public_url,
                        conversation_id=conversation_id,
                        source_kind=source_kind,
                        description=f"来自 {name or '网页工具'}",
                        referer=referer,
                    )
                except Exception as exc:
                    logging.warning(
                        "Failed to cache web search image %s from %s: %s",
                        public_url,
                        name or "web tool",
                        exc,
                    )
                    continue
                existing_urls.add(public_url)
                results.append(image)
        return results

    @classmethod
    def _object_image_urls(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            result = [str(item) for item in value.get("image_urls", []) or []]
            for item in value.values():
                result.extend(cls._object_image_urls(item))
            return result
        if isinstance(value, (list, tuple, set)):
            result: list[str] = []
            for item in value:
                result.extend(cls._object_image_urls(item))
            return result
        return [str(item) for item in getattr(value, "image_urls", []) or []]


def build_deepseek_content(
    text: str, images: Iterable[DeepSeekVisionInput]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, 1):
        label = image.description or image.source_url
        if image.source_kind in {"web_search", "forum_search"}:
            label = (
                f"{label}；展示标识 {image.artifact.uri}。"
                "最终回复需要展示此图时，只能把该展示标识作为图片地址"
            )
        content.append({"type": "text", "text": f"【图片 {index}：{label}】"})
        content.append(image.content_block)
    if text:
        content.append({"type": "text", "text": text})
    return content


async def save_uploaded_image(
    state_store,
    *,
    conversation_id: str,
    data: bytes,
    filename: str | None,
) -> VisualMediaArtifact:
    """Validate and persist a browser upload without exposing a provider client."""
    mime_type, width, height = sniff_image(data)
    digest = hashlib.sha256(data).hexdigest()
    existing = await state_store.find_artifact_by_sha256(
        conversation_id, digest, "user_upload"
    )
    if existing is not None:
        return DeepSeekVisionMediaManager.artifact_from_record(existing)
    artifact_id = str(uuid.uuid4())
    extension = SUPPORTED_MIME_TYPES[mime_type]
    output_dir = state_directory() / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{artifact_id}{extension}"
    path.write_bytes(data)
    try:
        await state_store.register_artifact(
            artifact_id=artifact_id,
            local_path=str(path),
            mime_type=mime_type,
            byte_count=len(data),
            width=width,
            height=height,
            conversation_id=conversation_id,
            source_kind="user_upload",
            filename=filename,
            sha256=digest,
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return VisualMediaArtifact(
        artifact_id=artifact_id,
        mime_type=mime_type,
        local_path=str(path),
        byte_count=len(data),
        source_kind="user_upload",
        filename=filename,
        width=width,
        height=height,
    )

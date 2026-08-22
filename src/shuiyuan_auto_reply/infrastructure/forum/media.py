"""Forum publication of local image artifacts."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Protocol

from PIL import Image

from shuiyuan_auto_reply.shuiyuan.objects import normalize_upload_short_path


_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(\s*(?P<open><)?(?P<target>[^\s)>]+)(?(open)>)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[(?P<label>[^\]]*)\]\(\s*(?P<open><)?(?P<target>[^\s)>]+)(?(open)>)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)",
    re.IGNORECASE,
)
_HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>['\"]?)(?P<target>[^'\"\s>]+)(?P=quote)[^>]*>",
    re.IGNORECASE,
)
_HTML_LINK_RE = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*(?P<quote>['\"]?)(?P<target>[^'\"\s>]+)(?P=quote)[^>]*>.*?</a\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_ALT_RE = re.compile(
    r"\balt\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))",
    re.IGNORECASE,
)
_ESCAPED_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\\\(\s*(?P<target>[^\s)]+)\\?\)",
    re.IGNORECASE,
)
_UPLOAD_URI_RE = re.compile(r"upload://[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class LocalImageArtifact(Protocol):
    artifact_id: str
    mime_type: str
    local_path: str

    @property
    def uri(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ForumMediaUpload:
    short_path: str
    reused: bool = False


@dataclass(frozen=True, slots=True)
class PublishedForumMedia:
    artifact: LocalImageArtifact
    short_path: str
    source_kind: str
    reused: bool


@dataclass(frozen=True, slots=True)
class ForumMediaFailure:
    artifact: LocalImageArtifact
    source_kind: str
    error: str


@dataclass(frozen=True, slots=True)
class ForumPublicationResult:
    text: str
    published: tuple[PublishedForumMedia, ...] = ()
    failures: tuple[ForumMediaFailure, ...] = ()


class ForumMediaUploader:
    def __init__(self, forum_model, state_store=None) -> None:
        self.forum_model = forum_model
        self.state_store = state_store

    async def upload(self, artifact: LocalImageArtifact) -> ForumMediaUpload:
        if self.state_store is not None:
            try:
                record = await self.state_store.get_artifact(artifact.artifact_id)
                if record is not None and record.forum_short_path:
                    cached_path = str(record.forum_short_path).strip()
                    try:
                        return ForumMediaUpload(
                            self._require_upload_uri(cached_path), reused=True
                        )
                    except ValueError:
                        logging.warning(
                            "Ignoring invalid cached forum path for artifact %s: %r",
                            artifact.artifact_id,
                            cached_path,
                        )
            except Exception:
                logging.exception(
                    "Failed to read cached forum path for artifact %s; uploading again",
                    artifact.artifact_id,
                )

        image_bytes = Path(artifact.local_path).read_bytes()
        mime_type = (
            artifact.mime_type
            if artifact.mime_type in _MIME_EXTENSIONS
            else "image/jpeg"
        )
        filename = getattr(artifact, "filename", None) or (
            f"{artifact.artifact_id}{_MIME_EXTENSIONS.get(mime_type, '.jpg')}"
        )
        try:
            response = await self.forum_model.upload_image(
                image_bytes,
                mime_type=mime_type,
                filename=filename,
            )
            short_path = self._require_upload_uri(response.short_path)
        except Exception as original_error:
            logging.warning(
                "Original forum image upload failed for %s (%s); retrying as JPEG: %s",
                artifact.artifact_id,
                mime_type,
                original_error,
            )
            jpeg_bytes = self._as_jpeg(image_bytes)
            try:
                response = await self.forum_model.upload_image(
                    jpeg_bytes,
                    mime_type="image/jpeg",
                    filename=f"{artifact.artifact_id}.jpg",
                )
                short_path = self._require_upload_uri(response.short_path)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"原格式上传失败: {original_error}; JPEG 降级上传失败: {fallback_error}"
                ) from fallback_error

        if self.state_store is not None:
            try:
                await self.state_store.set_forum_short_path(
                    artifact.artifact_id, short_path
                )
            except Exception:
                logging.exception(
                    "Forum upload succeeded but cache update failed for artifact %s",
                    artifact.artifact_id,
                )
        return ForumMediaUpload(short_path)

    @staticmethod
    def _require_upload_uri(value: object) -> str:
        short_path = normalize_upload_short_path(unescape(str(value).strip()))
        if not _UPLOAD_URI_RE.fullmatch(short_path):
            raise ValueError(
                "水源上传接口没有返回有效的 upload:// 短地址: "
                f"{short_path!r}"
            )
        return short_path

    @staticmethod
    def _as_jpeg(image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.seek(0)
            image.load()
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                converted = Image.new("RGB", rgba.size, "white")
                converted.paste(rgba, mask=rgba.getchannel("A"))
            else:
                converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=88, optimize=True)
            return buffer.getvalue()


class ForumReplyMediaPublisher:
    """Upload only the local artifacts explicitly selected by the final reply."""

    _SEARCH_SOURCE_KINDS = {"web_search", "forum_search"}

    def __init__(self, uploader: ForumMediaUploader, state_store=None) -> None:
        self.uploader = uploader
        self.state_store = state_store

    async def publish(
        self,
        text: str,
        artifacts: tuple[LocalImageArtifact, ...] | list[LocalImageArtifact],
    ) -> ForumPublicationResult:
        current = self._normalize_escaped_image_markdown(text)
        image_targets = self._image_targets(current)
        selected: list[tuple[LocalImageArtifact, str]] = []
        seen_ids: set[str] = set()
        for artifact in artifacts:
            source_kind = str(getattr(artifact, "source_kind", "generated"))
            source_url = getattr(artifact, "source_url", None)
            is_generated = source_kind == "generated"
            is_selected_search = source_kind in self._SEARCH_SOURCE_KINDS and (
                artifact.uri in current
                or (source_url is not None and source_url in image_targets)
            )
            if artifact.artifact_id in seen_ids or not (
                is_generated or is_selected_search
            ):
                continue
            seen_ids.add(artifact.artifact_id)
            selected.append((artifact, source_kind))

        published: list[PublishedForumMedia] = []
        failures: list[ForumMediaFailure] = []
        for artifact, source_kind in selected:
            source_url = getattr(artifact, "source_url", None)
            try:
                if source_kind == "forum_search" and self._is_local_upload_path(
                    source_url
                ):
                    upload = ForumMediaUpload(
                        ForumMediaUploader._require_upload_uri(source_url), reused=True
                    )
                    if self.state_store is not None:
                        try:
                            await self.state_store.set_forum_short_path(
                                artifact.artifact_id, upload.short_path
                            )
                        except Exception:
                            logging.exception(
                                "Failed to cache reused forum path for artifact %s",
                                artifact.artifact_id,
                            )
                else:
                    upload = await self.uploader.upload(artifact)
                replacement = ForumMediaUploader._require_upload_uri(
                    upload.short_path
                )
                current = self._rewrite_artifact(
                    current,
                    artifact,
                    replacement=replacement,
                    failed=False,
                )
                if source_url and str(source_url).startswith(
                    ("http://", "https://")
                ):
                    current = self._remove_source_references(
                        current, str(source_url)
                    )
                published.append(
                    PublishedForumMedia(
                        artifact, replacement, source_kind, upload.reused
                    )
                )
            except Exception as exc:
                logging.warning(
                    "Forum publication upload failed for artifact %s: %s",
                    artifact.artifact_id,
                    exc,
                )
                current = self._rewrite_artifact(
                    current,
                    artifact,
                    replacement=None,
                    failed=True,
                )
                failures.append(ForumMediaFailure(artifact, source_kind, str(exc)))

        current = self._enforce_local_upload_images(current)
        return ForumPublicationResult(current, tuple(published), tuple(failures))

    @staticmethod
    def _is_local_upload_path(value: object) -> bool:
        try:
            ForumMediaUploader._require_upload_uri(value)
        except ValueError:
            return False
        return True

    @staticmethod
    def _normalized_target(value: str) -> str:
        target = unescape(value.strip().removeprefix("<").removesuffix(">"))
        return re.sub(r"\\([./])", r"\1", target)

    @classmethod
    def _normalize_escaped_image_markdown(cls, text: str) -> str:
        def image(match: re.Match[str]) -> str:
            target = cls._normalized_target(match.group("target"))
            return f"![{match.group('alt')}]({target})"

        return _ESCAPED_MARKDOWN_IMAGE_RE.sub(image, text)

    @staticmethod
    def _html_alt(tag: str) -> str:
        match = _HTML_ALT_RE.search(tag)
        if not match:
            return "图片"
        return unescape(
            match.group("double") or match.group("single") or match.group("bare") or "图片"
        )

    @classmethod
    def _image_targets(cls, text: str) -> set[str]:
        return {
            cls._normalized_target(match.group("target"))
            for pattern in (_MARKDOWN_IMAGE_RE, _HTML_IMAGE_RE)
            for match in pattern.finditer(text)
        }

    @classmethod
    def _rewrite_artifact(
        cls,
        text: str,
        artifact: LocalImageArtifact,
        *,
        replacement: str | None,
        failed: bool,
    ) -> str:
        source_url = getattr(artifact, "source_url", None)
        targets = {artifact.uri}
        if source_url:
            targets.add(str(source_url))

        def markdown(match: re.Match[str]) -> str:
            if cls._normalized_target(match.group("target")) not in targets:
                return match.group(0)
            if not failed and replacement:
                return match.group(0).replace(
                    match.group("target"), replacement, 1
                )
            return "（图片上传失败）"

        def html_image(match: re.Match[str]) -> str:
            if cls._normalized_target(match.group("target")) not in targets:
                return match.group(0)
            if not failed and replacement:
                alt = cls._html_alt(match.group(0))
                return f"![{alt or '图片'}]({replacement})"
            return "（图片上传失败）"

        rewritten = _MARKDOWN_IMAGE_RE.sub(markdown, text)
        rewritten = _HTML_IMAGE_RE.sub(html_image, rewritten)
        if artifact.uri in rewritten:
            if not failed and replacement:
                rewritten = rewritten.replace(artifact.uri, replacement)
            else:
                rewritten = rewritten.replace(artifact.uri, "（图片上传失败）")
        return rewritten

    @classmethod
    def _remove_source_references(cls, text: str, source_url: str) -> str:
        def markdown_link(match: re.Match[str]) -> str:
            return (
                ""
                if cls._normalized_target(match.group("target")) == source_url
                else match.group(0)
            )

        def html_link(match: re.Match[str]) -> str:
            return (
                ""
                if cls._normalized_target(match.group("target")) == source_url
                else match.group(0)
            )

        rewritten = _MARKDOWN_LINK_RE.sub(markdown_link, text)
        return _HTML_LINK_RE.sub(html_link, rewritten)

    @classmethod
    def _enforce_local_upload_images(cls, text: str) -> str:
        """Keep uploaded images as Markdown using canonical upload:// URLs."""

        def markdown(match: re.Match[str]) -> str:
            target = cls._normalized_target(match.group("target"))
            try:
                target = ForumMediaUploader._require_upload_uri(target)
            except ValueError:
                return "（图片本地化或上传失败）"
            alt = match.group("alt").strip() or "图片"
            return f"![{alt}]({target})"

        def html_image(match: re.Match[str]) -> str:
            target = cls._normalized_target(match.group("target"))
            try:
                target = ForumMediaUploader._require_upload_uri(target)
            except ValueError:
                return "（图片本地化或上传失败）"
            alt = cls._html_alt(match.group(0))
            return f"![{alt or '图片'}]({target})"

        return _HTML_IMAGE_RE.sub(
            html_image, _MARKDOWN_IMAGE_RE.sub(markdown, text)
        )

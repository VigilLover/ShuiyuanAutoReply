import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from html import unescape
from typing import Iterable, Sequence
from urllib.parse import urlparse

from PIL import Image, ImageOps, UnidentifiedImageError


logger = logging.getLogger(__name__)

SHUIYUAN_HOSTS = {"shuiyuan.sjtu.edu.cn"}
UPLOAD_SHORT_PATH_PREFIX = "/uploads/short-url/"
USER_AVATAR_PATH_PREFIX = "/user_avatar/"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif")

DEFAULT_MAX_LONG_EDGE = 1024
DEFAULT_JPEG_QUALITY = 82
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 20 * 1024 * 1024

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\(\s*(?P<url>[^)\s]+)(?:\s+['\"][^'\"]*['\"])?\s*\)")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>['\"]?)(?P<url>[^'\"\s>]+)(?P=quote)", re.IGNORECASE)
_RAW_UPLOAD_RE = re.compile(r"(?<![\w/])upload://[^\s<>)\"']+", re.IGNORECASE)
_RAW_SHORT_PATH_RE = re.compile(r"(?<![\w-])/uploads/short-url/[^\s<>)\"']+", re.IGNORECASE)
_RAW_SHUIYUAN_URL_RE = re.compile(
    r"https?://shuiyuan\.sjtu\.edu\.cn/uploads/short-url/[^\s<>)\"']+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MentionImageInput:
    source_url: str
    data_url: str
    origin: str
    mime_type: str
    byte_count: int
    description: str = ""


@dataclass(frozen=True)
class ImageInspectResult:
    image_urls: list[str]
    source: str = "inspect_image"
    description: str = ""


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using %s", name, value, default)
        return default
    return parsed if parsed > 0 else default


def _env_int_with_fallback(name: str, fallback_name: str, default: int) -> int:
    if os.getenv(name) is not None:
        return _env_int(name, default)
    return _env_int(fallback_name, default)


def _is_probable_image_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path if parsed.scheme in {"http", "https"} else url
    return path.lower().endswith(IMAGE_EXTENSIONS)


def _strip_url_wrapping(url: str) -> str:
    return unescape(url.strip().strip("<>").rstrip(".,;:"))


def normalize_shuiyuan_image_url(url: str) -> str | None:
    candidate = _strip_url_wrapping(url)
    if not candidate:
        return None

    if candidate.startswith("upload://"):
        return candidate if _is_probable_image_url(candidate) else None

    if candidate.startswith(UPLOAD_SHORT_PATH_PREFIX):
        normalized = "upload://" + candidate[len(UPLOAD_SHORT_PATH_PREFIX):]
        return normalized if _is_probable_image_url(normalized) else None

    if candidate.startswith(USER_AVATAR_PATH_PREFIX):
        return candidate if _is_probable_image_url(candidate) else None

    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() not in SHUIYUAN_HOSTS:
            return None
        if parsed.path.startswith(UPLOAD_SHORT_PATH_PREFIX):
            filename = parsed.path[len(UPLOAD_SHORT_PATH_PREFIX):]
            normalized = "upload://" + filename
            return normalized if _is_probable_image_url(normalized) else None
        if parsed.path.startswith(USER_AVATAR_PATH_PREFIX):
            return parsed.path if _is_probable_image_url(parsed.path) else None

    return None


def extract_image_urls(text: str | None) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []
    for pattern in (_MARKDOWN_IMAGE_RE, _HTML_IMG_RE, _RAW_SHUIYUAN_URL_RE, _RAW_SHORT_PATH_RE, _RAW_UPLOAD_RE):
        for match in pattern.finditer(text):
            candidates.append(match.group("url") if "url" in match.groupdict() else match.group(0))

    seen: set[str] = set()
    normalized_urls: list[str] = []
    for candidate in candidates:
        normalized = normalize_shuiyuan_image_url(candidate)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        normalized_urls.append(normalized)
    return normalized_urls


def _compress_to_jpeg_data_url(image_bytes: bytes, *, max_long_edge: int, jpeg_quality: int) -> tuple[str, str, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if "A" in image.getbands():
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        longest_edge = max(image.size)
        if longest_edge > max_long_edge:
            image.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded_bytes = output.getvalue()

    encoded = base64.b64encode(encoded_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", "image/jpeg", len(encoded_bytes)


async def prepare_image_input(
    image_url: str,
    *,
    shuiyuan_model,
    origin: str,
    description: str = "",
    max_image_bytes: int | None = None,
    max_long_edge: int | None = None,
    jpeg_quality: int | None = None,
) -> MentionImageInput | None:
    normalized = normalize_shuiyuan_image_url(image_url)
    if normalized is None:
        return None

    if shuiyuan_model is None:
        logger.warning("No ShuiyuanModel available, cannot download image: %s", normalized)
        return None

    max_bytes = max_image_bytes if max_image_bytes is not None else _env_int_with_fallback(
        "MIMO_MULTIMODAL_MAX_IMAGE_BYTES",
        "MIMO_MAX_IMAGE_BYTES",
        DEFAULT_MAX_IMAGE_BYTES,
    )
    long_edge = max_long_edge if max_long_edge is not None else _env_int_with_fallback(
        "MIMO_MULTIMODAL_MAX_LONG_EDGE",
        "MIMO_IMAGE_MAX_LONG_EDGE",
        DEFAULT_MAX_LONG_EDGE,
    )
    quality = jpeg_quality if jpeg_quality is not None else _env_int_with_fallback(
        "MIMO_MULTIMODAL_JPEG_QUALITY",
        "MIMO_IMAGE_JPEG_QUALITY",
        DEFAULT_JPEG_QUALITY,
    )
    quality = max(1, min(95, quality))

    try:
        if normalized.startswith("upload://"):
            image_bytes = await shuiyuan_model.download_image(normalized)
        else:
            image_bytes = await shuiyuan_model.download_raw_image(normalized)
    except Exception as exc:
        logger.warning("Shuiyuan download_image failed for %s: %s", normalized, exc)
        return None

    if len(image_bytes) > max_bytes:
        logger.warning("Downloaded image too large for MiMo input: %s bytes from %s", len(image_bytes), normalized)
        return None

    try:
        data_url, mime_type, byte_count = _compress_to_jpeg_data_url(
            image_bytes,
            max_long_edge=long_edge,
            jpeg_quality=quality,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Failed to encode MiMo image input for %s: %s", normalized, exc)
        return None

    if byte_count > max_bytes:
        logger.warning("Compressed image too large for MiMo input: %s bytes from %s", byte_count, normalized)
        return None

    return MentionImageInput(
        source_url=normalized,
        data_url=data_url,
        origin=origin,
        mime_type=mime_type,
        byte_count=byte_count,
        description=description,
    )


async def collect_post_image_inputs(
    posts: Iterable[object],
    *,
    shuiyuan_model,
    origin: str,
    max_images: int = 4,
    max_total_bytes: int | None = None,
    existing_urls: Sequence[str] = (),
    existing_byte_count: int = 0,
) -> list[MentionImageInput]:
    total_limit = max_total_bytes if max_total_bytes is not None else _env_int_with_fallback(
        "MIMO_MULTIMODAL_MAX_TOTAL_BYTES",
        "MIMO_MAX_TOTAL_IMAGE_BYTES",
        DEFAULT_MAX_TOTAL_BYTES,
    )
    images: list[MentionImageInput] = []
    seen: set[str] = {
        normalized
        for url in existing_urls
        if (normalized := normalize_shuiyuan_image_url(url)) is not None
    }
    total_bytes = max(0, existing_byte_count)
    if total_bytes >= total_limit:
        logger.warning("Skipping MiMo image input because total image bytes cap is already reached")
        return images

    for post in posts:
        if isinstance(post, dict):
            post_image_urls = list(post.get("image_urls", []) or [])
        else:
            post_image_urls = list(getattr(post, "image_urls", []) or [])
        post_description = (
            post.get("description", "") if isinstance(post, dict)
            else getattr(post, "description", "")
        )
        for field_name in ("raw", "cooked"):
            field_value = post.get(field_name) if isinstance(post, dict) else getattr(post, field_name, None)
            if isinstance(field_value, str):
                post_image_urls.extend(extract_image_urls(field_value))

        for image_url in post_image_urls:
            normalized = normalize_shuiyuan_image_url(image_url)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            if len(images) >= max_images:
                return images

            image = await prepare_image_input(
                normalized,
                shuiyuan_model=shuiyuan_model,
                origin=origin,
                description=post_description,
            )
            if image is None:
                continue
            if total_bytes + image.byte_count > total_limit:
                logger.warning("Skipping MiMo image input because total image bytes cap would be exceeded")
                return images

            images.append(image)
            total_bytes += image.byte_count

    return images


def build_mimo_content(text: str, images: Iterable[MentionImageInput]) -> list[dict]:
    content: list[dict] = []
    for image in images:
        if image.description:
            content.append({"type": "text", "text": f"【{image.description}】"})
        content.append({"type": "image_url", "image_url": {"url": image.data_url}})
    if text:
        content.append({"type": "text", "text": text})
    return content

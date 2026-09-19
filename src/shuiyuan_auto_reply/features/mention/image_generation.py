import asyncio
import base64
import io
import ipaddress
import json
import logging
import math
import os
import re
import socket as _socket
import time
import uuid
import weakref
from urllib.parse import urljoin, urlsplit

import aiohttp
from PIL import Image

from shuiyuan_auto_reply.application.tool_results import current_turn
from shuiyuan_auto_reply.domain import GeneratedImageArtifact
from shuiyuan_auto_reply.infrastructure.image_transport import (
    ImageDownloadError,
    cached_media_attempt,
    encoded_image_url,
    remembered_media,
)
from shuiyuan_auto_reply.infrastructure.persistence.state import state_directory

from .mention_multimodal import normalize_shuiyuan_image_url

# Single and combined reference-image limits. Base64 increases the wire size.
_MAX_REFERENCE_BYTES = 10 * 1024 * 1024
_MAX_TOTAL_REFERENCE_BYTES = 20 * 1024 * 1024
# Reference image compression: resize to at most this many pixels on the longest edge.
# Reduces server-side processing time to stay under the ~60s Cloudflare/Caddy proxy timeout.
_MAX_REFERENCE_LONG_EDGE = 1024
_REFERENCE_JPEG_QUALITY = 80
_DEFAULT_IMAGE_MODEL = "gpt-image-2"
_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_MAX_API_ATTEMPTS = 3
_MAX_CONFIGURED_API_ATTEMPTS = 10
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 5.0
_RETRYABLE_HTTP_STATUSES = {408, 429}
_TCP_KEEPALIVE_IDLE_SECONDS = 30  # macOS: TCP_KEEPALIVE; Linux: TCP_KEEPIDLE
_TCP_KEEPALIVE_INTERVAL_SECONDS = 15
_TCP_KEEPALIVE_COUNT = 3
_NATIVE_SIZE_ALIGNMENT = 16
_NATIVE_MIN_PIXELS = 655_360
_NATIVE_MAX_PIXELS = 8_294_400
_NATIVE_MAX_EDGE = 3840
_NATIVE_MAX_ASPECT_RATIO = 3.0

logger = logging.getLogger(__name__)

# Shared session with TCP keepalive to prevent proxy/routing timeout during long image generation.
_shared_session: aiohttp.ClientSession | None = None
_shared_session_loop: asyncio.AbstractEventLoop | None = None
_shared_session_lock: asyncio.Lock | None = None
_shared_session_lock_loop: asyncio.AbstractEventLoop | None = None


class _KeepaliveConnector(aiohttp.TCPConnector):
    """TCPConnector that enables OS-level TCP keepalive on every connection."""

    async def _wrap_create_connection(self, *args, **kwargs):
        transport, protocol = await super()._wrap_create_connection(*args, **kwargs)
        sock = transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
            # macOS uses TCP_KEEPALIVE (0x10), Linux uses TCP_KEEPIDLE
            for option in ("TCP_KEEPIDLE", "TCP_KEEPALIVE"):
                opt_id = getattr(_socket, option, None)
                if opt_id is not None:
                    try:
                        sock.setsockopt(
                            _socket.IPPROTO_TCP, opt_id, _TCP_KEEPALIVE_IDLE_SECONDS
                        )
                    except OSError:
                        pass
                    break
            # Set keepalive interval and count on platforms that support them
            if hasattr(_socket, "TCP_KEEPINTVL"):
                try:
                    sock.setsockopt(
                        _socket.IPPROTO_TCP,
                        _socket.TCP_KEEPINTVL,
                        _TCP_KEEPALIVE_INTERVAL_SECONDS,
                    )
                except OSError:
                    pass
            if hasattr(_socket, "TCP_KEEPCNT"):
                try:
                    sock.setsockopt(
                        _socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, _TCP_KEEPALIVE_COUNT
                    )
                except OSError:
                    pass
        return transport, protocol


async def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session, _shared_session_loop, _shared_session_lock, _shared_session_lock_loop
    loop = asyncio.get_running_loop()
    if _shared_session_lock is None or _shared_session_lock_loop is not loop:
        _shared_session_lock = asyncio.Lock()
        _shared_session_lock_loop = loop

    stale_session = (
        _shared_session is not None
        and not _shared_session.closed
        and _shared_session_loop is not loop
    )
    if stale_session:
        logger.info("Recreating image API session for the current event loop")
        await _shared_session.close()
        _shared_session = None
        _shared_session_loop = None

    if _shared_session is None or _shared_session.closed:
        async with _shared_session_lock:
            if _shared_session is None or _shared_session.closed:
                connector = _KeepaliveConnector(
                    force_close=False,
                    limit=4,
                    ttl_dns_cache=300,
                )
                _shared_session = aiohttp.ClientSession(connector=connector)
                _shared_session_loop = loop
    return _shared_session


async def close_shared_session() -> None:
    """Close the process-wide image client exactly once during app shutdown."""
    global _shared_session, _shared_session_loop
    if _shared_session is not None and not _shared_session.closed:
        await _shared_session.close()
    _shared_session = None
    _shared_session_loop = None


_SUPPORTED_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
}
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


class _ImageAPIError(Exception):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _image_api_http_error(
    response: aiohttp.ClientResponse, body: str
) -> _ImageAPIError:
    """Create a support-actionable error for an Images API HTTP response."""
    request_id = response.headers.get("x-oneapi-request-id")
    message = f"API 返回 HTTP {response.status}, {body[:200]}"
    if request_id:
        message += f" (4Router request_id={request_id})"
    return _ImageAPIError(
        message,
        retryable=response.status in _RETRYABLE_HTTP_STATUSES or response.status >= 500,
    )


def _image_timeout_seconds() -> float:
    raw_value = os.getenv("IMAGE_GEN_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS))
    try:
        timeout = float(raw_value)
        if timeout <= 0:
            raise ValueError
        return timeout
    except ValueError:
        logger.warning(
            "Invalid IMAGE_GEN_TIMEOUT_SECONDS=%r; using %.0fs",
            raw_value,
            _DEFAULT_TIMEOUT_SECONDS,
        )
        return _DEFAULT_TIMEOUT_SECONDS


def _image_max_api_attempts() -> int:
    raw_value = os.getenv("IMAGE_GEN_MAX_ATTEMPTS", str(_DEFAULT_MAX_API_ATTEMPTS))
    try:
        attempts = int(raw_value)
        if not 1 <= attempts <= _MAX_CONFIGURED_API_ATTEMPTS:
            raise ValueError
        return attempts
    except ValueError:
        logger.warning(
            "Invalid IMAGE_GEN_MAX_ATTEMPTS=%r; using %d (valid range: 1-%d)",
            raw_value,
            _DEFAULT_MAX_API_ATTEMPTS,
            _MAX_CONFIGURED_API_ATTEMPTS,
        )
        return _DEFAULT_MAX_API_ATTEMPTS


def _image_retry_base_delay_seconds() -> float:
    raw_value = os.getenv(
        "IMAGE_GEN_RETRY_BASE_DELAY_SECONDS",
        str(_DEFAULT_RETRY_BASE_DELAY_SECONDS),
    )
    try:
        delay = float(raw_value)
        if delay < 0:
            raise ValueError
        return delay
    except ValueError:
        logger.warning(
            "Invalid IMAGE_GEN_RETRY_BASE_DELAY_SECONDS=%r; using %.0fs",
            raw_value,
            _DEFAULT_RETRY_BASE_DELAY_SECONDS,
        )
        return _DEFAULT_RETRY_BASE_DELAY_SECONDS


def _image_request_timeout(timeout_seconds: float) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(
        total=None,
        connect=30.0,
        sock_read=timeout_seconds,
    )


def _image_api_endpoint(base_url: str, image_operation: str) -> str:
    if image_operation not in {"generations", "edits"}:
        raise ValueError(f"Unsupported image operation: {image_operation}")
    return f"{base_url.strip().rstrip('/')}/images/{image_operation}"


def _aligned_native_edge(value: float, *, direction: str = "nearest") -> int:
    units = value / _NATIVE_SIZE_ALIGNMENT
    if direction == "up":
        units = math.ceil(units)
    elif direction == "down":
        units = math.floor(units)
    else:
        units = round(units)
    return max(_NATIVE_SIZE_ALIGNMENT, int(units) * _NATIVE_SIZE_ALIGNMENT)


def _openai_image_size(aspect_ratio: str) -> str:
    short_edge = 1024
    width_ratio, height_ratio = (int(value) for value in aspect_ratio.split(":", 1))
    requested_ratio = width_ratio / height_ratio
    output_ratio = min(
        max(requested_ratio, 1 / _NATIVE_MAX_ASPECT_RATIO),
        _NATIVE_MAX_ASPECT_RATIO,
    )
    if requested_ratio != output_ratio:
        logger.warning(
            "OpenAI Images endpoint does not support aspect ratio %s; clamping to %.0f:1 limit",
            aspect_ratio,
            _NATIVE_MAX_ASPECT_RATIO,
        )

    if output_ratio >= 1:
        width = _aligned_native_edge(short_edge * output_ratio)
        height = _aligned_native_edge(short_edge)
    else:
        width = _aligned_native_edge(short_edge)
        height = _aligned_native_edge(short_edge / output_ratio)

    pixel_count = width * height
    if pixel_count < _NATIVE_MIN_PIXELS:
        scale = math.sqrt(_NATIVE_MIN_PIXELS / pixel_count)
        width = _aligned_native_edge(width * scale, direction="up")
        height = _aligned_native_edge(height * scale, direction="up")

    pixel_count = width * height
    if max(width, height) > _NATIVE_MAX_EDGE or pixel_count > _NATIVE_MAX_PIXELS:
        scale = min(
            _NATIVE_MAX_EDGE / max(width, height),
            math.sqrt(_NATIVE_MAX_PIXELS / pixel_count),
        )
        width = _aligned_native_edge(width * scale, direction="down")
        height = _aligned_native_edge(height * scale, direction="down")

    if width > height * _NATIVE_MAX_ASPECT_RATIO:
        width = int(height * _NATIVE_MAX_ASPECT_RATIO)
    elif height > width * _NATIVE_MAX_ASPECT_RATIO:
        height = int(width * _NATIVE_MAX_ASPECT_RATIO)
    return f"{width}x{height}"


async def _read_images_response(response: aiohttp.ClientResponse) -> bytes:
    try:
        payload = await response.json(content_type=None)
    except (aiohttp.ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _ImageAPIError(f"API 响应不是有效 JSON: {exc}") from exc

    try:
        image = payload["data"][0]
    except (KeyError, IndexError, TypeError):
        raise _ImageAPIError("API 响应未包含 data[0].") from None
    if not isinstance(image, dict):
        raise _ImageAPIError("API 响应 data[0] 格式异常.")

    b64_json = image.get("b64_json")
    if isinstance(b64_json, str) and b64_json:
        try:
            return base64.b64decode(b64_json, validate=True)
        except Exception as exc:
            raise _ImageAPIError(f"API 响应 b64_json 不是有效 base64: {exc}") from exc

    image_url = image.get("url")
    if isinstance(image_url, str) and image_url:
        try:
            return await _generated_image_bytes(image_url)
        except Exception as exc:
            raise _ImageAPIError(f"下载图片异常 {exc}") from exc

    raise _ImageAPIError("API 响应未包含 data[0].b64_json 或 data[0].url.")


async def _request_image_bytes(
    api_url: str,
    api_key: str,
    payload_bytes: bytes,
    *,
    timeout_seconds: float,
) -> bytes:
    timeout = _image_request_timeout(timeout_seconds)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request_started_at = time.monotonic()
    session = await _get_shared_session()
    async with session.post(
        api_url, headers=headers, data=payload_bytes, timeout=timeout
    ) as response:
        if response.status != 200:
            raise _image_api_http_error(response, await response.text())

        image_bytes = await _read_images_response(response)
        logger.info(
            "Image API response received after %.2fs",
            time.monotonic() - request_started_at,
        )
        return image_bytes


async def _request_image_bytes_multipart(
    api_url: str,
    api_key: str,
    form_data: aiohttp.FormData,
    *,
    timeout_seconds: float,
) -> bytes:
    timeout = _image_request_timeout(timeout_seconds)
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    request_started_at = time.monotonic()
    session = await _get_shared_session()
    async with session.post(
        api_url, headers=headers, data=form_data, timeout=timeout
    ) as response:
        if response.status != 200:
            raise _image_api_http_error(response, await response.text())

        image_bytes = await _read_images_response(response)
        logger.info(
            "Image API multipart response received after %.2fs",
            time.monotonic() - request_started_at,
        )
        return image_bytes


@cached_media_attempt
async def _download_and_encode(
    session: aiohttp.ClientSession | None,
    url: str,
    *,
    shuiyuan_model=None,
    max_bytes: int = _MAX_REFERENCE_BYTES,
    strict_remote: bool = False,
    _redirects_remaining: int = 3,
    raise_errors: bool = False,
) -> str | None:
    """下载图片并转为 base64 data URL，整合了水源认证下载。"""
    cached_bytes = remembered_media(url)
    if cached_bytes is not None:
        return _encode_bytes(cached_bytes, url, max_bytes)
    if url.startswith("data:"):
        match = _DATA_URL_RE.match(url)
        if match:
            mime = match.group("mime").lower()
            if strict_remote and mime not in {
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/webp",
                "image/gif",
            }:
                logger.warning("Blocked unsupported data URL MIME: %s", mime)
                return None
            try:
                image_bytes = base64.b64decode(match.group("data"), validate=True)
            except Exception:
                logger.warning("Failed to decode data URL, skipping")
                return None
            if len(image_bytes) > max_bytes:
                logger.warning("Data URL reference image exceeds the size limit")
                return None
            # 保留原始 MIME 类型的扩展名
            ext = ".png" if "png" in mime else ".jpg"
            return _encode_bytes(image_bytes, f"data_url{ext}", max_bytes)
        return None

    # Shuiyuan image URLs need the authenticated forum session.  In particular,
    # proxy/TUN DNS resolvers commonly map the public Shuiyuan host to a reserved
    # synthetic address (for example 198.18.0.0/15), which the generic SSRF guard
    # must reject.  Normalize only the explicitly supported Shuiyuan image paths
    # and route them through the already host-restricted forum downloader instead.
    shuiyuan_image_url = normalize_shuiyuan_image_url(url)
    if shuiyuan_image_url is not None:
        if shuiyuan_model is not None:
            try:
                if shuiyuan_image_url.startswith("upload://"):
                    image_bytes = await shuiyuan_model.download_image(
                        shuiyuan_image_url
                    )
                else:
                    image_bytes = await shuiyuan_model.download_raw_image(
                        shuiyuan_image_url
                    )
            except Exception as exc:
                if raise_errors:
                    raise
                logger.warning(
                    "Shuiyuan reference image download failed for %s: %s",
                    shuiyuan_image_url,
                    exc,
                )
                return None
        else:
            logger.warning(
                "No ShuiyuanModel available, cannot download Shuiyuan image: %s",
                shuiyuan_image_url,
            )
            return None
        return _encode_bytes(image_bytes, url, max_bytes)

    if not url.startswith(("http://", "https://")):
        if strict_remote:
            logger.warning("Blocked local reference image path")
            return None
        try:
            with open(url, "rb") as file:
                image_bytes = file.read()
        except Exception as exc:
            logger.warning("Read local reference image failed: %s %s", url, exc)
            return None
        return _encode_bytes(image_bytes, url, max_bytes)

    try:
        if session is None:
            async with aiohttp.ClientSession() as owned_session:
                return await _download_and_encode(
                    owned_session,
                    url,
                    shuiyuan_model=shuiyuan_model,
                    max_bytes=max_bytes,
                    strict_remote=strict_remote,
                    _redirects_remaining=_redirects_remaining,
                    raise_errors=raise_errors,
                )
        if strict_remote and not await _is_public_http_url(url):
            logger.warning("Blocked non-public reference image URL: %s", url[:80])
            return None
        async with session.get(
            encoded_image_url(url),
            timeout=aiohttp.ClientTimeout(total=30),
            allow_redirects=not strict_remote,
        ) as response:
            if strict_remote and 300 <= response.status < 400:
                if _redirects_remaining <= 0:
                    logger.warning("Reference image exceeded redirect limit")
                    return None
                location = response.headers.get("Location")
                redirected = urljoin(url, location) if location else ""
                if not redirected or not await _is_public_http_url(redirected):
                    logger.warning("Blocked unsafe reference image redirect")
                    return None
                return await _download_and_encode(
                    session,
                    redirected,
                    shuiyuan_model=shuiyuan_model,
                    max_bytes=max_bytes,
                    strict_remote=True,
                    _redirects_remaining=_redirects_remaining - 1,
                    raise_errors=raise_errors,
                )
            if response.status != 200:
                if raise_errors:
                    raise ImageDownloadError(
                        response.status, response.headers.get("Retry-After")
                    )
                logger.warning(
                    "Download reference image failed: %s HTTP %s",
                    url[:80],
                    response.status,
                )
                return None
            if strict_remote:
                content_type = (
                    response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                )
                if not content_type.startswith("image/"):
                    logger.warning(
                        "Reference URL did not return an image MIME type: %s",
                        content_type,
                    )
                    return None
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                logger.warning(
                    "Reference image too large: %s bytes, skipping", content_length
                )
                return None
            image_bytes = await response.read()
            if len(image_bytes) > max_bytes:
                logger.warning("Reference image exceeded size limit after download")
                return None
    except Exception as exc:
        if raise_errors:
            raise
        logger.warning("Download reference image error: %s %s", url[:80], exc)
        return None

    return _encode_bytes(image_bytes, url, max_bytes)


async def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=_socket.SOCK_STREAM,
        )
        if not addresses:
            return False
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                return False
        return True
    except (OSError, ValueError):
        return False


def _compress_reference_image(image_bytes: bytes) -> bytes:
    """压缩参考图片以加速服务端处理，目标是保持在 Cloudflare 60s 超时内完成。

    对超过 _MAX_REFERENCE_LONG_EDGE 的图片进行缩放，并转为 JPEG。
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            original_size = len(image_bytes)
            original_mode = img.mode
            width, height = img.size
            long_edge = max(width, height)

            if long_edge <= _MAX_REFERENCE_LONG_EDGE and original_size <= 200 * 1024:
                return image_bytes  # 已经够小，无需压缩

            # 缩放长边到 _MAX_REFERENCE_LONG_EDGE
            if long_edge > _MAX_REFERENCE_LONG_EDGE:
                ratio = _MAX_REFERENCE_LONG_EDGE / long_edge
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            # 转为 RGB（JPEG 不支持 RGBA/P）
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=_REFERENCE_JPEG_QUALITY)
            compressed = buffer.getvalue()

            logger.info(
                "Compressed reference: %dx%d → %dx%d, %d→%d bytes (%.0f%%), "
                "mode %s→RGB",
                width,
                height,
                img.size[0],
                img.size[1],
                original_size,
                len(compressed),
                len(compressed) / original_size * 100 if original_size else 0,
                original_mode,
            )
            return compressed
    except Exception as exc:
        logger.warning("Reference image compression failed, using original: %s", exc)
        return image_bytes


def _image_mime_from_bytes(image_bytes: bytes, fallback: str = "image/jpeg") -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return fallback


def _encode_bytes(image_bytes: bytes, source_hint: str, max_bytes: int) -> str | None:
    """将图片字节压缩并编码为 base64 data URL"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except Exception:
        logger.warning("Reference payload is not a decodable image")
        return None
    # 预压缩以减小服务端处理时间
    image_bytes = _compress_reference_image(image_bytes)

    if len(image_bytes) > max_bytes:
        logger.warning(
            "Reference image exceeds max size: %d > %d, skipping",
            len(image_bytes),
            max_bytes,
        )
        return None

    ext = os.path.splitext(source_hint.split("?")[0])[1].lower()
    fallback_mime = "image/png" if ext == ".png" else "image/jpeg"
    mime = _image_mime_from_bytes(image_bytes, fallback=fallback_mime)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    logger.info(
        "Encoded reference image: %s (%d bytes)", source_hint[:80], len(image_bytes)
    )
    return f"data:{mime};base64,{encoded}"


def _decode_data_url(data_url: str) -> tuple[bytes, str, str]:
    match = _DATA_URL_RE.match(data_url)
    if not match:
        raise ValueError("无法解析 data URL")
    mime_type = match.group("mime")
    image_bytes = base64.b64decode(match.group("data"), validate=True)
    extension = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime_type.lower(), ".png")
    return image_bytes, mime_type, extension


async def _generated_image_bytes(image_url: str) -> bytes:
    if image_url.startswith("data:"):
        image_bytes, _, _ = _decode_data_url(image_url)
        return image_bytes

    async with aiohttp.ClientSession() as session:
        async with session.get(
            image_url,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                raise ValueError(f"下载图片 HTTP {response.status}")
            return await response.read()


def _prepare_image_upload(image_bytes: bytes) -> tuple[str, bytes]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image_format = (image.format or "PNG").upper()
        extension = {
            "JPEG": ".jpg",
            "JPG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "GIF": ".gif",
        }.get(image_format, f".{image_format.lower()}")
        converted = image.convert("RGB") if image.mode != "RGB" else image
        buffer = io.BytesIO()
        converted.save(buffer, format="JPEG", quality=95)
    return extension, buffer.getvalue()


async def resolve_image_endpoint(state_store=None) -> tuple[str, str, str]:
    """(api_url, api_key, model) for image generation.

    The active stored configuration wins per field; anything it leaves out falls
    back to the deployment environment, so a partially filled entry still works.
    """
    stored: dict = {}
    resolver = getattr(state_store, "model_config_resolver", None)
    if resolver is not None:
        try:
            stored = await resolver() or {}
        except Exception:
            logger.exception("Reading the stored image configuration failed")
    api_key = (stored.get("api_key") or "").strip() or os.getenv(
        "IMAGE_GEN_API_KEY", ""
    ).strip()
    api_url = (stored.get("base_url") or "").strip() or os.getenv(
        "IMAGE_GEN_API_URL", ""
    ).strip()
    image_model = (
        (stored.get("model") or "").strip()
        or os.getenv("IMAGE_GEN_MODEL", "").strip()
        or _DEFAULT_IMAGE_MODEL
    )
    return api_url, api_key, image_model


def _generation_gate() -> asyncio.Semaphore:
    """Per-loop semaphore sized by ``[common.runtime].image_concurrency``."""
    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

    loop = asyncio.get_running_loop()
    gate = _generation_gates.get(loop)
    if gate is None:
        limit = int(get_deployment().section("runtime").get("image_concurrency", 2))
        gate = asyncio.Semaphore(max(1, limit))
        _generation_gates[loop] = gate
    return gate


_generation_gates: (
    "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]"
) = weakref.WeakKeyDictionary()


def _result(status: str, **fields) -> str:
    return json.dumps(
        {
            "status": status,
            **{k: v for k, v in fields.items() if v not in (None, "", [])},
        },
        ensure_ascii=False,
    )


def _failure(code: str, message: str, *, hint: str | None = None) -> str:
    return _result("error", code=code, message=message, hint=hint, retryable=False)


async def _submit_image_request(
    *,
    request_url: str,
    api_key: str,
    image_model: str,
    prompt: str,
    size: str,
    reference_data_urls: list[str],
    request_id: str,
) -> bytes | str:
    """Call the Images API with retries; returns bytes or a failure JSON string."""
    timeout_seconds = _image_timeout_seconds()
    max_api_attempts = _image_max_api_attempts()
    retry_base_delay_seconds = _image_retry_base_delay_seconds()
    edit_images: list[tuple[bytes, str, str]] = []
    for index, data_url in enumerate(reference_data_urls):
        reference_bytes, mime_type, extension = _decode_data_url(data_url)
        edit_images.append(
            (reference_bytes, mime_type, f"reference_{index}{extension}")
        )
    request_body = json.dumps(
        {"model": image_model, "prompt": prompt, "size": size},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    logger.info(
        "Submitting image generation: request_id=%s model=%s endpoint=%s "
        "reference_images=%d timeout=%.0fs max_attempts=%d prompt_preview=%r",
        request_id,
        image_model,
        "edits" if edit_images else "generations",
        len(edit_images),
        timeout_seconds,
        max_api_attempts,
        prompt[:200],
    )
    last_error = ""
    for attempt in range(max_api_attempts):
        if attempt:
            wait_seconds = retry_base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Retrying image API call (attempt %d/%d) in %.1fs request_id=%s; "
                "the upstream may still bill or complete the prior request",
                attempt + 1,
                max_api_attempts,
                wait_seconds,
                request_id,
            )
            await asyncio.sleep(wait_seconds)
        started_at = time.monotonic()
        try:
            if edit_images:
                form = aiohttp.FormData()
                form.add_field("model", image_model)
                form.add_field("prompt", prompt)
                form.add_field("size", size)
                for reference_bytes, mime_type, filename in edit_images:
                    form.add_field(
                        "image[]",
                        reference_bytes,
                        filename=filename,
                        content_type=mime_type,
                    )
                image_bytes = await _request_image_bytes_multipart(
                    request_url, api_key, form, timeout_seconds=timeout_seconds
                )
            else:
                image_bytes = await _request_image_bytes(
                    request_url, api_key, request_body, timeout_seconds=timeout_seconds
                )
            logger.info(
                "Image API request completed: request_id=%s attempt=%d duration=%.2fs image_bytes=%d",
                request_id,
                attempt + 1,
                time.monotonic() - started_at,
                len(image_bytes),
            )
            return image_bytes
        except _ImageAPIError as exc:
            last_error = str(exc)
            if exc.retryable and attempt < max_api_attempts - 1:
                logger.warning(
                    "Image API retryable response (attempt %d/%d): %s",
                    attempt + 1,
                    max_api_attempts,
                    exc,
                )
                continue
            logger.error("Image API response failed: %s", exc)
            return _failure("api_error", f"图片生成失败: {exc}")
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Image API transient error (attempt %d/%d): %s",
                attempt + 1,
                max_api_attempts,
                last_error,
            )
            if attempt < max_api_attempts - 1:
                continue
            if isinstance(exc, aiohttp.ServerDisconnectedError):
                return _failure(
                    "connection_lost",
                    f"图片生成失败: API 连接异常（{max_api_attempts} 次尝试均失败），"
                    "服务端可能仍在后台完成并产生图片，但当前客户端连接已经断开，"
                    "无法接收该次 response；未提供断线续取能力，"
                    f"已执行的重试均为独立请求。最后错误: {last_error}",
                )
            return _failure(
                "connection_error",
                f"图片生成失败: API 连接异常（{max_api_attempts} 次尝试均失败）"
                f"，最后错误: {last_error}",
            )
    return _failure("api_error", f"图片生成失败: API 调用异常 {last_error}")


class ImageGenerationService:
    """Channel-neutral image generation that stores the result as a local Artifact.

    The tool result is JSON so the model can branch on ``status`` without
    parsing prose; the artifact is returned separately for the graph to attach
    to the reply and, on the Responses API, to show back to the model.
    """

    def __init__(self, forum_model, state_store) -> None:
        if state_store is None:
            raise ValueError("ImageGenerationService requires the local state store")
        self.forum_model = forum_model
        self.state_store = state_store

    async def generate(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        references: list[dict[str, str]] | None = None,
        allow_partial: bool = False,
    ) -> tuple[str, GeneratedImageArtifact | None]:
        """根据文字描述生成一张图片，可附带带标签的参考图。

        何时用：用户要求生成、绘制、创作或修改图片时；这是唯一能产生图片的途径。
        参数要点：prompt 用中文详细描述画面（无参考图时写清外貌、服饰、姿态、光影、
        背景、氛围；有参考图时简述要求并说明"参照参考图"）；references 为
        [{"key","url","label"}]，url 可以是 upload://、水源头像地址或本轮工具返回的
        图片 URL，label 说明该图代表谁或什么；任一参考图读取失败时默认不生成，
        只有用户接受缺项时才传 allow_partial=true。
        返回：{"status":"ok","artifact":"artifact://…","width","height"} —— 必须用
        ![描述](artifact://…) 嵌入最终回复；{"status":"partial","failed":[…]} 表示
        参考图缺失、尚未生成；{"status":"error","code","message"} 表示失败原因。
        """
        prompt = str(prompt).strip()
        if len(prompt) < 10 or prompt.isdigit() or len(set(prompt)) <= 2:
            return (
                _failure(
                    "invalid_prompt",
                    "prompt 过短或无意义，请提供至少 10 个字符的画面描述",
                ),
                None,
            )
        api_url, api_key, image_model = await resolve_image_endpoint(self.state_store)
        if not api_key:
            return _failure("not_configured", "IMAGE_GEN_API_KEY 未配置"), None
        if not api_url:
            return _failure("not_configured", "IMAGE_GEN_API_URL 未配置"), None
        if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
            aspect_ratio = "1:1"

        reference_data_urls: list[str] = []
        if references:
            from .image_references import prepare_references

            prepared = await prepare_references(
                references, model=self.forum_model, strict_remote=True
            )
            good = [item for item in prepared["items"] if item["status"] == "ok"]
            missing = [item for item in prepared["items"] if item["status"] != "ok"]
            if prepared["status"] == "error" or (missing and not allow_partial):
                return (
                    _result(
                        "partial" if good else "error",
                        code="reference_failed",
                        message="部分参考素材读取失败，尚未生成",
                        failed=[
                            {
                                "key": item["key"],
                                "label": item["label"],
                                "error": item.get("error", "unavailable"),
                            }
                            for item in missing
                        ],
                        loaded=[item["key"] for item in good],
                        hint=(
                            "更换失败素材的 url 后重试；若用户接受缺少这些对象，"
                            "传 allow_partial=true 只用成功素材生成"
                        ),
                        retryable=False,
                    ),
                    None,
                )
            reference_data_urls = prepared["data_urls"]
            mapping = "\n".join(
                f"参考图{i + 1}：{item['label']}" for i, item in enumerate(good)
            )
            prompt += "\n\n【实际参考素材对应关系】\n" + mapping
            if missing:
                labels = "、".join(item["label"] for item in missing)
                prompt += (
                    f"\n仅使用上述 {len(good)} 项成功素材；未提供的素材（{labels}）"
                    "及其对应对象不纳入生成，不猜测其形象。"
                )

        request_id = uuid.uuid4().hex[:12]
        operation = "edits" if reference_data_urls else "generations"
        queued_at = time.monotonic()
        async with _generation_gate():
            logger.info(
                "Image generation slot acquired after %.2fs, request_id=%s",
                time.monotonic() - queued_at,
                request_id,
            )
            outcome = await _submit_image_request(
                request_url=_image_api_endpoint(api_url, operation),
                api_key=api_key,
                image_model=image_model,
                prompt=prompt,
                size=_openai_image_size(aspect_ratio),
                reference_data_urls=reference_data_urls,
                request_id=request_id,
            )
        if isinstance(outcome, str):
            return outcome, None
        image_bytes = outcome

        try:
            with Image.open(io.BytesIO(image_bytes)) as generated:
                generated.verify()
            with Image.open(io.BytesIO(image_bytes)) as generated:
                width, height = generated.size
                image_format = (generated.format or "PNG").upper()
                mime_type = Image.MIME.get(image_format, "image/png")
                extension = {
                    "JPEG": ".jpg",
                    "PNG": ".png",
                    "WEBP": ".webp",
                    "GIF": ".gif",
                }.get(image_format, ".png")
        except Exception as exc:
            logger.error("Generated image is not decodable: %s", exc)
            return _failure("invalid_image", f"生成结果不是有效图片: {exc}"), None

        artifact_id = str(uuid.uuid4())
        output_dir = state_directory() / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{artifact_id}{extension}"
        try:
            path.write_bytes(image_bytes)
            await self.state_store.register_artifact(
                artifact_id=artifact_id,
                local_path=str(path),
                mime_type=mime_type,
                byte_count=len(image_bytes),
                width=width,
                height=height,
                filename=f"generated{extension}",
            )
        except Exception as exc:
            path.unlink(missing_ok=True)
            logger.exception("Failed to register generated image artifact")
            return _failure("storage_error", f"保存本地 Artifact 失败: {exc}"), None
        artifact = GeneratedImageArtifact(
            artifact_id=artifact_id,
            mime_type=mime_type,
            local_path=str(path),
            byte_count=len(image_bytes),
            width=width,
            height=height,
        )
        logger.info(
            "Generated image stored: request_id=%s artifact=%s bytes=%d size=%dx%d",
            request_id,
            artifact_id,
            len(image_bytes),
            width,
            height,
        )
        return (
            _result(
                "ok",
                artifact=artifact.uri,
                width=width,
                height=height,
                note="在最终回复中用 ![描述](" + artifact.uri + ") 展示这张图",
            ),
            artifact,
        )

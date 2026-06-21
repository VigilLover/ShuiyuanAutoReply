import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import socket as _socket
import time
from datetime import datetime

import aiohttp
from PIL import Image

from shuiyuan_auto_reply.constants import settings

# Single and combined reference-image limits. Base64 increases the wire size.
_MAX_REFERENCE_BYTES = 10 * 1024 * 1024
_MAX_TOTAL_REFERENCE_BYTES = 20 * 1024 * 1024
# Reference image compression: resize to at most this many pixels on the longest edge.
# Reduces server-side processing time to stay under the ~60s Cloudflare/Caddy proxy timeout.
_MAX_REFERENCE_LONG_EDGE = 1024
_REFERENCE_JPEG_QUALITY = 80
_DEFAULT_IMAGE_MODEL = "gpt-image-2"
_FIXED_IMAGE_SIZE = "1K"
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
                        sock.setsockopt(_socket.IPPROTO_TCP, opt_id, _TCP_KEEPALIVE_IDLE_SECONDS)
                    except OSError:
                        pass
                    break
            # Set keepalive interval and count on platforms that support them
            if hasattr(_socket, "TCP_KEEPINTVL"):
                try:
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, _TCP_KEEPALIVE_INTERVAL_SECONDS)
                except OSError:
                    pass
            if hasattr(_socket, "TCP_KEEPCNT"):
                try:
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, _TCP_KEEPALIVE_COUNT)
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

_SUPPORTED_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
}
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


class _ImageAPIError(Exception):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


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
    async with session.post(api_url, headers=headers, data=payload_bytes, timeout=timeout) as response:
        if response.status != 200:
            body = (await response.text())[:200]
            message = f"API 返回 HTTP {response.status}, {body}"
            retryable = (
                response.status in _RETRYABLE_HTTP_STATUSES
                or response.status >= 500
            )
            raise _ImageAPIError(message, retryable=retryable)

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
    async with session.post(api_url, headers=headers, data=form_data, timeout=timeout) as response:
        if response.status != 200:
            body = (await response.text())[:200]
            message = f"API 返回 HTTP {response.status}, {body}"
            retryable = (
                response.status in _RETRYABLE_HTTP_STATUSES
                or response.status >= 500
            )
            raise _ImageAPIError(message, retryable=retryable)

        image_bytes = await _read_images_response(response)
        logger.info(
            "Image API multipart response received after %.2fs",
            time.monotonic() - request_started_at,
        )
        return image_bytes


async def _download_and_encode(
    session: aiohttp.ClientSession | None,
    url: str,
    *,
    shuiyuan_model=None,
    max_bytes: int = _MAX_REFERENCE_BYTES,
) -> str | None:
    """下载图片并转为 base64 data URL，整合了水源认证下载。"""
    if url.startswith("data:"):
        match = _DATA_URL_RE.match(url)
        if match:
            try:
                image_bytes = base64.b64decode(match.group("data"), validate=True)
            except Exception:
                logger.warning("Failed to decode data URL, skipping")
                return None
            mime = match.group("mime")
            # 保留原始 MIME 类型的扩展名
            ext = ".png" if "png" in mime else ".jpg"
            return _encode_bytes(image_bytes, f"data_url{ext}", max_bytes)
        return None

    is_upload = url.startswith("upload://")
    is_short_path = url.startswith("/uploads/short-url/")
    if is_upload or is_short_path:
        upload_url = url if is_upload else url.replace("/uploads/short-url/", "upload://")
        if shuiyuan_model is not None:
            try:
                image_bytes = await shuiyuan_model.download_image(upload_url)
            except Exception as exc:
                logger.warning("Shuiyuan download_image failed for %s: %s", upload_url, exc)
                return None
        else:
            logger.warning("No ShuiyuanModel available, cannot download upload:// image: %s", upload_url)
            return None
        return _encode_bytes(image_bytes, url, max_bytes)

    if not url.startswith(("http://", "https://")):
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
                )
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                logger.warning("Download reference image failed: %s HTTP %s", url[:80], response.status)
                return None
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                logger.warning("Reference image too large: %s bytes, skipping", content_length)
                return None
            image_bytes = await response.read()
    except Exception as exc:
        logger.warning("Download reference image error: %s %s", url[:80], exc)
        return None

    return _encode_bytes(image_bytes, url, max_bytes)


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
                width, height,
                img.size[0], img.size[1],
                original_size, len(compressed),
                len(compressed) / original_size * 100 if original_size else 0,
                original_mode,
            )
            return compressed
    except Exception as exc:
        logger.warning("Reference image compression failed, using original: %s", exc)
        return image_bytes


def _encode_bytes(image_bytes: bytes, source_hint: str, max_bytes: int) -> str | None:
    """将图片字节压缩并编码为 base64 data URL"""
    # 预压缩以减小服务端处理时间
    image_bytes = _compress_reference_image(image_bytes)

    if len(image_bytes) > max_bytes:
        logger.warning("Reference image exceeds max size: %d > %d, skipping", len(image_bytes), max_bytes)
        return None

    # 统一用 JPEG MIME（压缩后都是 JPEG），除非仍为 PNG
    ext = os.path.splitext(source_hint.split("?")[0])[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    logger.info("Encoded reference image: %s (%d bytes)", source_hint[:80], len(image_bytes))
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


def create_image_generation_tool(model):
    """
    创建一个与 ShuiyuanModel 绑定的文生图工具函数.

    :param model: ShuiyuanModel 实例, 用于调用 upload_image 上传图片到水源.
    :return: async callable, 可作为 StructuredTool 的 coroutine.
    """
    generation_gate = asyncio.Semaphore(1)

    async def generate_image(
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        reference_images: str | list[str] | None = None,
        output_dir: str | None = None,
    ) -> str:
        """
        根据用户的文字描述生成图片，自动上传到水源并返回图片的短链接。

        这是生成图片的唯一方式。如果你没有调用此工具，你没有任何图片可以展示。
        绝对禁止在没有调用本工具的情况下编造或输出任何图片链接。

        【重要】你必须使用 Markdown 图片语法将返回的短链接嵌入最终回复：`![描述](短链接)`
        例如返回 `upload://zuyICpNdsQZCsV4cWeOwgcDLLak.jpeg`，你在回复中写 `![生成的图片](upload://zuyICpNdsQZCsV4cWeOwgcDLLak.jpeg)`

        提示词(prompt)编写规则（根据是否有参考图区别对待）：
        - 有参考图（reference_images 非空）：prompt 只需用纯中文简要描述原本要求，不要自行添加任何风格词或细节描写，让参考图主导视觉，并且强调"根据给定的参考图生成图片"。
        - 需要参考水源用户头像时，先通过 search_user 或 search_user_by_id 获取 avatar，再把 avatar URL 传入 reference_images。
        - 无参考图（reference_images 为空）：必须用纯中文进行极其详细的画面描述，涵盖外貌、服饰、姿态、光影、背景、氛围等。如果绘画对象是人物，画风默认二次元精美插画，强调"唯美、精细、干净通透"，避免过度锐化、畸变与崩坏。若用户提供设定/附件/印象，必须将关键元素具象化融入画面。

        :param prompt: 详细的纯中文生图提示词。
        :param aspect_ratio: 画面宽高比，默认 1:1。支持 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 1:4, 4:1, 1:8, 8:1。
        :param image_size: 保留用于兼容已有工具调用；当前始终使用固定 1K 分辨率，传入其他值会被忽略。
        :param reference_images: 参考图片 URL 列表。传入 Python 列表格式如 ["upload://xxx.jpeg"]，支持 upload://、http(s)://、data: 等格式。
        :param output_dir: 可选的自定义输出目录，用于保存生成的图片备份。
        :return: 图片的短链接。你必须用 `![描述](链接)` 格式嵌入回复中。
        """
        api_key = os.getenv("IMAGE_GEN_API_KEY", "").strip()
        api_url = os.getenv("IMAGE_GEN_API_URL", "").strip()
        image_model = os.getenv("IMAGE_GEN_MODEL", "").strip() or _DEFAULT_IMAGE_MODEL
        if not api_key:
            return "图片生成失败: IMAGE_GEN_API_KEY 未配置."
        if not api_url:
            return "图片生成失败: IMAGE_GEN_API_URL 未配置."

        if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
            aspect_ratio = "1:1"
        if image_size != _FIXED_IMAGE_SIZE:
            logger.info(
                "Ignoring requested image_size=%s; using fixed image_size=%s",
                image_size,
                _FIXED_IMAGE_SIZE,
            )
        image_size = _FIXED_IMAGE_SIZE

        if reference_images is None:
            pass
        elif isinstance(reference_images, list):
            reference_images = [item for item in reference_images if isinstance(item, str) and item]
        elif isinstance(reference_images, str):
            text = reference_images.strip()
            if text.startswith("["):
                try:
                    parsed_references = json.loads(text)
                    reference_images = (
                        [item for item in parsed_references if isinstance(item, str) and item]
                        if isinstance(parsed_references, list)
                        else [text]
                    )
                except (json.JSONDecodeError, TypeError):
                    reference_images = [text] if text else None
            else:
                reference_images = [text] if text else None
        else:
            reference_images = None

        reference_data_urls: list[str] = []
        total_ref_bytes = 0
        use_edit_endpoint = bool(reference_images)
        if reference_images:
            async with aiohttp.ClientSession() as session:
                for url in reference_images:
                    data_url = await _download_and_encode(session, url, shuiyuan_model=model)
                    if not data_url:
                        continue
                    encoded_length = len(data_url.split(",", 1)[1]) if "," in data_url else len(data_url)
                    estimated_bytes = int(encoded_length * 3 / 4)
                    if total_ref_bytes + estimated_bytes > _MAX_TOTAL_REFERENCE_BYTES:
                        logger.warning(
                            "Reference images total size would exceed %dMB, skipping remaining",
                            _MAX_TOTAL_REFERENCE_BYTES // (1024 * 1024),
                        )
                        break
                    total_ref_bytes += estimated_bytes
                    reference_data_urls.append(data_url)
        if use_edit_endpoint and not reference_data_urls:
            return "图片生成失败: 未能读取可用的参考图片."

        image_operation = "edits" if use_edit_endpoint else "generations"
        request_url = _image_api_endpoint(api_url, image_operation)
        timeout_seconds = _image_timeout_seconds()
        max_api_attempts = _image_max_api_attempts()
        retry_base_delay_seconds = _image_retry_base_delay_seconds()
        image_size_value = _openai_image_size(aspect_ratio)
        if use_edit_endpoint:
            edit_images: list[tuple[bytes, str, str]] = []
            for index, data_url in enumerate(reference_data_urls):
                reference_bytes, mime_type, extension = _decode_data_url(data_url)
                edit_images.append((reference_bytes, mime_type, f"reference_{index}{extension}"))
            payload_bytes_len = total_ref_bytes
        else:
            payload = {
                "model": image_model,
                "prompt": prompt,
                "size": image_size_value,
                "n": 1,
            }
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            payload_bytes_len = len(request_body)
        logger.info(
            "Submitting image generation: model=%s endpoint=%s reference_images=%d "
            "reference_bytes=%d request_bytes=%d timeout=%.0fs max_attempts=%d "
            "retry_base_delay=%.1fs",
            image_model,
            image_operation,
            len(reference_data_urls),
            total_ref_bytes,
            payload_bytes_len,
            timeout_seconds,
            max_api_attempts,
            retry_base_delay_seconds,
        )

        queued_at = time.monotonic()
        image_bytes = b""
        last_error = ""
        async with generation_gate:
            logger.info(
                "Image generation request acquired single-flight slot after %.2fs",
                time.monotonic() - queued_at,
            )
            for attempt in range(max_api_attempts):
                if attempt:
                    wait_seconds = retry_base_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "Retrying image API call (attempt %d/%d) in %.1fs after a transient failure; "
                        "the upstream may still bill or complete the prior request",
                        attempt + 1,
                        max_api_attempts,
                        wait_seconds,
                    )
                    await asyncio.sleep(wait_seconds)
                started_at = time.monotonic()
                try:
                    if use_edit_endpoint:
                        request_body = aiohttp.FormData()
                        request_body.add_field("model", image_model)
                        request_body.add_field("prompt", prompt)
                        request_body.add_field("size", image_size_value)
                        request_body.add_field("n", "1")
                        for reference_bytes, mime_type, filename in edit_images:
                            request_body.add_field(
                                "image",
                                reference_bytes,
                                filename=filename,
                                content_type=mime_type,
                            )
                        image_bytes = await _request_image_bytes_multipart(
                            request_url,
                            api_key,
                            request_body,
                            timeout_seconds=timeout_seconds,
                        )
                    else:
                        image_bytes = await _request_image_bytes(
                            request_url,
                            api_key,
                            request_body,
                            timeout_seconds=timeout_seconds,
                        )
                    logger.info(
                        "Image API request completed: attempt=%d duration=%.2fs image_bytes=%d",
                        attempt + 1,
                        time.monotonic() - started_at,
                        len(image_bytes),
                    )
                    break
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
                    return f"图片生成失败: {exc}"
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
                        return (
                            f"图片生成失败: API 连接异常（{max_api_attempts} 次尝试均失败），"
                            "服务端可能仍在后台完成并产生图片，但当前客户端连接已经断开，"
                            "无法接收该次 response；未提供断线续取能力，"
                            "已执行的重试均为独立请求。"
                            f"最后错误: {last_error}"
                        )
                    return (
                        f"图片生成失败: API 连接异常（{max_api_attempts} 次尝试均失败）"
                        f"，最后错误: {last_error}"
                    )
                except Exception as exc:
                    logger.exception("Image API call failed")
                    return f"图片生成失败: API 调用异常 {exc}"
            else:
                return f"图片生成失败: API 调用异常 {last_error}"

        try:
            extension, upload_bytes = _prepare_image_upload(image_bytes)
            logger.info(
                "Got generated image: original_bytes=%d upload_jpeg_bytes=%d",
                len(image_bytes),
                len(upload_bytes),
            )
        except Exception as exc:
            logger.error("Image download/parse failed: %s", exc)
            return f"图片生成失败: 下载图片异常 {exc}"

        try:
            backup_dir = output_dir or os.path.join(settings.assets_directory, "generated_images")
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_prompt = prompt[:20].replace(" ", "_").replace("/", "_")
            backup_path = os.path.join(backup_dir, f"{timestamp}_{safe_prompt}{extension}")
            with open(backup_path, "wb") as file:
                file.write(image_bytes)
            logger.info("Saved backup to: %s", backup_path)
        except Exception as exc:
            logger.warning("Backup save failed (non-fatal): %s", exc)

        try:
            response = await model.upload_image(upload_bytes)
            logger.info("Uploaded to Shuiyuan: %s", response.short_path)
            return response.short_path
        except Exception as exc:
            logger.error("Shuiyuan upload failed: %s", exc)
            return f"图片生成失败: 上传到水源异常 {exc}"

    return generate_image

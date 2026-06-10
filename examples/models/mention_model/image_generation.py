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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


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


def _normalize_image_api_url(api_url: str) -> str:
    normalized = api_url.strip()
    lower_url = normalized.rstrip("/").lower()
    if lower_url.endswith("/v1/chat/completion"):
        normalized = normalized.rstrip("/") + "s"
        logger.warning(
            "IMAGE_GEN_API_URL ended with /v1/chat/completion; using documented "
            "/v1/chat/completions endpoint instead"
        )
    return normalized


def _uses_native_images_endpoint(api_url: str) -> bool:
    return api_url.rstrip("/").lower().endswith("/v1/images/generations")


def _aligned_native_edge(value: float, *, direction: str = "nearest") -> int:
    units = value / _NATIVE_SIZE_ALIGNMENT
    if direction == "up":
        units = math.ceil(units)
    elif direction == "down":
        units = math.floor(units)
    else:
        units = round(units)
    return max(_NATIVE_SIZE_ALIGNMENT, int(units) * _NATIVE_SIZE_ALIGNMENT)


def _native_image_size(aspect_ratio: str) -> str:
    short_edge = 1024
    width_ratio, height_ratio = (int(value) for value in aspect_ratio.split(":", 1))
    requested_ratio = width_ratio / height_ratio
    output_ratio = min(
        max(requested_ratio, 1 / _NATIVE_MAX_ASPECT_RATIO),
        _NATIVE_MAX_ASPECT_RATIO,
    )
    if requested_ratio != output_ratio:
        logger.warning(
            "Native image endpoint does not support aspect ratio %s; clamping to %.0f:1 limit",
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


def _extract_url_from_text(content: str) -> str | None:
    markdown_matches = re.findall(r"!\[.*?\]\(([^)]+)\)", content)
    if markdown_matches:
        return markdown_matches[-1]

    data_matches = re.findall(r"data:image/[a-zA-Z0-9.+-]+;base64,[^\s)]+", content)
    if data_matches:
        return data_matches[-1]

    upload_matches = re.findall(r"upload://[^\s)]+", content)
    if upload_matches:
        return upload_matches[-1]

    http_matches = re.findall(r"https?://[^\s)]+", content)
    return http_matches[-1].rstrip(".,，。") if http_matches else None


def _extract_url_from_json(value: object) -> str | None:
    if isinstance(value, dict):
        # 直接检查常见 key
        for key in ("url", "image_url", "short_path", "short_url"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                image_url = _extract_url_from_text(candidate)
                if image_url:
                    return image_url
        # 检查 image_url 对象格式: {"image_url": {"url": "https://..."}}
        image_url_obj = value.get("image_url")
        if isinstance(image_url_obj, dict):
            url = image_url_obj.get("url")
            if isinstance(url, str):
                image_url = _extract_url_from_text(url)
                if image_url:
                    return image_url
        # 检查 type=image_url 的内容块
        if value.get("type") == "image_url":
            if isinstance(image_url_obj, dict):
                url = image_url_obj.get("url")
                if isinstance(url, str):
                    image_url = _extract_url_from_text(url)
                    if image_url:
                        return image_url
        # 递归搜索子项
        for item in value.values():
            image_url = _extract_url_from_json(item)
            if image_url:
                return image_url
    elif isinstance(value, list):
        for item in reversed(value):
            image_url = _extract_url_from_json(item)
            if image_url:
                return image_url
    elif isinstance(value, str):
        return _extract_url_from_text(value)
    return None


def _extract_image_url(content: str) -> str | None:
    """从完整回复中提取最后一张图片，跳过流中的中间预览图。"""
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            # 优先尝试 images/generations 格式（data[0].url）
            try:
                return _image_generation_url(payload)
            except _ImageAPIError:
                pass
            # 再尝试通用的 JSON 递归提取
            image_url = _extract_url_from_json(payload)
            if image_url:
                return image_url

    image_url = _extract_url_from_text(content)
    if image_url:
        return image_url

    # 所有方法都失败时，记录原始内容便于调试
    logger.warning(
        "Failed to extract image URL from response content (first 500 chars): %.500s",
        content,
    )
    return None


def _text_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                # 保留 text 类型内容
                if isinstance(item.get("text"), str):
                    pieces.append(item["text"])
                # 提取 image_url 中的 URL（OpenAI 兼容的多模态响应格式）
                elif item.get("type") == "image_url":
                    image_url_obj = item.get("image_url")
                    if isinstance(image_url_obj, dict):
                        url = image_url_obj.get("url", "")
                        if isinstance(url, str) and url:
                            pieces.append(url)
        return "".join(pieces)
    return ""


def _completion_content(payload: object) -> str:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise _ImageAPIError("API 响应未包含 choices[0].message.") from None

    if not isinstance(message, dict):
        raise _ImageAPIError("API 响应 choices[0].message 格式异常.")

    content = _text_content(message.get("content"))

    # 回退：某些 API 将生成图片放在 message.images 字段
    if not content:
        images = message.get("images")
        if isinstance(images, list) and images:
            image_url = _extract_url_from_json(images)
            if image_url:
                return image_url

    if not content:
        logger.error(
            "API response content is empty. Raw message keys: %s; "
            "content snippet: %.500s",
            list(message.keys()) if isinstance(message, dict) else type(message),
            str(message.get("content"))[:500] if isinstance(message, dict) else str(message)[:500],
        )
        raise _ImageAPIError("API 响应未包含可用的图片内容.")

    return content


def _image_generation_url(payload: object) -> str:
    try:
        image_url = payload["data"][0]["url"]
    except (KeyError, IndexError, TypeError):
        raise _ImageAPIError("API 响应未包含 data[0].url.") from None
    if not isinstance(image_url, str) or not image_url:
        raise _ImageAPIError("API 响应未包含可用的图片 URL.")
    return image_url


def _stream_content_piece(payload: object) -> str:
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(choice, dict):
        return ""

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = _text_content(delta.get("content"))
        if content:
            return content

    message = choice.get("message")
    if isinstance(message, dict):
        return _text_content(message.get("content"))
    return ""


def _take_sse_event(buffer: bytes) -> tuple[bytes | None, bytes]:
    delimiters = []
    for separator in (b"\n\n", b"\r\n\r\n"):
        index = buffer.find(separator)
        if index >= 0:
            delimiters.append((index, len(separator)))
    if not delimiters:
        return None, buffer
    index, separator_length = min(delimiters)
    return buffer[:index], buffer[index + separator_length:]


def _sse_data(event: bytes) -> bytes | None:
    data_lines = []
    for line in event.replace(b"\r\n", b"\n").split(b"\n"):
        if line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip(b" "))
    if not data_lines:
        return None
    return b"\n".join(data_lines)


def _sse_event_type(event: bytes) -> str:
    for line in event.replace(b"\r\n", b"\n").split(b"\n"):
        if line.startswith(b"event:"):
            return line[6:].strip().decode("utf-8", errors="replace").lower()
    return ""


def _consume_sse_event(event: bytes, pieces: list[str]) -> tuple[bool, bool]:
    raw_data = _sse_data(event)
    if raw_data is None:
        return False, False
    raw_data = raw_data.strip()
    if not raw_data:
        return False, False
    if raw_data == b"[DONE]":
        return True, False

    event_type = _sse_event_type(event)
    try:
        text = raw_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ImageAPIError(f"API SSE 响应包含无效 UTF-8 数据: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if event_type in {"error", "failed", "failure"}:
            raise _ImageAPIError(f"API SSE 返回错误事件: {text[:200]}")
        # Some compatible relays emit human-readable progress or heartbeat data.
        # Keep it out of protocol errors; Markdown image extraction ignores it.
        pieces.append(text)
        return False, True

    if event_type in {"error", "failed", "failure"}:
        raise _ImageAPIError(f"API SSE 返回错误事件: {str(payload)[:200]}")
    if isinstance(payload, dict) and payload.get("error"):
        raise _ImageAPIError(f"API SSE 返回错误: {str(payload['error'])[:200]}")
    content_piece = _stream_content_piece(payload)
    if content_piece:
        pieces.append(content_piece)
    try:
        finish_reason = payload["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        finish_reason = None
    # Right Code documents that the final stream chunk carries usage. Some
    # relays send that chunk without another content delta or [DONE] marker.
    return finish_reason is not None or "usage" in payload, False


async def _read_json_completion(
    response: aiohttp.ClientResponse,
    *,
    native_images_endpoint: bool = False,
) -> str:
    try:
        payload = await response.json(content_type=None)
    except (aiohttp.ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _ImageAPIError(f"API 响应不是有效 JSON: {exc}") from exc
    if native_images_endpoint:
        return _image_generation_url(payload)
    return _completion_content(payload)


async def _read_streaming_completion(
    response: aiohttp.ClientResponse,
    *,
    request_started_at: float,
) -> str:
    pending = b""
    pieces: list[str] = []
    response_is_sse: bool | None = None
    first_bytes_logged = False
    non_json_event_count = 0
    done = False

    async for chunk in response.content.iter_chunked(64 * 1024):
        if not chunk:
            continue
        if not first_bytes_logged:
            logger.info(
                "Image API first response bytes received after %.2fs",
                time.monotonic() - request_started_at,
            )
            first_bytes_logged = True
        pending += chunk

        if response_is_sse is None:
            leading = pending.lstrip()
            if leading.startswith((b"data:", b"event:", b":")):
                response_is_sse = True
                logger.info("Image API response mode: SSE stream")
            elif leading.startswith((b"{", b"[")):
                response_is_sse = False
                logger.warning(
                    "Image API response mode: complete JSON despite stream=true; "
                    "the relay may remain silent while generating"
                )
            elif len(leading) > 32:
                raise _ImageAPIError("API 流式响应既不是 SSE 也不是 JSON.")

        if response_is_sse:
            while True:
                event, pending = _take_sse_event(pending)
                if event is None:
                    break
                done, non_json_event = _consume_sse_event(event, pieces)
                if non_json_event:
                    non_json_event_count += 1
                if done:
                    break
        if done:
            break

    if response_is_sse:
        if pending.strip() and not done:
            done, non_json_event = _consume_sse_event(pending, pieces)
            if non_json_event:
                non_json_event_count += 1
        if non_json_event_count:
            logger.info(
                "Image API retained %d non-JSON SSE progress event(s) while awaiting output",
                non_json_event_count,
            )
        if not done:
            raise _ImageAPIError(
                "API SSE 响应在最终完成标记前结束，已忽略可能的临时图片.",
                retryable=True,
            )
        if not pieces:
            raise _ImageAPIError("API SSE 响应未包含可用的图片内容.")
        return "".join(pieces)

    try:
        payload = json.loads(pending.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ImageAPIError(f"API 响应不是有效 JSON 或 SSE: {exc}") from exc
    return _completion_content(payload)


async def _request_image_content(
    api_url: str,
    api_key: str,
    payload_bytes: bytes,
    *,
    stream: bool,
    timeout_seconds: float,
    native_images_endpoint: bool = False,
) -> str:
    timeout = _image_request_timeout(timeout_seconds)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"

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
            if stream and 400 <= response.status < 500 and not retryable:
                message += (
                    "；当前中转站可能不支持 stream=true，请确认接口能力，"
                    "或临时设置 IMAGE_GEN_STREAM=false 诊断"
                )
            raise _ImageAPIError(message, retryable=retryable)

        if native_images_endpoint:
            return await _read_json_completion(
                response,
                native_images_endpoint=True,
            )
        if stream:
            return await _read_streaming_completion(
                response,
                request_started_at=request_started_at,
            )
        return await _read_json_completion(response)


async def _download_and_encode(
    session: aiohttp.ClientSession | None,
    url: str,
    *,
    shuiyuan_model=None,
    max_bytes: int = _MAX_REFERENCE_BYTES,
) -> str | None:
    """下载图片并转为 base64 data URL，整合了水源认证下载。"""
    if url.startswith("data:"):
        encoded_data = url.split(",", 1)[1] if "," in url else ""
        estimated_bytes = int(len(encoded_data) * 3 / 4)
        if estimated_bytes > max_bytes:
            logger.warning(
                "Reference data URL exceeds max size: %d > %d, skipping",
                estimated_bytes,
                max_bytes,
            )
            return None
        return url

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


def _encode_bytes(image_bytes: bytes, source_hint: str, max_bytes: int) -> str | None:
    """将图片字节编码为 base64 data URL"""
    if len(image_bytes) > max_bytes:
        logger.warning("Reference image exceeds max size: %d > %d, skipping", len(image_bytes), max_bytes)
        return None
    ext = os.path.splitext(source_hint.split("?")[0])[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    logger.info("Encoded reference image: %s (%d bytes)", source_hint[:80], len(image_bytes))
    return f"data:{mime};base64,{encoded}"


async def _generated_image_bytes(image_url: str) -> bytes:
    if image_url.startswith("data:"):
        match = _DATA_URL_RE.match(image_url)
        if not match:
            raise ValueError("无法解析 data URL")
        return base64.b64decode(match.group("data"), validate=True)

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
        api_url = _normalize_image_api_url(os.getenv("IMAGE_GEN_API_URL", ""))
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

        content: list[dict] = [{"type": "text", "text": prompt}]
        reference_data_urls: list[str] = []
        total_ref_bytes = 0
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
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })

        native_images_endpoint = _uses_native_images_endpoint(api_url)
        configured_stream = _env_bool("IMAGE_GEN_STREAM", True)
        # 流式只在 chat completions 端点有效
        stream = configured_stream and not native_images_endpoint
        if native_images_endpoint:
            logger.warning(
                "Using /v1/images/generations endpoint without streaming. "
                "The upstream proxy (Cloudflare/Caddy) enforces a ~60s idle timeout; "
                "long image tasks may be disconnected before completion. "
                "Use /v1/chat/completions with IMAGE_GEN_STREAM=true instead."
            )
        elif not stream:
            logger.warning(
                "IMAGE_GEN_STREAM=false disables SSE streaming. "
                "Without streaming, the connection may be killed by the ~60s proxy "
                "idle timeout during long generations. Enabling streaming is strongly "
                "recommended for reliable image generation."
            )
        timeout_seconds = _image_timeout_seconds()
        max_api_attempts = _image_max_api_attempts()
        retry_base_delay_seconds = _image_retry_base_delay_seconds()
        if native_images_endpoint:
            payload = {
                "model": image_model,
                "prompt": prompt,
                "image": reference_data_urls,
                "size": _native_image_size(aspect_ratio),
                "response_format": "url",
            }
        else:
            payload = {
                "model": image_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4096,
                "image_config": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size,
                },
                "stream": stream,
            }
        payload_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        logger.info(
            "Submitting image generation: model=%s endpoint_mode=%s stream=%s reference_images=%d "
            "reference_bytes=%d request_bytes=%d timeout=%.0fs max_attempts=%d "
            "retry_base_delay=%.1fs",
            image_model,
            "images" if native_images_endpoint else "chat",
            stream,
            len(reference_data_urls),
            total_ref_bytes,
            len(payload_bytes),
            timeout_seconds,
            max_api_attempts,
            retry_base_delay_seconds,
        )

        queued_at = time.monotonic()
        generated_content = ""
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
                    generated_content = await _request_image_content(
                        api_url,
                        api_key,
                        payload_bytes,
                        stream=stream,
                        timeout_seconds=timeout_seconds,
                        native_images_endpoint=native_images_endpoint,
                    )
                    logger.info(
                        "Image API request completed: attempt=%d duration=%.2fs content_chars=%d",
                        attempt + 1,
                        time.monotonic() - started_at,
                        len(generated_content),
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

        image_url = (
            generated_content
            if native_images_endpoint
            else _extract_image_url(generated_content)
        )
        # 回退：非 native 端点也可能返回 data[0].url 格式，尝试作为 JSON 解析
        if not image_url and not native_images_endpoint:
            try:
                payload = json.loads(generated_content)
                image_url = _image_generation_url(payload)
            except (json.JSONDecodeError, _ImageAPIError):
                pass
        # 最后回退：尝试直接从 generated_content 中提取 HTTP URL（整个字符串就是 URL）
        if not image_url:
            trimmed = generated_content.strip()
            if trimmed.startswith("http://") or trimmed.startswith("https://"):
                image_url = trimmed
        if not image_url:
            return "图片生成失败: 未在响应中找到图片 URL."

        try:
            image_bytes = await _generated_image_bytes(image_url)
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

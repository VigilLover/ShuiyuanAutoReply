import asyncio
import base64
import io
import json
import logging
import os
import re
import time
from datetime import datetime

import aiohttp
from PIL import Image

from shuiyuan_auto_reply.constants import settings

# Single and combined reference-image limits. Base64 increases the wire size.
_MAX_REFERENCE_BYTES = 10 * 1024 * 1024
_MAX_TOTAL_REFERENCE_BYTES = 20 * 1024 * 1024
_DEFAULT_IMAGE_MODEL = "gpt-image-2"
_DEFAULT_TIMEOUT_SECONDS = 600.0
_MAX_API_ATTEMPTS = 2
_RETRYABLE_HTTP_STATUSES = {408, 429}

logger = logging.getLogger(__name__)

_SUPPORTED_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
}
_SUPPORTED_IMAGE_SIZES = {"0.5K", "512", "1K", "2K", "4K"}
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


def _extract_image_url(content: str) -> str | None:
    """从 chat completions 返回的 Markdown 内容中提取图片 URL（含 data URL）"""
    if not isinstance(content, str):
        return None
    match = re.search(r"!\[.*?\]\(([^)]+)\)", content)
    if match:
        return match.group(1)
    return None


def _text_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return ""


def _completion_content(payload: object) -> str:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise _ImageAPIError("API 响应未包含 choices[0].message.") from None

    content = _text_content(message.get("content") if isinstance(message, dict) else None)
    if not content:
        raise _ImageAPIError("API 响应未包含可用的图片内容.")
    return content


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
    return False, False


async def _read_json_completion(response: aiohttp.ClientResponse) -> str:
    try:
        payload = await response.json(content_type=None)
    except (aiohttp.ClientError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _ImageAPIError(f"API 响应不是有效 JSON: {exc}") from exc
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
            _, non_json_event = _consume_sse_event(pending, pieces)
            if non_json_event:
                non_json_event_count += 1
        if non_json_event_count:
            logger.info(
                "Image API retained %d non-JSON SSE progress event(s) while awaiting output",
                non_json_event_count,
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
) -> str:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"

    request_started_at = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(api_url, headers=headers, data=payload_bytes) as response:
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
        - 无参考图（reference_images 为空）：必须用纯中文进行极其详细的画面描述，涵盖外貌、服饰、姿态、光影、背景、氛围等。如果绘画对象是人物，画风默认二次元精美插画，强调"唯美、精细、干净通透"，避免过度锐化、畸变与崩坏。若用户提供设定/附件/印象，必须将关键元素具象化融入画面。

        :param prompt: 详细的纯中文生图提示词。
        :param aspect_ratio: 画面宽高比，默认 1:1。支持 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 1:4, 4:1, 1:8, 8:1。
        :param image_size: 图片分辨率，默认 1K。支持 0.5K, 512, 1K, 2K, 4K。
        :param reference_images: 参考图片 URL 列表。传入 Python 列表格式如 [“upload://xxx.jpeg”]，支持 upload://、http(s)://、data: 等格式。        :param output_dir: 可选的自定义输出目录，用于保存生成的图片备份。
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
        if image_size not in _SUPPORTED_IMAGE_SIZES:
            image_size = "1K"

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
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })

        stream = _env_bool("IMAGE_GEN_STREAM", False)
        timeout_seconds = _image_timeout_seconds()
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
            "Submitting image generation: model=%s stream=%s reference_images=%d "
            "reference_bytes=%d request_bytes=%d timeout=%.0fs",
            image_model,
            stream,
            len(content) - 1,
            total_ref_bytes,
            len(payload_bytes),
            timeout_seconds,
        )

        queued_at = time.monotonic()
        generated_content = ""
        last_error = ""
        async with generation_gate:
            logger.info(
                "Image generation request acquired single-flight slot after %.2fs",
                time.monotonic() - queued_at,
            )
            for attempt in range(_MAX_API_ATTEMPTS):
                if attempt:
                    logger.warning(
                        "Retrying image API call (attempt %d/%d) after a transient failure; "
                        "the upstream may still bill or complete the prior request",
                        attempt + 1,
                        _MAX_API_ATTEMPTS,
                    )
                    await asyncio.sleep(2)
                started_at = time.monotonic()
                try:
                    generated_content = await _request_image_content(
                        api_url,
                        api_key,
                        payload_bytes,
                        stream=stream,
                        timeout_seconds=timeout_seconds,
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
                    if exc.retryable and attempt < _MAX_API_ATTEMPTS - 1:
                        logger.warning(
                            "Image API retryable response (attempt %d/%d): %s",
                            attempt + 1,
                            _MAX_API_ATTEMPTS,
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
                        _MAX_API_ATTEMPTS,
                        last_error,
                    )
                    if attempt < _MAX_API_ATTEMPTS - 1:
                        continue
                    return f"图片生成失败: API 调用异常 {last_error}"
                except Exception as exc:
                    logger.exception("Image API call failed")
                    return f"图片生成失败: API 调用异常 {exc}"
            else:
                return f"图片生成失败: API 调用异常 {last_error}"

        image_url = _extract_image_url(generated_content)
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

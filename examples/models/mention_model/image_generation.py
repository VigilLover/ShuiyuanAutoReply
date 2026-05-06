import base64
import json
import logging
import os
import re
from datetime import datetime

import aiohttp

from shuiyuan_auto_reply.constants import assets_directory

# 单张参考图最大体积 (10MB)，超过则跳过
_MAX_REFERENCE_BYTES = 10 * 1024 * 1024
# 所有参考图总体积上限 (20MB)
_MAX_TOTAL_REFERENCE_BYTES = 20 * 1024 * 1024

logger = logging.getLogger(__name__)

IMAGE_API_URL = os.getenv("IMAGE_GEN_API_URL", "https://www.openclaudecode.cn/v1/chat/completions")
IMAGE_MODEL = os.getenv("IMAGE_GEN_MODEL", "gpt-image-2-pro")

_SUPPORTED_ASPECT_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
}
_SUPPORTED_IMAGE_SIZES = {"0.5K", "512", "1K", "2K", "4K"}


def _extract_image_url(content: str) -> str | None:
    """从 chat completions 返回的 Markdown 内容中提取图片 URL（含 data URL）"""
    if not isinstance(content, str):
        return None
    match = re.search(r'!\[.*?\]\(([^)]+)\)', content)
    if match:
        return match.group(1)
    return None


async def _download_and_encode(
    session: aiohttp.ClientSession | None,
    url: str,
    *,
    shuiyuan_model=None,
    max_bytes: int = _MAX_REFERENCE_BYTES,
) -> str | None:
    """下载图片并转为 base64 data URL，整合了水源认证下载。

    支持所有水源社区图片格式：
    - data: URL → 直接返回
    - upload://xxx → 通过 ShuiyuanModel.download_image 认证下载（无需 cookie）
    - /uploads/short-url/xxx → 同 upload://，自动补前缀
    - http(s):// → 直接下载
    - 本地文件路径 → 读取并编码

    :param session: aiohttp session（可复用，为 None 时自动创建）
    :param url: 图片 URL 或路径
    :param shuiyuan_model: ShuiyuanModel 实例，用于认证下载 upload:// 格式
    :param max_bytes: 单张图片最大字节数，超过则跳过
    :return: data:image/...;base64,... 或 None
    """
    if url.startswith("data:"):
        return url

    # 统一处理 upload:// 和 /uploads/short-url/ 两种水源内部格式
    is_upload = url.startswith("upload://")
    is_short_path = url.startswith("/uploads/short-url/")
    if is_upload or is_short_path:
        # 转为 upload:// 统一格式
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

    # 本地文件路径
    if not url.startswith(("http://", "https://")):
        try:
            with open(url, "rb") as f:
                image_bytes = f.read()
        except Exception as exc:
            logger.warning("Read local reference image failed: %s %s", url, exc)
            return None
        return _encode_bytes(image_bytes, url, max_bytes)

    # HTTP(S) 下载
    try:
        if session is None:
            async with aiohttp.ClientSession() as _sess:
                return await _download_and_encode(_sess, url, shuiyuan_model=shuiyuan_model, max_bytes=max_bytes)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning("Download reference image failed: %s HTTP %s", url[:80], resp.status)
                return None
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                logger.warning("Reference image too large: %s bytes, skipping", content_length)
                return None
            image_bytes = await resp.read()
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
    b64 = base64.b64encode(image_bytes).decode("ascii")
    logger.info("Encoded reference image: %s (%d bytes)", source_hint[:80], len(image_bytes))
    return f"data:{mime};base64,{b64}"


def create_image_generation_tool(model):
    """
    创建一个与 ShuiyuanModel 绑定的文生图工具函数.

    :param model: ShuiyuanModel 实例, 用于调用 upload_image 上传图片到水源.
    :return: async callable, 可作为 StructuredTool 的 coroutine.
    """

    async def generate_image(
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        reference_images: str | list[str] | None = None,
        output_dir: str | None = None,
    ) -> str:
        """
        根据用户的文字描述生成图片，自动上传到水源并返回图片的短链接。

        【重要】你必须使用 Markdown 图片语法将返回的短链接嵌入最终回复：`![描述](短链接)`
        例如返回 `upload://zuyICpNdsQZCsV4cWeOwgcDLLak.jpeg`，你在回复中写 `![生成的图片](upload://zuyICpNdsQZCsV4cWeOwgcDLLak.jpeg)`

        提示词(prompt)编写规则（根据是否有参考图区别对待）：
        - 有参考图（reference_images 非空）：prompt 只需用纯中文简要描述原本要求，不要自行添加任何风格词或细节描写，让参考图主导视觉，并且强调“根据给定的参考图生成图片”。
        - 无参考图（reference_images 为空）：必须用纯中文进行极其详细的画面描述，涵盖外貌、服饰、姿态、光影、背景、氛围等。画风默认二次元精美插画，强调"唯美、精细、干净通透"，避免过度锐化、畸变与崩坏。若用户提供设定/附件/印象，必须将关键元素具象化融入画面。

        :param prompt: 详细的纯中文生图提示词。
        :param aspect_ratio: 画面宽高比，默认 1:1。支持 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 1:4, 4:1, 1:8, 8:1。
        :param image_size: 图片分辨率，默认 1K。支持 0.5K, 512, 1K, 2K, 4K。
        :param reference_images: 参考图片 URL 列表。传入 Python 列表格式如 ["upload://xxx.jpeg"]，支持 upload://、http(s)://、data: 等格式。
        :param output_dir: 可选的自定义输出目录，用于保存生成的图片备份。
        :return: 图片的短链接。你必须用 `![描述](链接)` 格式嵌入回复中。
        """
        # ── 参数归一化 ──
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            return "图片生成失败: IMAGE_GEN_API_KEY 未配置."

        if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
            aspect_ratio = "1:1"
        if image_size not in _SUPPORTED_IMAGE_SIZES:
            image_size = "1K"

        # reference_images 归一化为 list[str] | None
        if reference_images is None:
            pass
        elif isinstance(reference_images, list):
            pass  # 正确格式，无需处理
        elif isinstance(reference_images, str):
            # LLM 偶尔传入 JSON 字符串如 '["upload://xxx"]'，或单个裸 URL
            s = reference_images.strip()
            if s.startswith("["):
                try:
                    reference_images = json.loads(s)
                except (json.JSONDecodeError, TypeError):
                    reference_images = [s] if s else None
            else:
                reference_images = [s]  # 单个裸 URL → 包成列表
        else:
            reference_images = None

        # 构建 content 数组：文本 + 参考图
        content: list[dict] = [{"type": "text", "text": prompt}]
        if reference_images:
            total_ref_bytes = 0
            async with aiohttp.ClientSession() as session:
                for url in reference_images:
                    data_url = await _download_and_encode(session, url, shuiyuan_model=model)
                    if not data_url:
                        continue
                    # 估算 base64 解码后的字节数
                    b64_len = len(data_url.split(",", 1)[1]) if "," in data_url else len(data_url)
                    est_bytes = int(b64_len * 3 / 4)
                    if total_ref_bytes + est_bytes > _MAX_TOTAL_REFERENCE_BYTES:
                        logger.warning("Reference images total size would exceed %dMB, skipping remaining",
                                       _MAX_TOTAL_REFERENCE_BYTES // (1024 * 1024))
                        break
                    total_ref_bytes += est_bytes
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
            if total_ref_bytes:
                logger.info("Total reference images size: %.1fMB", total_ref_bytes / (1024 * 1024))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    IMAGE_API_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": IMAGE_MODEL,
                        "messages": [
                            {"role": "user", "content": content}
                        ],
                        "max_tokens": 4096,
                        "image_config": {
                            "aspect_ratio": aspect_ratio,
                            "image_size": image_size,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return f"图片生成失败: API 返回 HTTP {resp.status_code}, {body[:200]}"
                    data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("Image API response: %s", str(content)[:200])
        except Exception as exc:
            logger.error("Image API call failed: %s", exc)
            return f"图片生成失败: API 调用异常 {exc}"

        image_url = _extract_image_url(content)
        if not image_url:
            return f"图片生成失败: 未在响应中找到图片 URL."

        try:
            if image_url.startswith("data:"):
                import base64

                match = re.match(r"^data:[^;]+;base64,(.+)$", image_url)
                if not match:
                    return "图片生成失败: 无法解析 data URL。"
                image_bytes = base64.b64decode(match.group(1))
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        image_url, timeout=aiohttp.ClientTimeout(total=60)
                    ) as dl_resp:
                        if dl_resp.status != 200:
                            return f"图片生成失败: 下载图片 HTTP {dl_resp.status}"
                        image_bytes = await dl_resp.read()
            logger.info(
                "Got image: %d bytes (%.1fKB)",
                len(image_bytes),
                len(image_bytes) / 1024,
            )
        except Exception as exc:
            logger.error("Image download/parse failed: %s", exc)
            return f"图片生成失败: 下载图片异常 {exc}"

        try:
            backup_dir = output_dir or os.path.join(assets_directory, "generated_images")
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_prompt = prompt[:20].replace(" ", "_").replace("/", "_")
            backup_path = os.path.join(backup_dir, f"{timestamp}_{safe_prompt}.png")
            with open(backup_path, "wb") as f:
                f.write(image_bytes)
            logger.info("Saved backup to: %s", backup_path)
        except Exception as exc:
            logger.warning("Backup save failed (non-fatal): %s", exc)

        try:
            response = await model.upload_image(image_bytes)
            logger.info("Uploaded to Shuiyuan: %s", response.short_path)
            return response.short_path
        except Exception as exc:
            logger.error("Shuiyuan upload failed: %s", exc)
            return f"图片生成失败: 上传到水源异常 {exc}"

    return generate_image

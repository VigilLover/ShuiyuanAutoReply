import logging
import os
import re
from datetime import datetime

import aiohttp

from shuiyuan_auto_reply.constants import assets_directory

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


def create_image_generation_tool(model):
    """
    创建一个与 ShuiyuanModel 绑定的文生图工具函数.

    :param model: ShuiyuanModel 实例, 用于调用 upload_image 上传图片到水源.
    :return: async callable, 可作为 StructuredTool 的 coroutine.
    """

    async def generate_image(prompt: str, aspect_ratio: str = "1:1", image_size: str = "1K") -> str:
        """
        根据用户的文字描述生成图片，自动上传到水源并返回图片的短链接。

        【重要】你必须使用 Markdown 图片语法将返回的短链接嵌入最终回复：`![描述](短链接)`
        例如返回 `https://shuiyuan.sjtu.edu.cn/uploads/short-url/xxx`，你在回复中写 `![生成的图片](https://shuiyuan.sjtu.edu.cn/uploads/short-url/xxx)`

        提示词(prompt)编写参数规则：
        1. 必须使用纯中文进行极其详细的画面描述。
        2. 场景与人物扩写需精细：涵盖外貌特征、服饰、姿态、神态、光影、背景、时间、氛围等可视化细节。
        3. 画风设定为二次元精美插画，强调"唯美、精细、干净通透"，并要求"避免过度锐化、畸变与崩坏"。
        4. 若用户提供设定/附件/"用户印象"，必须将关键元素具象化融入画面。

        :param prompt: 优化后详细的纯中文生图提示词。
        :param aspect_ratio: 画面宽高比，默认 1:1。支持 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 1:4, 4:1, 1:8, 8:1。
        :param image_size: 图片分辨率，默认 1K。支持 0.5K, 512, 1K, 2K, 4K。
        :return: 图片的短链接。你必须用 `![描述](链接)` 格式嵌入回复中。
        """
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            return "图片生成失败: IMAGE_GEN_API_KEY 未配置."

        if aspect_ratio not in _SUPPORTED_ASPECT_RATIOS:
            aspect_ratio = "1:1"
        if image_size not in _SUPPORTED_IMAGE_SIZES:
            image_size = "1K"

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
                            {"role": "user", "content": [{"type": "text", "text": prompt}]}
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
            backup_dir = os.path.join(assets_directory, "generated_images")
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
            logger.info("Uploaded to Shuiyuan: %s", response.short_url)
            return response.short_url
        except Exception as exc:
            logger.error("Shuiyuan upload failed: %s", exc)
            return f"图片生成失败: 上传到水源异常 {exc}"

    return generate_image

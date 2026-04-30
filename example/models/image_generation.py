import logging
import os
import re
from datetime import datetime

import aiohttp
import requests

from shuiyuan_auto_reply.constants import assets_directory

logger = logging.getLogger(__name__)

IMAGE_API_URL = os.getenv("IMAGE_GEN_API_URL", "https://www.openclaudecode.cn/v1/chat/completions")
IMAGE_MODEL = os.getenv("IMAGE_GEN_MODEL", "gpt-image-2-pro")


def _extract_image_url(content: str) -> str | None:
    """从 chat completions 返回的 Markdown 内容中提取图片 URL"""
    if not isinstance(content, str):
        return None
    match = re.search(r'!\[.*?\]\(([^)]+)\)', content)
    if match:
        url = match.group(1)
        if not url.startswith("data:"):
            return url
    return None


def create_image_generation_tool(model):
    """
    创建一个与 ShuiyuanModel 绑定的文生图工具函数.

    :param model: ShuiyuanModel 实例, 用于调用 try_upload_image 上传图片到水源.
    :return: async callable, 可作为 StructuredTool 的 coroutine.
    """

    async def generate_image(prompt: str) -> str:
        """
        根据文字描述生成图片, 使用 GPT Image 模型, 自动上传到水源并返回 Markdown 图片链接.

        适用场景:
        - 用户要求画一张图、生成图片、帮我画个等与绘图、插图、生成图片相关的需求.
        - 用户希望将一段文字描述转换为可视化的图像.

        调用前必须做的提示词优化:
        在调用本工具之前, 你必须先将用户的原始需求改写为一段仅使用中文的生图提示词, 且必须尽可能详细, 不要在意字数长短, 规则如下:
        1. 只允许输出中文提示词, 不要混入英文句子; 可保留极少量通用风格标签但优先中文表达.
        2. 将场景扩写到足够细致, 细化并覆盖: 人物外貌特征(发型、发色、瞳色、五官、年龄感)、服装材质与褶皱、姿态动作、面部神态、镜头景别与机位、前中后景层次、空间透视关系、背景物件与装饰细节、时间天气季节、主辅光源方向与色温、色彩搭配、画面情绪与叙事氛围.
        3. 明确强调极致画质与细节密度, 可使用中文高质量关键词, 如: 杰作、顶级画质、超高精细、细节拉满、电影级光影、体积光、柔和辉光、超清背景、纹理清晰、边缘干净.
        4. 画风以精美二次元人物插画为主, 强调: 唯美、通透、精致、人物脸部和眼睛刻画细腻、服饰与道具细节丰富、整体观感高级.
        5. 只写可视化、可落地的具体描述, 不写空泛抽象词; 让模型能直接据此作画, 不要包含反向提示词(negative prompt).
        6. 当用户提供参考文本、人物信息或附件内容时, 必须尽可能完整吸收并具象化到画面里, 宁可写得更长更细, 也不要省略关键元素.
        7. 如果是“用户印象图”, 必须围绕该用户昵称与特征设计个性化场景, 增加可识别的专属元素与氛围细节, 让画面具有强记忆点.
        8. 输出时默认采用一整段长提示词, 优先保证信息密度和可视化细节完整性, 不因篇幅主动删减内容.

        返回值说明:
        - 成功时返回水源论坛的 Markdown 图片链接, 图片会自动附加到你的回复末尾.
        - 你只需用文字简要描述图片内容, 无需手动放置图片链接.

        :param prompt: 优化后的中文生图提示词(必须按照上述规则改写后再传入)
        :return: Markdown 图片链接, 系统会自动追加到回复.
        """
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            return "图片生成失败: IMAGE_GEN_API_KEY 未配置."

        try:
            resp = requests.post(
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
                },
                timeout=240,
            )
            if resp.status_code != 200:
                return f"图片生成失败: API 返回 HTTP {resp.status_code}"

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("Image API response: %s", str(content)[:200])
        except Exception as exc:
            logger.error("Image API call failed: %s", exc)
            return f"图片生成失败: API 调用异常 {exc}"

        image_url = _extract_image_url(content)
        if not image_url:
            return f"图片生成失败: 未在响应中找到图片 URL."

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=60)) as dl_resp:
                    if dl_resp.status != 200:
                        return f"图片生成失败: 下载图片 HTTP {dl_resp.status}"
                    image_bytes = await dl_resp.read()
            logger.info("Downloaded image: %d bytes (%.1fKB)", len(image_bytes), len(image_bytes) / 1024)
        except Exception as exc:
            logger.error("Image download failed: %s", exc)
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
            img_url = await model.try_upload_image(
                image_bytes, try_base64=True, try_base64_size_kb=40
            )
            logger.info("Uploaded to Shuiyuan: type=%s", img_url.type)
            return img_url.data
        except Exception as exc:
            logger.error("Shuiyuan upload failed: %s", exc)
            return f"图片生成失败: 上传到水源异常 {exc}"

    return generate_image

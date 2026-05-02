import asyncio
import logging
import os
import sys

import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _MockModel:
    """模拟 ShuiyuanModel, 只记录上传调用而不实际请求水源"""

    def __init__(self):
        self.uploaded_images = []

    async def try_upload_image(self, image_bytes, try_base64=True, try_base64_size_kb=40):
        from shuiyuan_auto_reply.shuiyuan.objects import ImageURL
        self.uploaded_images.append(image_bytes)
        logging.info(
            f"==> [Mock] try_upload_image called: {len(image_bytes) / 1024:.1f}KB, "
            f"skip real upload to Shuiyuan"
        )
        return ImageURL("url", f"![img](mock_uploaded_url_{len(self.uploaded_images)})")


async def test_image_generation(prompt: str):
    """测试本地图片生成工具 (生成 + 下载 + 本地备份, 不上传水源)"""
    from examples.models.mention_model.image_generation import create_image_generation_tool

    # 检查配置
    api_key = os.getenv("IMAGE_GEN_API_KEY")
    if not api_key:
        logging.error("IMAGE_GEN_API_KEY is not set in .env")
        return

    api_url = os.getenv("IMAGE_GEN_API_URL", "https://www.openclaudecode.cn/v1/chat/completions")
    model = os.getenv("IMAGE_GEN_MODEL", "gpt-image-2-pro")
    logging.info(f"==> [Test] API URL: {api_url}")
    logging.info(f"==> [Test] Model:   {model}")
    logging.info(f"==> [Test] Key:     {api_key[:12]}...")

    # 创建工具函数 (与 MentionChatModel._load_shuiyuan_tools() 相同方式)
    mock_model = _MockModel()
    gen_img_func = create_image_generation_tool(mock_model)

    logging.info(f"==> [Test] Calling generate_image with prompt: '{prompt[:100]}...'")

    # 与 Agent 调用工具的相同方式 — 直接 await 工具函数
    result = await gen_img_func(prompt)

    logging.info(f"==> [Test] Result type: {type(result).__name__}")
    logging.info(f"==> [Test] Result: {str(result)[:500]}")

    if mock_model.uploaded_images:
        total_kb = sum(len(img) for img in mock_model.uploaded_images) / 1024
        logging.info(f"==> [Test] Uploaded {len(mock_model.uploaded_images)} image(s) ({total_kb:.1f}KB total)")
        logging.info(f"==> [Test] Images saved locally in assets/generated_images/")
    else:
        logging.warning("==> [Test] No images were uploaded.")


if __name__ == "__main__":
    _DEFAULT_PROMPT = (
        "一幅高细节的二次元风格插画：一个带有狼耳和蓬松尾巴的少年坐在卧室书桌前。他有凌乱的银灰色短发和温柔的蓝色眼睛，单手托着脸，神情略显疲惫又安静。他穿着一件宽松的浅色连帽卫衣。"

        "书桌上摆满了生活化的小物件：一台打开的笔记本电脑（屏幕是蓝色主题网页）、一部正在播放幻想类游戏画面的手机、一份汉堡和薯条、一杯外带饮料。桌上还有一个发光的南瓜灯和一个可爱的粉色小猪摆件。"

        "他身后的墙上贴着旅行照片（京都塔、大阪城）、一张写有手写标注的日历、一些中日英混合的便利贴，以及一张车票或机票。"

        "右侧窗户透进温暖的金色夕阳光，照亮房间并形成柔和阴影，窗外是傍晚的彩色云霞天空。整体氛围温馨、安静、治愈，具有生活感。"

        "二次元插画风格，光影柔和，暖色调，高细节，构图干净，日常感，画面精致。anime style, anime art, manga style, highly detailed, soft lighting, masterpiece, high quality."
    )
    prompt = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PROMPT
    asyncio.run(test_image_generation(prompt))

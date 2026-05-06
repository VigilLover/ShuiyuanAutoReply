"""
Test image generation tool, including reference-image support.

Usage:
    python -m pytest test/test_image_generation.py -v
    python -m pytest test/test_image_generation.py::TestImageGeneration -v
    python -m pytest test/test_image_generation.py::TestImageGenerationWithReference -v
"""

import asyncio
import base64
import logging
import os
import sys
import unittest
from pathlib import Path

import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.models.mention_model.image_generation import (
    _download_and_encode,
    _extract_image_url,
    create_image_generation_tool,
)

_TEST_DIR = Path(__file__).resolve().parent
_REFERENCE_DIR = _TEST_DIR / "image_generation_reference"
_OUTPUT_DIR = _TEST_DIR / "image_generation_test"

# Shuiyuan 上传链接（upload:// 格式，与 reference 目录中的 17663417780751842.jpg 对应）
_SHUIYUAN_UPLOAD_PATH = "upload://nmJhpoTDTvnjmrYg0oIn8dZFVF2.jpeg"
# 水源图片下载的基础 URL（与 src/.../constants.py 中 download_url 一致）
_SHUIYUAN_DOWNLOAD_BASE = "https://shuiyuan.sjtu.edu.cn/uploads/short-url"


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


_load_env()
os.makedirs(_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class _MockModel:
    def __init__(self):
        self.uploaded_images = []

    async def upload_image(self, image_bytes):
        from shuiyuan_auto_reply.shuiyuan.objects import ImageUploadResponse
        self.uploaded_images.append(image_bytes)
        logging.info("==> [Mock] upload_image: %.1fKB", len(image_bytes) / 1024)
        idx = len(self.uploaded_images)
        return ImageUploadResponse(
            id=idx,
            url=f"mock_url_{idx}",
            original_filename=f"mock_{idx}.png",
            short_url=f"mock_short_url_{idx}",
            short_path=f"upload://mockShortPath{idx}.jpeg",
        )

    async def download_image(self, image_url: str) -> bytes:
        """模拟 ShuiyuanModel.download_image：将 upload:// 解析为 HTTPS 并下载"""
        if not image_url.startswith("upload://"):
            raise ValueError(f"Invalid image URL: {image_url}")
        resolved = image_url.replace("upload://", _SHUIYUAN_DOWNLOAD_BASE + "/")
        logging.info("==> [Mock] download_image: %s → %s", image_url, resolved)
        async with aiohttp.ClientSession() as session:
            async with session.get(resolved) as resp:
                if resp.status != 200:
                    raise Exception(f"Download failed: HTTP {resp.status}")
                data = await resp.read()
        logging.info("==> [Mock] downloaded: %.1fKB", len(data) / 1024)
        return data


# ── 基础文生图 ────────────────────────────────────────────────────────

class TestImageGenerationBasic(unittest.IsolatedAsyncioTestCase):
    """纯文本文生图（无参考图）"""

    async def test_generate_image_basic(self):
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            self.skipTest("IMAGE_GEN_API_KEY not set")

        mock = _MockModel()
        gen_img = create_image_generation_tool(mock)
        prompt = (
            "一幅简单的二次元风格插画，1:1。一只白色的卡通小猫坐在木地板上，"
            "背景为纯浅蓝色，阳光从左侧窗户照入。简洁干净，无多余细节。"
        )
        result = await gen_img(prompt, aspect_ratio="1:1", image_size="0.5K", output_dir=str(_OUTPUT_DIR))
        logging.info("Result: %s", str(result)[:200])
        self.assertTrue(result)
        self.assertFalse(result.startswith("图片生成失败"))
        self.assertEqual(len(mock.uploaded_images), 1)


# ── 参考图生图 ────────────────────────────────────────────────────────

class TestImageGenerationWithReference(unittest.IsolatedAsyncioTestCase):
    """使用参考图进行文生图"""

    async def test_with_local_reference(self):
        """使用 test/image_generation_reference/ 中的本地图片作为参考"""
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            self.skipTest("IMAGE_GEN_API_KEY not set")

        ref_files = list(_REFERENCE_DIR.glob("*"))
        if not ref_files:
            self.skipTest("No reference images in image_generation_reference/")
        ref_path = str(ref_files[0])
        logging.info("Using local reference: %s", ref_path)

        mock = _MockModel()
        gen_img = create_image_generation_tool(mock)
        prompt = (
            "参考提供的图片，生成一张新的图片。"
            "保持相似的风格和角色特征，可适当加入新的姿态或场景元素。"
        )
        result = await gen_img(
            prompt, aspect_ratio="1:1", image_size="1K",
            reference_images=[ref_path], output_dir=str(_OUTPUT_DIR),
        )
        logging.info("Result: %s", str(result)[:200])
        self.assertTrue(result)
        self.assertFalse(result.startswith("图片生成失败"))
        self.assertEqual(len(mock.uploaded_images), 1)

    async def test_with_shuiyuan_reference_local_file(self):
        """使用 image_generation_reference 中的本地参考图（该文件即水源 upload:// 对应图片）"""
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            self.skipTest("IMAGE_GEN_API_KEY not set")

        ref_files = list(_REFERENCE_DIR.glob("*"))
        if not ref_files:
            self.skipTest("No reference images in image_generation_reference/")
        ref_path = str(ref_files[0])
        logging.info("Using reference (downloaded from upload://): %s", ref_path)

        mock = _MockModel()
        gen_img = create_image_generation_tool(mock)
        prompt = (
            "参考提供的图片，生成一张新的图片。"
            "保持相似的风格和角色特征，可适当加入新的姿态或场景元素。"
        )
        result = await gen_img(
            prompt, aspect_ratio="1:1", image_size="1K",
            reference_images=[ref_path], output_dir=str(_OUTPUT_DIR),
        )
        logging.info("Result: %s", str(result)[:200])
        self.assertTrue(result)
        self.assertFalse(result.startswith("图片生成失败"))
        self.assertEqual(len(mock.uploaded_images), 1)

    async def test_with_mixed_references(self):
        """混合使用本地参考图和小型 HTTP 参考图"""
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            self.skipTest("IMAGE_GEN_API_KEY not set")

        ref_files = list(_REFERENCE_DIR.glob("*"))
        refs = []
        if ref_files:
            refs.append(str(ref_files[0]))
        # 第二张用小图 HTTP URL 避免请求体过大导致服务端断开
        refs.append("https://www.python.org/static/img/python-logo.png")

        mock = _MockModel()
        gen_img = create_image_generation_tool(mock)
        prompt = "参考提供的两张图片风格，生成一张新的图片。"
        result = await gen_img(
            prompt, aspect_ratio="1:1", image_size="1K",
            reference_images=refs, output_dir=str(_OUTPUT_DIR),
        )
        logging.info("Result: %s", str(result)[:200])
        self.assertTrue(result)
        self.assertFalse(result.startswith("图片生成失败"))
        self.assertEqual(len(mock.uploaded_images), 1)


# ── Shuiyuan download_image ────────────────────────────────────────────

class TestShuiyuanDownloadImage(unittest.IsolatedAsyncioTestCase):
    """测试 upload:// 格式图片通过 download_image 下载（需要有效的 Shuiyuan cookies）"""

    async def test_download_upload_image(self):
        """upload:// 应通过 Shuiyuan 认证下载为图片 bytes"""
        from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
        cookies_path = os.path.join(PROJECT_ROOT, "cookies")
        if not os.path.exists(cookies_path):
            self.skipTest("Cookies file not found, cannot auth with Shuiyuan")

        model = await ShuiyuanModel.create(cookies_path)
        async with model:
            image_bytes = await model.download_image(_SHUIYUAN_UPLOAD_PATH)
        self.assertIsNotNone(image_bytes)
        self.assertGreater(len(image_bytes), 1000)
        is_jpeg = image_bytes[:3] == b"\xff\xd8\xff"
        is_png = image_bytes[:4] == b"\x89PNG"
        self.assertTrue(is_jpeg or is_png, "Should be valid image format (JPEG or PNG)")
        logging.info("download_image OK: %d bytes", len(image_bytes))


# ── 图片下载/编码工具 ─────────────────────────────────────────────────

class TestDownloadAndEncode(unittest.IsolatedAsyncioTestCase):
    async def test_encode_local_file(self):
        """本地文件应正确转为 base64 data URL"""
        ref_files = list(_REFERENCE_DIR.glob("*"))
        if not ref_files:
            self.skipTest("No reference images found")
        ref_path = str(ref_files[0])

        data_url = await _download_and_encode(None, ref_path)
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/"))
        b64_part = data_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        self.assertGreater(len(decoded), 100)

    async def test_encode_http_url(self):
        """HTTP URL 应正确下载并编码"""
        async with aiohttp.ClientSession() as session:
            data_url = await _download_and_encode(session, "https://www.python.org/static/img/python-logo.png")
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_pass_through_data_url(self):
        """已是 data URL 应直接返回"""
        result = asyncio.run(_download_and_encode(None, "data:image/png;base64,abc123"))
        self.assertEqual(result, "data:image/png;base64,abc123")


# ── URL 提取工具 ──────────────────────────────────────────────────────

class TestExtractImageURL(unittest.TestCase):
    def test_standard_markdown(self):
        self.assertEqual(_extract_image_url("![desc](https://example.com/img.png)"), "https://example.com/img.png")

    def test_upload_format(self):
        self.assertEqual(_extract_image_url("![img](upload://abc123.jpeg)"), "upload://abc123.jpeg")

    def test_data_url(self):
        self.assertEqual(_extract_image_url("![img](data:image/png;base64,abc123)"), "data:image/png;base64,abc123")

    def test_plain_text_returns_none(self):
        self.assertIsNone(_extract_image_url("hello world"))

    def test_none_input_returns_none(self):
        self.assertIsNone(_extract_image_url(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)

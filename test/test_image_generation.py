"""
Test image generation tool, including reference-image support.

Usage:
    python -m pytest test/test_image_generation.py -v
    python -m pytest test/test_image_generation.py::TestImageGeneration -v
    python -m pytest test/test_image_generation.py::TestImageGenerationWithReference -v
"""

import asyncio
import base64
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
from aiohttp import web
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shuiyuan_auto_reply.features.mention.image_generation import (
    _download_and_encode,
    _encode_bytes,
    _image_api_endpoint,
    _image_request_timeout,
    _openai_image_size,
    create_image_generation_tool,
)

_TEST_DIR = Path(__file__).resolve().parent
_REFERENCE_DIR = _TEST_DIR / "image_generation_reference"
_OUTPUT_DIR = _TEST_DIR / "image_generation_test"

# Shuiyuan 上传链接（upload:// 格式，与 reference 目录中的 17663417780751842.jpg 对应）
_SHUIYUAN_UPLOAD_PATH = "upload://nmJhpoTDTvnjmrYg0oIn8dZFVF2.jpeg"
# 水源图片下载的基础 URL（与 src/.../constants.py 中 download_url 一致）
_SHUIYUAN_DOWNLOAD_BASE = "https://shuiyuan.sjtu.edu.cn/uploads/short-url"


def _require_live_image_api(testcase):
    if os.getenv("RUN_IMAGE_GEN_INTEGRATION") != "1":
        testcase.skipTest("Set RUN_IMAGE_GEN_INTEGRATION=1 to run live image API tests")


def _png_data_url(color=(30, 100, 220, 128)) -> str:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), color).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


class TestImageGenerationTransport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = _MockModel()
        self.requests = []
        self.runner = None
        self.output_dir = tempfile.TemporaryDirectory()
        self.env_patch = patch.dict(
            os.environ,
            {
                "IMAGE_GEN_API_KEY": "test-key",
                "IMAGE_GEN_MODEL": "test-image-model",
                "IMAGE_GEN_TIMEOUT_SECONDS": "5",
                "IMAGE_GEN_MAX_ATTEMPTS": "1",
                "IMAGE_GEN_RETRY_BASE_DELAY_SECONDS": "5",
            },
            clear=False,
        )
        self.env_patch.start()

    async def asyncTearDown(self):
        if self.runner is not None:
            await self.runner.cleanup()
        self.env_patch.stop()
        self.output_dir.cleanup()

    async def _start_images_server(self, generations_handler, edits_handler=None):
        app = web.Application()
        app.router.add_post("/v1/images/generations", generations_handler)
        if edits_handler is not None:
            app.router.add_post("/v1/images/edits", edits_handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        os.environ["IMAGE_GEN_API_URL"] = f"http://127.0.0.1:{port}/v1"

    @staticmethod
    def _images_response_url():
        return {
            "data": [
                {"url": _png_data_url()}
            ]
        }

    @staticmethod
    def _images_response_b64():
        raw = base64.b64decode(_png_data_url().split(",", 1)[1])
        return {
            "data": [
                {"b64_json": base64.b64encode(raw).decode("ascii")}
            ]
        }

    async def test_generation_endpoint_uses_openai_images_contract(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("测试原生图片生成接口调用", aspect_ratio="3:4", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(
            self.requests[0],
            {
                "model": "test-image-model",
                "prompt": "测试原生图片生成接口调用",
                "size": "1024x1360",
            },
        )
        self.assertNotIn("messages", self.requests[0])
        self.assertNotIn("image_config", self.requests[0])

    async def test_edit_endpoint_is_selected_when_reference_images_exist(self):
        reference = _png_data_url()
        seen_paths = []

        async def generations_handler(request):
            seen_paths.append(request.path)
            return web.Response(status=500, text="generation should not be used")

        async def edits_handler(request):
            seen_paths.append(request.path)
            self.assertTrue(request.content_type.startswith("multipart/"))
            reader = await request.multipart()
            fields = {}
            images = []
            async for part in reader:
                if part.name == "image[]":
                    images.append(
                        {
                            "filename": part.filename,
                            "field_name": part.name,
                            "content_type": part.headers.get("Content-Type"),
                            "bytes": await part.read(),
                        }
                    )
                else:
                    fields[part.name] = await part.text()
            fields["image_count"] = len(images)
            fields["image_field_names"] = [image["field_name"] for image in images]
            fields["image_content_types"] = [image["content_type"] for image in images]
            fields["image_bytes"] = [len(image["bytes"]) for image in images]
            self.requests.append(fields)
            return web.json_response(self._images_response_b64())

        await self._start_images_server(generations_handler, edits_handler)
        tool = create_image_generation_tool(self.model)
        result = await tool(
            "参考图编辑测试生图功能验证",
            aspect_ratio="3:4",
            reference_images=[reference],
            output_dir=self.output_dir.name,
        )

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(seen_paths, ["/v1/images/edits"])
        self.assertEqual(
            self.requests[0],
            {
                "model": "test-image-model",
                "prompt": "参考图编辑测试生图功能验证",
                "size": "1024x1360",
                "image_count": 1,
                "image_field_names": ["image[]"],
                "image_content_types": ["image/png"],
                "image_bytes": [len(base64.b64decode(reference.split(",", 1)[1]))],
            },
        )

    async def test_url_response_is_downloaded_and_uploaded(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_url())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("测试URL响应图片的下载和上传", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(len(self.model.uploaded_images), 1)

    async def test_non_200_response_code_and_body_are_returned_to_tool_caller(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(
                {"error": {"message": "bad image prompt"}},
                status=400,
            )

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("测试服务端错误信息回传给机器人", output_dir=self.output_dir.name)

        self.assertIn("图片生成失败: API 返回 HTTP 400", result)
        self.assertIn("bad image prompt", result)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.model.uploaded_images, [])

    async def test_4router_request_id_is_returned_with_http_error(self):
        async def handler(request):
            return web.json_response(
                {"error": {"message": "control plane unavailable"}},
                status=502,
                headers={"x-oneapi-request-id": "request-abc123"},
            )

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("测试图片代理请求编号错误回传", output_dir=self.output_dir.name)

        self.assertIn("HTTP 502", result)
        self.assertIn("4Router request_id=request-abc123", result)

    async def test_numeric_prompt_values_are_rejected_before_server_request(self):
        async def handler(request):
            self.requests.append(await request.read())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)

        for prompt in ("0", "1", 0, 1):
            result = await tool(prompt, output_dir=self.output_dir.name)
            self.assertTrue(result.startswith("图片生成失败"))

        self.assertEqual(self.requests, [])
        self.assertEqual(self.model.uploaded_images, [])

    async def test_repeated_valid_prompt_still_submits_real_requests(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        prompt = "重复有效提示词也必须真实请求服务器"

        first = await tool(prompt, output_dir=self.output_dir.name)
        second = await tool(prompt, output_dir=self.output_dir.name)

        self.assertEqual(first, "upload://mockShortPath1.jpeg")
        self.assertEqual(second, "upload://mockShortPath2.jpeg")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual([request["prompt"] for request in self.requests], [prompt, prompt])

    async def test_runtime_model_config_is_read_for_each_call(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        os.environ["IMAGE_GEN_MODEL"] = "runtime-model-after-import"

        result = await tool("测试运行时模型配置读取功能", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(self.requests[0]["model"], "runtime-model-after-import")

    async def test_disconnect_is_not_retried_by_safe_default(self):
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            await request.read()
            if attempts == 1:
                request.transport.close()
                return web.Response()
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("测试断连不重提独立请求", output_dir=self.output_dir.name)

        self.assertIn("未提供断线续取能力", result)
        self.assertEqual(attempts, 1)
        self.assertEqual(self.model.uploaded_images, [])

    async def test_explicit_multiple_attempts_repeat_submissions_with_exponential_backoff(self):
        os.environ["IMAGE_GEN_MAX_ATTEMPTS"] = "4"
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            await request.read()
            request.transport.close()
            return web.Response()

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ) as mocked_sleep:
            result = await tool("测试连续断连重试提交功能", output_dir=self.output_dir.name)

        self.assertIn("已执行的重试均为独立请求", result)
        self.assertEqual(attempts, 4)
        self.assertEqual(
            mocked_sleep.await_args_list,
            [call(5.0), call(10.0), call(20.0)],
        )
        self.assertEqual(self.model.uploaded_images, [])

    async def test_retry_count_and_base_delay_can_be_configured(self):
        os.environ["IMAGE_GEN_MAX_ATTEMPTS"] = "3"
        os.environ["IMAGE_GEN_RETRY_BASE_DELAY_SECONDS"] = "0.25"
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            await request.read()
            request.transport.close()
            return web.Response()

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ) as mocked_sleep:
            result = await tool("测试配置重连次数和延迟参数", output_dir=self.output_dir.name)

        self.assertIn("API 连接异常", result)
        self.assertIn("无法接收该次 response", result)
        self.assertEqual(attempts, 3)
        self.assertEqual(mocked_sleep.await_args_list, [call(0.25), call(0.5)])

    async def test_retryable_http_status_retries_once(self):
        os.environ["IMAGE_GEN_MAX_ATTEMPTS"] = "2"
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return web.Response(status=503, text="busy")
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ):
            result = await tool("测试HTTP状态码重试处理逻辑", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(attempts, 2)

    async def test_image_generation_requests_are_serialized(self):
        active_requests = 0
        maximum_active_requests = 0
        submissions = 0
        first_started = asyncio.Event()
        release = asyncio.Event()

        async def handler(request):
            nonlocal active_requests, maximum_active_requests, submissions
            submissions += 1
            active_requests += 1
            maximum_active_requests = max(maximum_active_requests, active_requests)
            first_started.set()
            await release.wait()
            active_requests -= 1
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        first = asyncio.create_task(tool("第一个并发生图测试请求任务", output_dir=self.output_dir.name))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = asyncio.create_task(tool("第二个并发生图测试请求任务", output_dir=self.output_dir.name))
        await asyncio.sleep(0.05)

        self.assertEqual(submissions, 1)
        release.set()
        results = await asyncio.gather(first, second)

        self.assertEqual(maximum_active_requests, 1)
        self.assertEqual(submissions, 2)
        self.assertEqual(len(self.model.uploaded_images), 2)
        self.assertTrue(all(result.startswith("upload://") for result in results))


# ── 基础文生图 ────────────────────────────────────────────────────────

class TestImageGenerationBasic(unittest.IsolatedAsyncioTestCase):
    """纯文本文生图（无参考图）"""

    async def test_generate_image_basic(self):
        _require_live_image_api(self)
        api_key = os.getenv("IMAGE_GEN_API_KEY")
        if not api_key:
            self.skipTest("IMAGE_GEN_API_KEY not set")

        mock = _MockModel()
        gen_img = create_image_generation_tool(mock)
        prompt = (
            "一幅简单的二次元风格插画，1:1。一只白色的卡通小猫坐在木地板上，"
            "背景为纯浅蓝色，阳光从左侧窗户照入。简洁干净，无多余细节。"
        )
        result = await gen_img(prompt, aspect_ratio="1:1", image_size="1K", output_dir=str(_OUTPUT_DIR))
        logging.info("Result: %s", str(result)[:200])
        self.assertTrue(result)
        self.assertFalse(result.startswith("图片生成失败"))
        self.assertEqual(len(mock.uploaded_images), 1)


# ── 参考图生图 ────────────────────────────────────────────────────────

class TestImageGenerationWithReference(unittest.IsolatedAsyncioTestCase):
    """使用参考图进行文生图"""

    async def test_with_local_reference(self):
        """使用 test/image_generation_reference/ 中的本地图片作为参考"""
        _require_live_image_api(self)
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
        _require_live_image_api(self)
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
        _require_live_image_api(self)
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
        _require_live_image_api(self)
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
    async def test_shuiyuan_avatar_uses_authenticated_raw_image_download(self):
        model = MagicMock()
        model.download_image = AsyncMock()
        model.download_raw_image = AsyncMock(
            return_value=base64.b64decode(_png_data_url().split(",", 1)[1])
        )
        avatar_url = (
            "https://shuiyuan.sjtu.edu.cn/user_avatar/"
            "shuiyuan.sjtu.edu.cn/wolf_lumine/288/2066071_2.png"
        )

        result = await _download_and_encode(
            None,
            avatar_url,
            shuiyuan_model=model,
            strict_remote=True,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/png;base64,"))
        model.download_raw_image.assert_awaited_once_with(
            "/user_avatar/shuiyuan.sjtu.edu.cn/wolf_lumine/288/2066071_2.png"
        )
        model.download_image.assert_not_called()

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
        _require_live_image_api(self)
        async with aiohttp.ClientSession() as session:
            data_url = await _download_and_encode(session, "https://www.python.org/static/img/python-logo.png")
        self.assertIsNotNone(data_url)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_pass_through_data_url(self):
        """有效 data URL 应正确保留为可用图片引用"""
        data_url = _png_data_url()
        result = asyncio.run(_download_and_encode(None, data_url))
        self.assertEqual(result, data_url)

    def test_data_url_respects_single_reference_limit(self):
        result = asyncio.run(
            _download_and_encode(None, "data:image/png;base64,aGVsbG8=", max_bytes=1)
        )
        self.assertIsNone(result)

    def test_compressed_png_reference_is_labeled_as_jpeg(self):
        buffer = io.BytesIO()
        Image.new("RGB", (1300, 20), (200, 40, 40)).save(buffer, format="PNG")

        result = _encode_bytes(buffer.getvalue(), "reference.png", max_bytes=1024 * 1024)

        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        decoded = base64.b64decode(result.split(",", 1)[1])
        self.assertTrue(decoded.startswith(b"\xff\xd8\xff"))


class TestImageRequestTimeout(unittest.TestCase):
    def test_timeout_limits_silence_without_total_deadline(self):
        timeout = _image_request_timeout(600)
        self.assertIsNone(timeout.total)
        self.assertEqual(timeout.connect, 30.0)
        self.assertEqual(timeout.sock_read, 600)


class TestImageAPIEndpoint(unittest.TestCase):
    def test_appends_generation_endpoint_to_openai_base_url(self):
        self.assertEqual(
            _image_api_endpoint("https://4router.net/v1", "generations"),
            "https://4router.net/v1/images/generations",
        )

    def test_appends_edit_endpoint_to_openai_base_url_with_trailing_slash(self):
        self.assertEqual(
            _image_api_endpoint("https://4router.net/v1/", "edits"),
            "https://4router.net/v1/images/edits",
        )


class TestOpenAIImageSize(unittest.TestCase):
    def test_edges_are_aligned_for_custom_portrait_ratio(self):
        self.assertEqual(_openai_image_size("3:4"), "1024x1360")

    def test_extreme_ratio_fits_gpt_image_2_limits_at_max_allowed_size(self):
        width, height = map(int, _openai_image_size("1:8").split("x"))
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertLessEqual(max(width, height), 3840)
        self.assertLessEqual(width * height, 8_294_400)
        self.assertLessEqual(max(width, height) / min(width, height), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

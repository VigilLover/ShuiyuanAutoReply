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
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import aiohttp
from aiohttp import web
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.models.mention_model.image_generation import (
    _download_and_encode,
    _extract_image_url,
    _image_request_timeout,
    _native_image_size,
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
                "IMAGE_GEN_STREAM": "false",
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

    async def _start_server(self, handler):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        os.environ["IMAGE_GEN_API_URL"] = f"http://127.0.0.1:{port}/v1/chat/completions"

    async def _start_images_server(self, handler):
        app = web.Application()
        app.router.add_post("/v1/images/generations", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        os.environ["IMAGE_GEN_API_URL"] = f"http://127.0.0.1:{port}/v1/images/generations"

    @staticmethod
    def _json_completion():
        return {
            "choices": [
                {"message": {"content": f"![generated]({_png_data_url()})"}}
            ]
        }

    async def test_streamed_sse_response_uploads_jpeg_and_preserves_png_backup(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"

        async def handler(request):
            self.requests.append(await request.json())
            content = f"![generated]({_png_data_url()})"
            split_at = len(content) // 2
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(b": relay keep-alive\n\n")
            await response.write(b"data:\n\n")
            await response.write("data: image generation started\n\n".encode())
            for piece in (content[:split_at], content[split_at:]):
                payload = {"choices": [{"delta": {"content": piece}}]}
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("流式测试", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertTrue(self.requests[0]["stream"])
        self.assertTrue(self.model.uploaded_images[0].startswith(b"\xff\xd8\xff"))
        backups = list(Path(self.output_dir.name).glob("*.png"))
        self.assertEqual(len(backups), 1)
        self.assertTrue(backups[0].read_bytes().startswith(b"\x89PNG"))

    async def test_sse_error_event_fails_without_retry(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(b"event: error\ndata: upstream failed\n\n")
            await response.write_eof()
            return response

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("错误事件", output_dir=self.output_dir.name)

        self.assertIn("API SSE 返回错误事件", result)
        self.assertEqual(attempts, 1)

    async def test_stream_only_uploads_final_image_after_finish_reason(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"
        preview = _png_data_url((220, 30, 30, 255))
        final = _png_data_url((30, 100, 220, 255))
        uploads_seen_after_preview = None

        async def handler(request):
            nonlocal uploads_seen_after_preview
            self.requests.append(await request.json())
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            preview_chunk = {
                "choices": [
                    {
                        "delta": {"content": f"![preview]({preview})"},
                        "finish_reason": None,
                    }
                ]
            }
            await response.write(f"data: {json.dumps(preview_chunk)}\n\n".encode())
            await asyncio.sleep(0.02)
            uploads_seen_after_preview = len(self.model.uploaded_images)
            final_chunk = {
                "choices": [
                    {
                        "delta": {"content": f"![final]({final})"},
                        "finish_reason": "stop",
                    }
                ]
            }
            await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            await response.write_eof()
            return response

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("最终图片", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(uploads_seen_after_preview, 0)
        self.assertEqual(len(self.model.uploaded_images), 1)
        backup = next(Path(self.output_dir.name).glob("*.png"))
        with Image.open(backup) as image:
            self.assertEqual(image.getpixel((0, 0)), (30, 100, 220, 255))

    async def test_stream_disconnect_before_final_marker_ignores_preview(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"
        os.environ["IMAGE_GEN_MAX_ATTEMPTS"] = "1"
        preview = _png_data_url((220, 30, 30, 255))

        async def handler(request):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            preview_chunk = {
                "choices": [
                    {
                        "delta": {"content": f"![preview]({preview})"},
                        "finish_reason": None,
                    }
                ]
            }
            await response.write(f"data: {json.dumps(preview_chunk)}\n\n".encode())
            await response.write_eof()
            return response

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("中断流", output_dir=self.output_dir.name)

        self.assertIn("最终完成标记前结束", result)
        self.assertEqual(self.model.uploaded_images, [])
        self.assertEqual(list(Path(self.output_dir.name).glob("*")), [])

    async def test_stream_request_accepts_json_response_and_reads_runtime_config(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"

        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._json_completion())

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        os.environ["IMAGE_GEN_MODEL"] = "runtime-model-after-import"

        result = await tool("运行时配置", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(self.requests[0]["model"], "runtime-model-after-import")
        self.assertTrue(self.requests[0]["stream"])

    async def test_non_stream_is_the_default_transport(self):
        os.environ.pop("IMAGE_GEN_STREAM", None)

        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._json_completion())

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("默认非流", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertFalse(self.requests[0]["stream"])

    async def test_native_images_endpoint_uses_right_code_contract(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"
        reference = _png_data_url()

        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response({"data": [{"url": _png_data_url()}]})

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool(
            "原生图片接口",
            aspect_ratio="3:4",
            image_size="1K",
            reference_images=[reference],
            output_dir=self.output_dir.name,
        )

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(
            self.requests[0],
            {
                "model": "test-image-model",
                "prompt": "原生图片接口",
                "image": [reference],
                "size": "1024x1360",
                "response_format": "url",
            },
        )

    async def test_native_images_endpoint_uses_fixed_1k_for_other_requested_sizes(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response({"data": [{"url": _png_data_url()}]})

        await self._start_images_server(handler)
        tool = create_image_generation_tool(self.model)
        for requested_size in ("512", "4K"):
            result = await tool(
                "固定清晰度",
                aspect_ratio="3:4",
                image_size=requested_size,
                output_dir=self.output_dir.name,
            )
            self.assertTrue(result.startswith("upload://"))

        self.assertEqual(
            [request["size"] for request in self.requests],
            ["1024x1360", "1024x1360"],
        )

    async def test_disconnect_is_not_retried_by_safe_default(self):
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            await request.read()
            if attempts == 1:
                request.transport.close()
                return web.Response()
            return web.json_response(self._json_completion())

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("断连不重提", output_dir=self.output_dir.name)

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

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        with patch(
            "examples.models.mention_model.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ) as mocked_sleep:
            result = await tool("连续断连", output_dir=self.output_dir.name)

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

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        with patch(
            "examples.models.mention_model.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ) as mocked_sleep:
            result = await tool("配置重连", output_dir=self.output_dir.name)

        self.assertTrue(result.startswith("图片生成失败: API 调用异常"))
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
            return web.json_response(self._json_completion())

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        with patch(
            "examples.models.mention_model.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ):
            result = await tool("HTTP 重试", output_dir=self.output_dir.name)

        self.assertEqual(result, "upload://mockShortPath1.jpeg")
        self.assertEqual(attempts, 2)

    async def test_stream_rejection_does_not_retry(self):
        os.environ["IMAGE_GEN_STREAM"] = "true"
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            return web.Response(status=400, text="stream unsupported")

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        result = await tool("不兼容流式", output_dir=self.output_dir.name)

        self.assertIn("stream=true", result)
        self.assertEqual(attempts, 1)

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
            return web.json_response(self._json_completion())

        await self._start_server(handler)
        tool = create_image_generation_tool(self.model)
        first = asyncio.create_task(tool("请求一", output_dir=self.output_dir.name))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = asyncio.create_task(tool("请求二", output_dir=self.output_dir.name))
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
        """已是 data URL 应直接返回"""
        result = asyncio.run(_download_and_encode(None, "data:image/png;base64,abc123"))
        self.assertEqual(result, "data:image/png;base64,abc123")

    def test_data_url_respects_single_reference_limit(self):
        result = asyncio.run(
            _download_and_encode(None, "data:image/png;base64,aGVsbG8=", max_bytes=1)
        )
        self.assertIsNone(result)


# ── URL 提取工具 ──────────────────────────────────────────────────────

class TestExtractImageURL(unittest.TestCase):
    def test_standard_markdown(self):
        self.assertEqual(_extract_image_url("![desc](https://example.com/img.png)"), "https://example.com/img.png")

    def test_upload_format(self):
        self.assertEqual(_extract_image_url("![img](upload://abc123.jpeg)"), "upload://abc123.jpeg")

    def test_data_url(self):
        self.assertEqual(_extract_image_url("![img](data:image/png;base64,abc123)"), "data:image/png;base64,abc123")

    def test_multiple_images_returns_final_url(self):
        self.assertEqual(
            _extract_image_url("![preview](https://example.com/a.png) ![final](https://example.com/b.png)"),
            "https://example.com/b.png",
        )

    def test_plain_text_returns_none(self):
        self.assertIsNone(_extract_image_url("hello world"))

    def test_none_input_returns_none(self):
        self.assertIsNone(_extract_image_url(None))


class TestImageRequestTimeout(unittest.TestCase):
    def test_stream_timeout_limits_silence_without_total_deadline(self):
        timeout = _image_request_timeout(True, 600)
        self.assertIsNone(timeout.total)
        self.assertEqual(timeout.connect, 600)
        self.assertEqual(timeout.sock_read, 600)

    def test_non_stream_timeout_retains_total_deadline(self):
        timeout = _image_request_timeout(False, 600)
        self.assertEqual(timeout.total, 600)


class TestNativeImageSize(unittest.TestCase):
    def test_edges_are_aligned_for_custom_portrait_ratio(self):
        self.assertEqual(_native_image_size("3:4"), "1024x1360")

    def test_extreme_ratio_fits_native_limits_at_max_allowed_size(self):
        width, height = map(int, _native_image_size("1:8").split("x"))
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertLessEqual(max(width, height), 3840)
        self.assertLessEqual(width * height, 8_294_400)
        self.assertLessEqual(max(width, height) / min(width, height), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

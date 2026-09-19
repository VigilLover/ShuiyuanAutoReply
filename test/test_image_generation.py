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
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
from aiohttp import web
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shuiyuan_auto_reply.features.mention.image_generation import (
    ImageGenerationService,
    _download_and_encode,
    _encode_bytes,
    _image_api_endpoint,
    _image_request_timeout,
    _openai_image_size,
)
from shuiyuan_auto_reply.infrastructure.persistence import SQLiteStateStore

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


def _decode(result: str) -> dict:
    return json.loads(result)


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
                "SHUIYUAN_STATE_DIR": self.output_dir.name,
            },
            clear=False,
        )
        self.env_patch.start()
        self.store = SQLiteStateStore(Path(self.output_dir.name) / "state.sqlite3")
        await self.store.initialize()
        self.service = ImageGenerationService(self.model, self.store)
        self.last_artifact = None

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
        return {"data": [{"url": _png_data_url()}]}

    @staticmethod
    def _images_response_b64():
        raw = base64.b64decode(_png_data_url().split(",", 1)[1])
        return {"data": [{"b64_json": base64.b64encode(raw).decode("ascii")}]}

    async def generate(self, prompt, **kwargs):
        content, artifact = await self.service.generate(prompt, **kwargs)
        self.last_artifact = artifact
        return content

    def assertGenerated(self, result: str) -> dict:
        payload = json.loads(result)
        self.assertEqual(payload["status"], "ok", payload)
        self.assertTrue(payload["artifact"].startswith("artifact://"))
        self.assertIsNotNone(self.last_artifact)
        self.assertTrue(Path(self.last_artifact.local_path).is_file())
        return payload

    def assertFailed(self, result: str, *fragments: str) -> dict:
        payload = json.loads(result)
        self.assertEqual(payload["status"], "error", payload)
        self.assertIsNone(self.last_artifact)
        for fragment in fragments:
            self.assertIn(fragment, payload["message"])
        return payload

    async def test_generation_endpoint_uses_openai_images_contract(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        result = await self.generate("测试原生图片生成接口调用", aspect_ratio="3:4")

        payload = self.assertGenerated(result)
        self.assertEqual((payload["width"], payload["height"]), (4, 4))
        self.assertEqual(
            self.requests[0],
            {
                "model": "test-image-model",
                "prompt": "测试原生图片生成接口调用",
                "size": "1024x1360",
            },
        )
        self.assertNotIn("messages", self.requests[0])
        # The forum upload happens at publish time, never inside the tool.
        self.assertEqual(self.model.uploaded_images, [])

    async def test_edit_endpoint_is_selected_when_references_exist(self):
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
                            "content_type": part.headers.get("Content-Type"),
                            "bytes": await part.read(),
                        }
                    )
                else:
                    fields[part.name] = await part.text()
            fields["image_count"] = len(images)
            fields["image_content_types"] = [image["content_type"] for image in images]
            fields["image_bytes"] = [len(image["bytes"]) for image in images]
            self.requests.append(fields)
            return web.json_response(self._images_response_b64())

        await self._start_images_server(generations_handler, edits_handler)
        result = await self.generate(
            "参考图编辑测试生图功能验证",
            aspect_ratio="3:4",
            references=[{"key": "ref", "url": reference, "label": "原参考图1"}],
        )

        self.assertGenerated(result)
        self.assertEqual(seen_paths, ["/v1/images/edits"])
        self.assertEqual(
            self.requests[0],
            {
                "model": "test-image-model",
                "prompt": (
                    "参考图编辑测试生图功能验证\n\n"
                    "【实际参考素材对应关系】\n参考图1：原参考图1"
                ),
                "size": "1024x1360",
                "image_count": 1,
                "image_content_types": ["image/png"],
                "image_bytes": [len(base64.b64decode(reference.split(",", 1)[1]))],
            },
        )

    async def test_missing_reference_blocks_generation_unless_partial_allowed(self):
        submissions = 0

        async def edits_handler(request):
            nonlocal submissions
            submissions += 1
            await request.read()
            return web.json_response(self._images_response_b64())

        await self._start_images_server(edits_handler, edits_handler)
        references = [
            {"key": "good", "url": _png_data_url(), "label": "Alice"},
            {"key": "bad", "url": "data:image/png;base64,bm90aW1hZ2U=", "label": "Bob"},
        ]
        result = await self.generate("按参考图生成两个人的合照", references=references)

        payload = json.loads(result)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual([item["key"] for item in payload["failed"]], ["bad"])
        self.assertEqual(payload["loaded"], ["good"])
        self.assertEqual(submissions, 0)

        result = await self.generate(
            "按参考图生成两个人的合照", references=references, allow_partial=True
        )
        self.assertGenerated(result)
        self.assertEqual(submissions, 1)

    async def test_url_response_is_downloaded_and_stored(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_url())

        await self._start_images_server(handler)
        result = await self.generate("测试URL响应图片的下载和保存")

        self.assertGenerated(result)

    async def test_non_200_response_code_and_body_are_returned_to_tool_caller(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(
                {"error": {"message": "bad image prompt"}},
                status=400,
            )

        await self._start_images_server(handler)
        result = await self.generate("测试服务端错误信息回传给机器人")

        self.assertFailed(result, "API 返回 HTTP 400", "bad image prompt")
        self.assertEqual(len(self.requests), 1)

    async def test_4router_request_id_is_returned_with_http_error(self):
        async def handler(request):
            return web.json_response(
                {"error": {"message": "control plane unavailable"}},
                status=502,
                headers={"x-oneapi-request-id": "request-abc123"},
            )

        await self._start_images_server(handler)
        result = await self.generate("测试图片代理请求编号错误回传")

        self.assertFailed(result, "HTTP 502", "4Router request_id=request-abc123")

    async def test_numeric_prompt_values_are_rejected_before_server_request(self):
        async def handler(request):
            self.requests.append(await request.read())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        for prompt in ("0", "1", 0, 1):
            payload = json.loads(await self.generate(prompt))
            self.assertEqual(payload["code"], "invalid_prompt")

        self.assertEqual(self.requests, [])

    async def test_repeated_valid_prompt_still_submits_real_requests(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        prompt = "重复有效提示词也必须真实请求服务器"

        first = self.assertGenerated(await self.generate(prompt))
        second = self.assertGenerated(await self.generate(prompt))

        self.assertNotEqual(first["artifact"], second["artifact"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            [request["prompt"] for request in self.requests], [prompt, prompt]
        )

    async def test_runtime_model_config_is_read_for_each_call(self):
        async def handler(request):
            self.requests.append(await request.json())
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        os.environ["IMAGE_GEN_MODEL"] = "runtime-model-after-import"

        self.assertGenerated(await self.generate("测试运行时模型配置读取功能"))
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
        result = await self.generate("测试断连不重提独立请求")

        self.assertFailed(result, "未提供断线续取能力")
        self.assertEqual(attempts, 1)

    async def test_explicit_multiple_attempts_repeat_submissions_with_exponential_backoff(
        self,
    ):
        os.environ["IMAGE_GEN_MAX_ATTEMPTS"] = "4"
        attempts = 0

        async def handler(request):
            nonlocal attempts
            attempts += 1
            await request.read()
            request.transport.close()
            return web.Response()

        await self._start_images_server(handler)
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ) as mocked_sleep:
            result = await self.generate("测试连续断连重试提交功能")

        self.assertFailed(result, "已执行的重试均为独立请求")
        self.assertEqual(attempts, 4)
        self.assertEqual(
            mocked_sleep.await_args_list,
            [call(5.0), call(10.0), call(20.0)],
        )

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
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ) as mocked_sleep:
            result = await self.generate("测试配置重连次数和延迟参数")

        self.assertFailed(result, "API 连接异常", "无法接收该次 response")
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
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation.asyncio.sleep",
            new=AsyncMock(),
        ):
            result = await self.generate("测试HTTP状态码重试处理逻辑")

        self.assertGenerated(result)
        self.assertEqual(attempts, 2)

    async def test_image_generation_concurrency_follows_runtime_config(self):
        active_requests = 0
        maximum_active_requests = 0
        submissions = 0
        started = asyncio.Semaphore(0)
        release = asyncio.Event()

        async def handler(request):
            nonlocal active_requests, maximum_active_requests, submissions
            submissions += 1
            active_requests += 1
            maximum_active_requests = max(maximum_active_requests, active_requests)
            started.release()
            await release.wait()
            active_requests -= 1
            return web.json_response(self._images_response_b64())

        await self._start_images_server(handler)
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

        limit = int(get_deployment().section("runtime")["image_concurrency"])
        tasks = [
            asyncio.create_task(self.service.generate(f"并发生图测试请求任务 {i}"))
            for i in range(limit + 1)
        ]
        for _ in range(limit):
            await asyncio.wait_for(started.acquire(), timeout=2)
        await asyncio.sleep(0.05)

        self.assertEqual(submissions, limit)
        release.set()
        results = await asyncio.gather(*tasks)

        self.assertEqual(maximum_active_requests, limit)
        self.assertEqual(submissions, limit + 1)
        self.assertTrue(
            all(json.loads(content)["status"] == "ok" for content, _ in results)
        )


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
            data_url = await _download_and_encode(
                session, "https://www.python.org/static/img/python-logo.png"
            )
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

        result = _encode_bytes(
            buffer.getvalue(), "reference.png", max_bytes=1024 * 1024
        )

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

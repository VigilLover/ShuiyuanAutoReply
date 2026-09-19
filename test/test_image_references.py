import base64
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.image_generation import (
    ImageGenerationService,
    _encode_bytes,
)
from shuiyuan_auto_reply.features.mention.image_references import prepare_references
from shuiyuan_auto_reply.infrastructure.image_transport import (
    ImageDownloadError,
    encoded_image_url,
)


def data_image(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class ReferencePreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.turn = TurnResults()
        self.token = current_turn.set(self.turn)

    async def asyncTearDown(self):
        current_turn.reset(self.token)

    def test_url_encodes_unicode_without_reencoding_signature(self):
        url = "https://shuiyuan.sjtu.edu.cn/user_avatar/中文/288/x.png?signature=a%2Fb%2B&x=1+2"
        actual = str(encoded_image_url(url))
        self.assertIn("/%E4%B8%AD%E6%96%87/", actual)
        self.assertTrue(actual.endswith("?signature=a%2Fb%2B&x=1+2"))
        self.assertEqual(str(encoded_image_url(actual)), actual)
        self.assertIsNone(_encode_bytes(b"<html>not an image</html>", "a.png", 1024))

    async def test_partial_failure_keeps_actual_order_and_reuses_success(self):
        refs = [
            {"key": "a", "url": "https://example.org/a", "label": "A"},
            {"key": "b", "url": "https://example.org/b", "label": "B"},
            {"key": "c", "url": "https://example.org/c", "label": "C"},
        ]

        async def download(session, url, **kwargs):
            if url.endswith("/b"):
                raise ImageDownloadError(404)
            return data_image("red" if url.endswith("/a") else "blue")

        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode",
            side_effect=download,
        ) as mock:
            result = await prepare_references(refs, model=SimpleNamespace())
            self.assertEqual(result["status"], "partial")
            self.assertEqual([i.get("index") for i in result["items"]], [1, None, 2])
            self.assertEqual(
                result["data_urls"], [data_image("red"), data_image("blue")]
            )
            await prepare_references(refs, model=SimpleNamespace())
            self.assertEqual(
                mock.await_count, 3
            )  # deterministic missing image is cached

    async def test_retry_transient_only(self):
        with (
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode",
                side_effect=[ImageDownloadError(503), data_image()],
            ) as download,
            patch(
                "shuiyuan_auto_reply.features.mention.image_references.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            result = await prepare_references(
                [{"key": "a", "url": "https://example.org/a"}], model=SimpleNamespace()
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(download.await_count, 2)
            sleep.assert_awaited_once_with(1.0)

    async def test_404_refreshes_only_known_avatar(self):
        old = "https://example.org/old"
        self.turn.cache["user:alice"] = {"username": "alice", "avatar": old}
        model = SimpleNamespace(
            get_user_by_username=AsyncMock(
                return_value=SimpleNamespace(
                    id=2,
                    username="alice",
                    name=None,
                    avatar_template="https://example.org/new",
                )
            )
        )
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode",
            side_effect=[ImageDownloadError(404), data_image()],
        ) as mock:
            result = await prepare_references([{"key": "a", "url": old}], model=model)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(mock.await_args.args[1], "https://example.org/new")
            model.get_user_by_username.assert_awaited_once_with("alice")

    async def test_all_failures_never_generate(self):
        store = SimpleNamespace(model_config_resolver=None)
        service = ImageGenerationService(SimpleNamespace(), store)
        with (
            patch.dict(
                os.environ,
                {
                    "IMAGE_GEN_API_KEY": "test",
                    "IMAGE_GEN_API_URL": "https://example.org/v1",
                },
            ),
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode",
                return_value=None,
            ),
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._submit_image_request",
                new_callable=AsyncMock,
            ) as submit,
        ):
            content, artifact = await service.generate(
                "A sufficiently detailed prompt",
                references=[{"key": "bad", "url": "https://example.org/bad"}],
                allow_partial=True,
            )
        payload = json.loads(content)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["code"], "reference_failed")
        self.assertEqual([item["key"] for item in payload["failed"]], ["bad"])
        self.assertIsNone(artifact)
        submit.assert_not_awaited()

    async def test_generation_uses_prepared_order_without_exposing_missing_subject(
        self,
    ):
        good = [data_image("red"), data_image("blue")]
        captured = {}

        async def submit(url, api_key, form, **kwargs):
            captured["prompt"] = next(
                value
                for options, _, value in form._fields
                if options["name"] == "prompt"
            )
            captured["images"] = [
                value
                for options, _, value in form._fields
                if options["name"] == "image[]"
            ]
            return base64.b64decode(good[0].split(",", 1)[1])

        async def download(session, url, **kwargs):
            if url.endswith("/b"):
                raise ImageDownloadError(404)
            return good[0] if url.endswith("/a") else good[1]

        store = SimpleNamespace(
            model_config_resolver=None, register_artifact=AsyncMock()
        )
        service = ImageGenerationService(SimpleNamespace(), store)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "IMAGE_GEN_API_KEY": "test",
                    "IMAGE_GEN_API_URL": "https://example.org/v1",
                    "SHUIYUAN_STATE_DIR": directory,
                },
            ),
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._request_image_bytes_multipart",
                side_effect=submit,
            ),
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode",
                side_effect=download,
            ),
        ):
            content, artifact = await service.generate(
                "Draw Alice and Carol using their references",
                references=[
                    {"key": "a", "url": "https://example.org/a", "label": "Alice"},
                    {"key": "b", "url": "https://example.org/b", "label": "Bob"},
                    {"key": "c", "url": "https://example.org/c", "label": "Carol"},
                ],
                allow_partial=True,
            )
        self.assertEqual(json.loads(content)["status"], "ok")
        self.assertIsNotNone(artifact)
        self.assertEqual(
            captured["images"], [base64.b64decode(x.split(",", 1)[1]) for x in good]
        )
        self.assertIn("参考图1：Alice", captured["prompt"])
        self.assertIn("参考图2：Carol", captured["prompt"])
        self.assertIn("未提供的素材（Bob）", captured["prompt"])
        self.assertNotIn("参考图3", captured["prompt"])

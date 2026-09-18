import base64
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.image_generation import (
    _encode_bytes,
    create_image_generation_tool,
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

    async def test_all_failures_never_generate_and_legacy_partial_returns_set(self):
        model = SimpleNamespace()
        tool = create_image_generation_tool(model)
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
                side_effect=[None, data_image()],
            ),
        ):
            result = await tool(
                "A sufficiently detailed prompt",
                reference_images=[
                    "https://example.org/bad",
                    "https://example.org/good",
                ],
            )
            self.assertIn("尚未生成", result)
            self.assertIn("reference_set_id", result)
        with patch(
            "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode",
            return_value=None,
        ):
            result = await prepare_references(
                [{"key": "bad", "url": "https://example.org/bad"}], model=model
            )
        with patch.dict(
            os.environ,
            {
                "IMAGE_GEN_API_KEY": "test",
                "IMAGE_GEN_API_URL": "https://example.org/v1",
            },
        ):
            output = await tool(
                "A sufficiently detailed prompt",
                reference_set_id=result["reference_set_id"],
            )
            self.assertIn("未能读取", output)

    async def test_generation_uses_prepared_order_without_exposing_missing_subject(
        self,
    ):
        good = [data_image("red"), data_image("blue")]
        self.turn.references["set"] = {
            "status": "partial",
            "data_urls": good,
            "items": [
                {"key": "a", "label": "Alice", "status": "ok"},
                {"key": "b", "label": "Bob", "status": "error"},
                {"key": "c", "label": "Carol", "status": "ok"},
            ],
        }
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

        model = SimpleNamespace(
            upload_image=AsyncMock(
                return_value=SimpleNamespace(short_path="upload://result.jpeg")
            )
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "IMAGE_GEN_API_KEY": "test",
                    "IMAGE_GEN_API_URL": "https://example.org/v1",
                },
            ),
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._request_image_bytes_multipart",
                side_effect=submit,
            ),
            patch(
                "shuiyuan_auto_reply.features.mention.image_generation._download_and_encode"
            ) as download,
        ):
            result = await create_image_generation_tool(model)(
                "Draw Alice and Carol using their references",
                reference_set_id="set",
                output_dir=directory,
            )
        download.assert_not_called()
        self.assertEqual(
            captured["images"], [base64.b64decode(x.split(",", 1)[1]) for x in good]
        )
        self.assertIn("参考图1：Alice", captured["prompt"])
        self.assertIn("参考图2：Carol", captured["prompt"])
        self.assertIn("未提供的素材（Bob）", captured["prompt"])
        self.assertEqual(result, "upload://result.jpeg")
        self.assertEqual(self.turn.notices, [])

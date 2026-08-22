import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from shuiyuan_auto_reply.domain import GeneratedImageArtifact, VisualMediaArtifact
from shuiyuan_auto_reply.infrastructure.forum.media import (
    ForumMediaUpload,
    ForumMediaUploader,
    ForumReplyMediaPublisher,
)
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel


def image_bytes(image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (4, 3), (92, 134, 232, 160)).save(
        output, format=image_format
    )
    return output.getvalue()


class ShuiyuanImageUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_response_short_urls_are_canonicalized(self):
        response = SimpleNamespace(
            status=200,
            json=AsyncMock(
                return_value={
                    "id": 42,
                    "url": "https://shuiyuan.sjtu.edu.cn/uploads/default/original/asset.png",
                    "original_filename": "asset.png",
                    "short_url": "/uploads/short-url/shortUrlToken.png",
                    "short_path": "/uploads/short-url/shortPathToken.png",
                }
            ),
        )
        model = ShuiyuanModel()
        model._rate_limited_request = AsyncMock(return_value=response)

        result = await model.upload_image(
            image_bytes(), mime_type="image/png", filename="asset.png"
        )

        self.assertEqual(result.short_url, "upload://shortUrlToken.png")
        self.assertEqual(result.short_path, "upload://shortPathToken.png")


class ForumMediaUploaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_format_is_uploaded_first(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asset.png"
            data = image_bytes()
            path.write_bytes(data)
            artifact = VisualMediaArtifact(
                "asset-1",
                "image/png",
                str(path),
                len(data),
                "web_search",
                filename="result.png",
            )
            forum = SimpleNamespace(
                upload_image=AsyncMock(
                    return_value=SimpleNamespace(
                        short_path="/uploads/short-url/uB9mAVdAgjShS5HHmLgflguOW9F.jpeg"
                    )
                )
            )

            result = await ForumMediaUploader(forum).upload(artifact)

            self.assertEqual(
                result.short_path,
                "upload://uB9mAVdAgjShS5HHmLgflguOW9F.jpeg",
            )
            forum.upload_image.assert_awaited_once_with(
                data, mime_type="image/png", filename="result.png"
            )

    async def test_failed_original_format_retries_once_as_jpeg(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asset.png"
            data = image_bytes()
            path.write_bytes(data)
            artifact = VisualMediaArtifact(
                "asset-2", "image/png", str(path), len(data), "web_search"
            )
            forum = SimpleNamespace(
                upload_image=AsyncMock(
                    side_effect=[
                        RuntimeError("format rejected"),
                        SimpleNamespace(short_path="upload://fallback.jpg"),
                    ]
                )
            )

            result = await ForumMediaUploader(forum).upload(artifact)

            self.assertEqual(result.short_path, "upload://fallback.jpg")
            self.assertEqual(forum.upload_image.await_count, 2)
            fallback_call = forum.upload_image.await_args_list[1]
            self.assertEqual(fallback_call.kwargs["mime_type"], "image/jpeg")
            self.assertTrue(fallback_call.kwargs["filename"].endswith(".jpg"))
            with Image.open(io.BytesIO(fallback_call.args[0])) as fallback:
                self.assertEqual(fallback.format, "JPEG")

    async def test_cached_forum_path_skips_upload(self):
        artifact = GeneratedImageArtifact(
            "asset-3", "image/png", "/not/read/when/cached.png", 10
        )
        store = SimpleNamespace(
            get_artifact=AsyncMock(
                return_value=SimpleNamespace(
                    forum_short_path="upload://cached.png"
                )
            )
        )
        forum = SimpleNamespace(upload_image=AsyncMock())

        result = await ForumMediaUploader(forum, store).upload(artifact)

        self.assertTrue(result.reused)
        self.assertEqual(result.short_path, "upload://cached.png")
        forum.upload_image.assert_not_awaited()

    async def test_invalid_cached_path_is_ignored_and_reuploaded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "asset.png"
            data = image_bytes()
            path.write_bytes(data)
            artifact = VisualMediaArtifact(
                "asset-invalid-cache", "image/png", str(path), len(data), "web_search"
            )
            store = SimpleNamespace(
                get_artifact=AsyncMock(
                    return_value=SimpleNamespace(
                        forum_short_path="https://cdn.example/not-local.png"
                    )
                ),
                set_forum_short_path=AsyncMock(),
            )
            forum = SimpleNamespace(
                upload_image=AsyncMock(
                    return_value=SimpleNamespace(short_path="upload://local.png")
                )
            )

            result = await ForumMediaUploader(forum, store).upload(artifact)

            self.assertEqual(result.short_path, "upload://local.png")
            self.assertFalse(result.reused)
            forum.upload_image.assert_awaited_once()


class ForumReplyMediaPublisherTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def artifact(
        artifact_id: str,
        source_url: str | None,
        source_kind: str = "web_search",
    ) -> VisualMediaArtifact:
        return VisualMediaArtifact(
            artifact_id,
            "image/png",
            f"/tmp/{artifact_id}.png",
            10,
            source_kind,
            source_url=source_url,
        )

    async def test_only_selected_markdown_image_is_uploaded_and_source_removed(self):
        selected = self.artifact("selected", "https://cdn.example/selected.png")
        unused = self.artifact("unused", "https://cdn.example/unused.png")
        uploader = SimpleNamespace(
            upload=AsyncMock(
                return_value=ForumMediaUpload("upload://selected.png")
            )
        )
        text = (
            "![结果](https://cdn.example/selected.png)\n"
            "[原图](https://cdn.example/selected.png)\n"
            "普通链接：[未选择](https://cdn.example/unused.png)"
        )

        result = await ForumReplyMediaPublisher(uploader).publish(
            text, [selected, unused]
        )

        uploader.upload.assert_awaited_once_with(selected)
        self.assertIn("![结果](upload://selected.png)", result.text)
        self.assertNotIn("https://cdn.example/selected.png", result.text)
        self.assertIn("https://cdn.example/unused.png", result.text)
        self.assertEqual(len(result.published), 1)

    async def test_uploaded_short_url_is_canonicalized_in_final_markdown(self):
        selected = self.artifact("short-url", "https://cdn.example/selected.jpeg")
        short_url = "/uploads/short-url/uB9mAVdAgjShS5HHmLgflguOW9F.jpeg"
        uploader = SimpleNamespace(
            upload=AsyncMock(return_value=ForumMediaUpload(short_url))
        )

        result = await ForumReplyMediaPublisher(uploader).publish(
            "![专辑封面](https://cdn.example/selected.jpeg)", [selected]
        )

        expected = "upload://uB9mAVdAgjShS5HHmLgflguOW9F.jpeg"
        self.assertEqual(result.text, f"![专辑封面]({expected})")
        self.assertEqual(result.published[0].short_path, expected)

    async def test_plain_source_link_does_not_select_an_image(self):
        artifact = self.artifact("plain", "https://cdn.example/plain.png")
        uploader = SimpleNamespace(upload=AsyncMock())
        text = "[普通来源](https://cdn.example/plain.png)"

        result = await ForumReplyMediaPublisher(uploader).publish(text, [artifact])

        self.assertEqual(result.text, text)
        self.assertEqual(result.published, ())
        uploader.upload.assert_not_awaited()

    async def test_uncached_external_image_is_removed_not_published_as_link(self):
        uploader = SimpleNamespace(upload=AsyncMock())
        text = (
            "![未缓存](https://cdn.example/blocked.webp)\n"
            '<img src="https://cdn.example/blocked-2.webp">'
        )

        result = await ForumReplyMediaPublisher(uploader).publish(text, [])

        self.assertEqual(
            result.text,
            "（图片本地化或上传失败）\n"
            "（图片本地化或上传失败）",
        )
        self.assertNotIn("https://", result.text)

    async def test_artifact_marker_and_html_image_are_rewritten(self):
        marker = self.artifact("marker", "https://cdn.example/marker.png")
        html = self.artifact("html", "https://cdn.example/html.png")
        uploader = SimpleNamespace(
            upload=AsyncMock(
                side_effect=[
                    ForumMediaUpload("upload://marker.png"),
                    ForumMediaUpload("upload://html.png"),
                ]
            )
        )
        text = (
            f"![标识]({marker.uri})\n"
            '<img alt="HTML 图片说明" src="https://cdn.example/html.png">'
        )

        result = await ForumReplyMediaPublisher(uploader).publish(
            text, [marker, html]
        )

        self.assertIn("![标识](upload://marker.png)", result.text)
        self.assertIn("![HTML 图片说明](upload://html.png)", result.text)
        self.assertNotIn("https://cdn.example/", result.text)

    async def test_escaped_markdown_is_canonicalized_without_backslashes(self):
        artifact = self.artifact("escaped", "https://cdn.example/escaped.jpeg")
        uploader = SimpleNamespace(
            upload=AsyncMock(
                return_value=ForumMediaUpload("upload://token123.jpeg")
            )
        )

        result = await ForumReplyMediaPublisher(uploader).publish(
            r"![图片说明]\(https://cdn.example/escaped\.jpeg)", [artifact]
        )

        self.assertEqual(result.text, "![图片说明](upload://token123.jpeg)")
        self.assertNotIn("\\", result.text)

    async def test_existing_forum_upload_is_reused_without_upload(self):
        artifact = self.artifact(
            "forum", "upload://existing.png", "forum_search"
        )
        uploader = SimpleNamespace(upload=AsyncMock())
        store = SimpleNamespace(set_forum_short_path=AsyncMock())

        result = await ForumReplyMediaPublisher(uploader, store).publish(
            f"![论坛图片]({artifact.uri})", [artifact]
        )

        self.assertEqual(result.text, "![论坛图片](upload://existing.png)")
        self.assertTrue(result.published[0].reused)
        uploader.upload.assert_not_awaited()
        store.set_forum_short_path.assert_awaited_once_with(
            "forum", "upload://existing.png"
        )

    async def test_upload_failure_degrades_to_source_link(self):
        artifact = self.artifact("failed", "https://cdn.example/failed.png")
        uploader = SimpleNamespace(
            upload=AsyncMock(side_effect=RuntimeError("upload failed"))
        )

        result = await ForumReplyMediaPublisher(uploader).publish(
            "![结果图](https://cdn.example/failed.png) 后续正文", [artifact]
        )

        self.assertEqual(
            result.text,
            "（图片上传失败） 后续正文",
        )
        self.assertNotIn("https://", result.text)
        self.assertEqual(len(result.failures), 1)

    async def test_inline_upload_failure_has_text_fallback(self):
        artifact = self.artifact("inline", None)
        uploader = SimpleNamespace(
            upload=AsyncMock(side_effect=RuntimeError("upload failed"))
        )

        result = await ForumReplyMediaPublisher(uploader).publish(
            f"![内嵌图]({artifact.uri})", [artifact]
        )

        self.assertEqual(result.text, "（图片上传失败）")

    async def test_grid_with_more_than_four_images_is_not_truncated(self):
        artifacts = [
            self.artifact(str(index), f"https://cdn.example/{index}.png")
            for index in range(5)
        ]
        uploader = SimpleNamespace(
            upload=AsyncMock(
                side_effect=[
                    ForumMediaUpload(f"upload://{index}.png")
                    for index in range(5)
                ]
            )
        )
        text = "[grid]" + "".join(
            f"![图{index}]({artifact.uri})"
            for index, artifact in enumerate(artifacts)
        ) + "[/grid]"

        result = await ForumReplyMediaPublisher(uploader).publish(text, artifacts)

        self.assertTrue(result.text.startswith("[grid]"))
        self.assertTrue(result.text.endswith("[/grid]"))
        self.assertEqual(len(result.published), 5)
        self.assertEqual(uploader.upload.await_count, 5)

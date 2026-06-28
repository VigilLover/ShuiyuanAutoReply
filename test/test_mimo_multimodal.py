import base64
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shuiyuan_auto_reply.shuiyuan.objects import PostDetails, User

from examples.models.mention_model.mention_chat_model import MentionChatModel
from examples.models.mention_model.mention_multimodal import (
    MentionImageInput,
    build_mimo_content,
    collect_post_image_inputs,
    extract_image_urls,
    normalize_shuiyuan_image_url,
    prepare_image_input,
)
from examples.models.mention_model.shuiyuan_tools_objects import PostShort


def _post_details(*, raw: str | None = "hello", cooked: str = "<p>hello</p>") -> PostDetails:
    return PostDetails(
        id=1,
        name="Tester",
        user_id=42,
        username="tester",
        user_cakedate=None,
        created_at="2026-01-01T00:00:00Z",
        cooked=cooked,
        raw=raw,
        post_number=7,
        post_type=1,
        updated_at="2026-01-01T00:00:00Z",
        reply_count=0,
        reply_to_post_number=None,
        reply_to_user=None,
        polls=None,
        yours=False,
        topic_id=99,
        can_edit=False,
        can_delete=False,
        can_recover=False,
        can_wiki=False,
        can_retort=False,
        can_remove_retort=False,
        can_accept_answer=False,
        can_unaccept_answer=False,
        can_see_hidden_post=False,
        can_view_edit_history=False,
    )


def _tiny_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (32, 96, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestMentionMultimodalExtraction(unittest.TestCase):
    def test_extracts_markdown_html_upload_and_short_path_images(self):
        text = """
        ![one](upload://abc.jpeg)
        <img alt="two" src="/uploads/short-url/def.png">
        <img src="https://shuiyuan.sjtu.edu.cn/uploads/short-url/ghi.webp">
        ![duplicate](upload://abc.jpeg)
        """

        self.assertEqual(
            extract_image_urls(text),
            [
                "upload://abc.jpeg",
                "upload://def.png",
                "upload://ghi.webp",
            ],
        )

    def test_extract_ignores_non_image_links(self):
        text = "[plain](https://example.com/page) <a href='upload://not-image.txt'>x</a>"

        self.assertEqual(extract_image_urls(text), [])


class TestMentionMultimodalEncoding(unittest.IsolatedAsyncioTestCase):
    async def test_upload_image_is_downloaded_with_shuiyuan_model_and_encoded(self):
        model = MagicMock()
        model.download_image = AsyncMock(return_value=_tiny_png_bytes())

        image = await prepare_image_input(
            "upload://abc.jpeg",
            shuiyuan_model=model,
            origin="current_post",
        )

        self.assertIsInstance(image, MentionImageInput)
        self.assertEqual(image.source_url, "upload://abc.jpeg")
        self.assertEqual(image.origin, "current_post")
        self.assertTrue(image.data_url.startswith("data:image/"))
        self.assertIn(";base64,", image.data_url)
        model.download_image.assert_awaited_once_with("upload://abc.jpeg")

    async def test_external_shuiyuan_url_is_normalized_then_downloaded(self):
        model = MagicMock()
        model.download_image = AsyncMock(return_value=_tiny_png_bytes())

        image = await prepare_image_input(
            "https://shuiyuan.sjtu.edu.cn/uploads/short-url/abc.jpeg",
            shuiyuan_model=model,
            origin="search_result",
        )

        self.assertEqual(image.source_url, "upload://abc.jpeg")
        model.download_image.assert_awaited_once_with("upload://abc.jpeg")

    async def test_non_shuiyuan_http_url_is_skipped(self):
        model = MagicMock()
        model.download_image = AsyncMock()

        image = await prepare_image_input(
            "https://example.com/private.png",
            shuiyuan_model=model,
            origin="search_result",
        )

        self.assertIsNone(image)
        model.download_image.assert_not_called()

    async def test_shuiyuan_avatar_url_is_downloaded_with_authenticated_session(self):
        model = MagicMock()
        model.download_image = AsyncMock()
        model.download_raw_image = AsyncMock(return_value=_tiny_png_bytes())
        avatar_url = "https://shuiyuan.sjtu.edu.cn/user_avatar/shuiyuan.sjtu.edu.cn/alice/288/123.png"

        image = await prepare_image_input(
            avatar_url,
            shuiyuan_model=model,
            origin="inspect_image",
        )

        self.assertIsInstance(image, MentionImageInput)
        self.assertEqual(image.source_url, "/user_avatar/shuiyuan.sjtu.edu.cn/alice/288/123.png")
        model.download_raw_image.assert_awaited_once_with(
            "/user_avatar/shuiyuan.sjtu.edu.cn/alice/288/123.png"
        )
        model.download_image.assert_not_called()

    def test_normalize_accepts_shuiyuan_avatar_paths(self):
        self.assertEqual(
            normalize_shuiyuan_image_url("/user_avatar/shuiyuan.sjtu.edu.cn/alice/288/123.png"),
            "/user_avatar/shuiyuan.sjtu.edu.cn/alice/288/123.png",
        )

    async def test_collect_post_image_inputs_respects_caps_and_deduplication(self):
        model = MagicMock()
        model.download_image = AsyncMock(return_value=_tiny_png_bytes())
        posts = [
            MagicMock(image_urls=["upload://a.jpeg", "upload://b.jpeg"]),
            MagicMock(image_urls=["upload://a.jpeg", "upload://c.jpeg"]),
        ]

        images = await collect_post_image_inputs(
            posts,
            shuiyuan_model=model,
            origin="search_result",
            max_images=2,
        )

        self.assertEqual([image.source_url for image in images], ["upload://a.jpeg", "upload://b.jpeg"])
        self.assertEqual(model.download_image.await_count, 2)

    async def test_collect_post_image_inputs_seeds_deduplication_from_existing_urls(self):
        model = MagicMock()
        model.download_image = AsyncMock(return_value=_tiny_png_bytes())
        posts = [MagicMock(image_urls=["upload://a.jpeg", "upload://b.jpeg"])]

        images = await collect_post_image_inputs(
            posts,
            shuiyuan_model=model,
            origin="search_result",
            max_images=2,
            existing_urls=["upload://a.jpeg"],
        )

        self.assertEqual([image.source_url for image in images], ["upload://b.jpeg"])
        model.download_image.assert_awaited_once_with("upload://b.jpeg")

    async def test_collect_post_image_inputs_counts_existing_bytes_against_total_cap(self):
        model = MagicMock()
        model.download_image = AsyncMock(return_value=_tiny_png_bytes())
        posts = [MagicMock(image_urls=["upload://a.jpeg"])]

        images = await collect_post_image_inputs(
            posts,
            shuiyuan_model=model,
            origin="search_result",
            max_images=1,
            max_total_bytes=10,
            existing_byte_count=10,
        )

        self.assertEqual(images, [])
        model.download_image.assert_not_called()

    def test_build_mimo_content_places_images_before_text(self):
        content = build_mimo_content(
            "看一下这些图",
            [
                MentionImageInput(
                    source_url="upload://a.jpeg",
                    data_url="data:image/jpeg;base64," + base64.b64encode(b"abc").decode("ascii"),
                    origin="current_post",
                    mime_type="image/jpeg",
                    byte_count=3,
                )
            ],
        )

        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1], {"type": "text", "text": "看一下这些图"})


class TestPostShortImages(unittest.TestCase):
    def test_image_urls_are_extracted_from_full_raw_and_cooked_before_truncation(self):
        raw = "r" * 430 + " ![late](upload://raw-late.jpeg) ![dup](upload://same.png)"
        cooked = "c" * 430 + '<img src="/uploads/short-url/cooked-late.webp"><img src="upload://same.png">'

        post = PostShort(_post_details(raw=raw, cooked=cooked), "Topic")

        self.assertEqual(
            post.image_urls,
            [
                "upload://raw-late.jpeg",
                "upload://same.png",
                "upload://cooked-late.webp",
            ],
        )
        self.assertLessEqual(len(post.raw), 384)
        self.assertLessEqual(len(post.cooked), 384)
        self.assertIn("PostMeta:", str(post))
        self.assertIn("Images: upload://raw-late.jpeg, upload://same.png, upload://cooked-late.webp", str(post))


class TestMentionMimoModel(unittest.TestCase):
    def test_requires_mimo_api_key(self):
        from examples.models.mention_model.mention_mimo_model import MentionMimoModel

        with patch.dict(os.environ, {}, clear=True), \
             patch("examples.models.mention_model.mention_chat_model.get_global_text_embeddings", return_value=MagicMock()), \
             patch("examples.models.mention_model.mention_chat_model.MentionMemoryModel"):
            with self.assertRaisesRegex(ValueError, "MIMO_API_KEY"):
                MentionMimoModel(MagicMock())

    def test_defaults_to_mimo_v25_multimodal_and_thinking_enabled(self):
        from examples.models.mention_model.mention_mimo_model import (
            MIMO_BASE_URL,
            MIMO_DEFAULT_MODEL,
            MentionMimoModel,
            _mk_mimo_llm,
        )

        with patch.dict(os.environ, {"MIMO_API_KEY": "test-key"}, clear=True), \
             patch("examples.models.mention_model.mention_chat_model.get_global_text_embeddings", return_value=MagicMock()), \
             patch("examples.models.mention_model.mention_chat_model.MentionMemoryModel"):
            model = MentionMimoModel(MagicMock())

        self.assertEqual(MIMO_DEFAULT_MODEL, "mimo-v2.5")
        self.assertTrue(model.supports_multimodal)
        self.assertEqual(model.multimodal_search_image_limit, 2)

        llm = _mk_mimo_llm("test-key", MIMO_DEFAULT_MODEL)
        payload = llm._get_request_payload([HumanMessage(content="hi")])
        self.assertEqual(payload["model"], "mimo-v2.5")
        self.assertEqual(str(payload["base_url"]).rstrip("/") if "base_url" in payload else MIMO_BASE_URL, MIMO_BASE_URL)
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("max_tokens", payload)


class TestMentionChatModelMultimodal(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_messages_uses_plain_string_when_multimodal_disabled(self):
        state = {"conversation": "hello", "supports_multimodal": False, "image_inputs": []}

        result = await MentionChatModel._prepare_messages(state)

        self.assertIsInstance(result["messages"][0].content, str)
        self.assertIn("hello", result["messages"][0].content)

    async def test_prepare_messages_uses_content_blocks_when_multimodal_enabled(self):
        image = MentionImageInput(
            source_url="upload://a.jpeg",
            data_url="data:image/jpeg;base64," + base64.b64encode(b"abc").decode("ascii"),
            origin="current_post",
            mime_type="image/jpeg",
            byte_count=3,
        )
        state = {
            "conversation": "hello",
            "supports_multimodal": True,
            "image_inputs": [image],
        }

        result = await MentionChatModel._prepare_messages(state)

        content = result["messages"][0].content
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1]["type"], "text")

    async def test_collect_tool_output_images_ignores_post_artifacts_without_inspect_image(self):
        model = MentionChatModel.__new__(MentionChatModel)
        model.model = MagicMock()
        model.model.download_image = AsyncMock(return_value=_tiny_png_bytes())
        model.multimodal_search_image_limit = 1

        artifact = [
            MagicMock(image_urls=["upload://a.jpeg", "upload://b.jpeg"]),
        ]
        state = {
            "supports_multimodal": True,
            "image_inputs": [],
            "messages": [
                ToolMessage(content="posts", tool_call_id="call-1", artifact=artifact),
            ],
        }

        result = await MentionChatModel._collect_tool_output_images(model, state)

        self.assertEqual(result["image_inputs"], [])
        self.assertNotIn("messages", result)
        model.model.download_image.assert_not_called()

    async def test_collect_tool_output_images_uses_inspect_image_artifact(self):
        model = MentionChatModel.__new__(MentionChatModel)
        model.model = MagicMock()
        model.model.download_image = AsyncMock(return_value=_tiny_png_bytes())
        model.multimodal_search_image_limit = 1

        artifact = MagicMock(image_urls=["upload://a.jpeg"], source="inspect_image", description="")
        state = {
            "supports_multimodal": True,
            "image_inputs": [],
            "messages": [
                ToolMessage(content="image inspected", tool_call_id="call-1", artifact=artifact),
            ],
        }

        result = await MentionChatModel._collect_tool_output_images(model, state)

        self.assertEqual([image.source_url for image in result["image_inputs"]], ["upload://a.jpeg"])
        content = result["messages"][0].content
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1]["type"], "text")
        self.assertIn("inspect_image", content[1]["text"])
        model.model.download_image.assert_awaited_once_with("upload://a.jpeg")

    def test_inspect_image_is_artifact_tool_and_post_tools_are_plain_content(self):
        model = MentionChatModel.__new__(MentionChatModel)
        model.model = MagicMock()
        model.supports_multimodal = True

        tools = MentionChatModel._load_shuiyuan_tools(model)
        by_name = {tool.name: tool for tool in tools}

        for name in ["search_posts", "recent_posts", "search_posts_by_time", "get_post"]:
            self.assertEqual(by_name[name].response_format, "content")
        self.assertEqual(by_name["inspect_image"].response_format, "content_and_artifact")
        self.assertEqual(by_name["generate_image"].response_format, "content")

    async def test_inspect_image_tool_returns_artifact_for_requested_url(self):
        model = MentionChatModel.__new__(MentionChatModel)
        model.model = MagicMock()
        model.supports_multimodal = True

        tools = MentionChatModel._load_shuiyuan_tools(model)
        inspect_tool = {tool.name: tool for tool in tools}["inspect_image"]

        content, artifact = await inspect_tool.coroutine("upload://a.jpeg")

        self.assertIn("图片已读取", content)
        self.assertEqual(artifact.source, "inspect_image")
        self.assertEqual(artifact.image_urls, ["upload://a.jpeg"])


class TestMentionProviderSelection(unittest.TestCase):
    def test_mimo_provider_selects_mimo_model(self):
        from examples.models.mention_model.mention_model import MentionModel

        with patch.dict(os.environ, {"MENTION_CHAT_PROVIDER": "mimo"}), \
             patch("examples.models.mention_model.mention_model.MentionMimoModel") as mimo_cls, \
             patch("examples.models.mention_model.mention_model.MentionPetModel"):
            mention = MentionModel(MagicMock(), "bot", "wolf_lumine")

        self.assertIs(mention.pumpkin, mimo_cls.return_value)
        mimo_cls.assert_called_once()

    def test_unknown_provider_raises_value_error(self):
        from examples.models.mention_model.mention_model import MentionModel

        with patch.dict(os.environ, {"MENTION_CHAT_PROVIDER": "mystery"}):
            with self.assertRaisesRegex(ValueError, "MENTION_CHAT_PROVIDER"):
                MentionModel(MagicMock(), "bot", "wolf_lumine")

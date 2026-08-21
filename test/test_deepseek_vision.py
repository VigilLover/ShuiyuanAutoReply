import asyncio
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage
from PIL import Image

from shuiyuan_auto_reply.application import BotContext, BotService, HandlerRegistry
from shuiyuan_auto_reply.domain import ReplyResult, VisualMediaArtifact
from shuiyuan_auto_reply.features.mention.deepseek_vision import (
    DeepSeekVisionInput,
    VisionMediaError,
    extract_public_image_urls,
    sniff_image,
)
from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
    DEEPSEEK_DEFAULT_MODEL,
    _mk_deepseek_llm,
    MentionDeepSeekModel,
)
from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.infrastructure.persistence import (
    SQLiteSessionRepository,
    SQLiteStateStore,
)
from shuiyuan_auto_reply.interfaces.api.app import create_app


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), "#5c86e8").save(output, format="PNG")
    return output.getvalue()


class VisionContractTests(unittest.TestCase):
    def test_deepseek_payload_preserves_file_and_image_url_blocks(self):
        model = _mk_deepseek_llm("test-key", DEEPSEEK_DEFAULT_MODEL)
        content = [
            {"type": "text", "text": "看图"},
            {"type": "file", "file_id": "file_123"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]

        payload = model._get_request_payload([HumanMessage(content=content)])

        self.assertEqual(payload["model"], "deepseek-v4-flash-vision-exp")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], content)

    def test_image_extractor_only_returns_explicit_supported_urls(self):
        value = [
            {"type": "text", "text": "![a](https://cdn.example/a.png)"},
            '<img src="https://cdn.example/b.webp">',
            "page https://example.com/not-an-image",
        ]
        self.assertEqual(
            extract_public_image_urls(value),
            ["https://cdn.example/a.png", "https://cdn.example/b.webp"],
        )

    def test_markdown_image_with_cdn_transform_path_is_not_duplicated(self):
        url = "https://cdn.example/original.png/_image_transform.jpg"
        self.assertEqual(extract_public_image_urls(f"![result]({url})"), [url])

    def test_sniff_image_uses_bytes_not_claimed_content_type(self):
        mime_type, width, height = sniff_image(png_bytes())
        self.assertEqual((mime_type, width, height), ("image/png", 4, 3))
        with self.assertRaises(VisionMediaError):
            sniff_image(b"not an image")

    def test_prompt_trace_redacts_deepseek_file_ids(self):
        value = MentionChatModel._prompt_event_value(
            {"type": "file", "file_id": "file_secret"}
        )
        self.assertEqual(value["file_id"], "[REDACTED]")

    def test_deepseek_tool_catalog_has_no_inspect_image_tool(self):
        model = MentionDeepSeekModel.__new__(MentionDeepSeekModel)
        model.model = MagicMock()
        model.supports_multimodal = True
        model.uses_inspect_image_tool = False
        model.state_store = None
        names = {tool.name for tool in model._load_shuiyuan_tools()}
        self.assertNotIn("inspect_image", names)


class VisionAgentNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_images_are_appended_as_synthetic_user_message(self):
        artifact = VisualMediaArtifact(
            artifact_id="asset-1",
            mime_type="image/png",
            local_path="/tmp/asset-1.png",
            byte_count=10,
            source_kind="web_search",
            source_url="https://cdn.example/a.png",
        )
        image = DeepSeekVisionInput(
            source_url="https://cdn.example/a.png",
            source_kind="web_search",
            content_block={"type": "image_url", "image_url": {"url": "https://cdn.example/a.png"}},
            artifact=artifact,
        )
        model = MentionDeepSeekModel.__new__(MentionDeepSeekModel)
        model.vision_media = SimpleNamespace(
            prepare_tool_output=AsyncMock(return_value=[image])
        )
        state = {
            "image_inputs": [],
            "messages": [ToolMessage(content="result", tool_call_id="call-1", name="web_search")],
            "conversation_id": "conversation-1",
        }

        result = await model._collect_tool_output_images(state)

        message = result["messages"][0]
        self.assertIsInstance(message, HumanMessage)
        self.assertEqual(message.content[1]["type"], "image_url")
        self.assertEqual(result["response_visual_artifacts"], [artifact])


class _EchoHandler:
    name = "chat"
    priority = 40

    async def matches(self, _context: BotContext) -> bool:
        return True

    async def handle(self, context: BotContext) -> ReplyResult:
        return ReplyResult(f"images={len(context.request.attachments)}")


class VisionUploadApiTests(unittest.TestCase):
    def test_existing_v1_artifact_table_is_migrated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.sqlite3"
            with sqlite3.connect(path) as db:
                db.executescript(
                    """
                    CREATE TABLE schema_version (version INTEGER NOT NULL);
                    INSERT INTO schema_version(version) VALUES (1);
                    CREATE TABLE artifacts (
                      id TEXT PRIMARY KEY, conversation_id TEXT, run_id TEXT,
                      local_path TEXT NOT NULL, mime_type TEXT NOT NULL,
                      byte_count INTEGER NOT NULL, width INTEGER, height INTEGER,
                      forum_short_path TEXT, created_at TEXT NOT NULL
                    );
                    """
                )
            store = SQLiteStateStore(path)
            asyncio.run(store.initialize())
            with sqlite3.connect(path) as db:
                columns = {
                    row[1] for row in db.execute("PRAGMA table_info(artifacts)")
                }
                version = db.execute("SELECT version FROM schema_version").fetchone()[0]
            self.assertTrue(
                {"source_kind", "source_url", "filename", "sha256", "last_accessed_at"}
                <= columns
            )
            self.assertEqual(version, 2)

    def test_multipart_image_only_message_is_persisted_and_renderable(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ", {"SHUIYUAN_STATE_DIR": temp}
        ):
            store = SQLiteStateStore(Path(temp) / "state.sqlite3")
            asyncio.run(store.initialize())
            service = BotService(
                SQLiteSessionRepository(store), HandlerRegistry([_EchoHandler()])
            )
            container = SimpleNamespace(
                state_store=store,
                bot_service=service,
                aclose=lambda: asyncio.sleep(0),
            )

            async def factory():
                return container

            with TestClient(create_app(factory)) as client:
                conversation = client.post("/api/conversations", json={}).json()
                response = client.post(
                    f"/api/conversations/{conversation['id']}/messages/stream",
                    data={"message": ""},
                    files=[("images", ("sample.png", png_bytes(), "image/png"))],
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("message.completed", response.text)
                detail = client.get(
                    f"/api/conversations/{conversation['id']}"
                ).json()
                self.assertEqual(detail["messages"][0]["role"], "user")
                attachment = detail["messages"][0]["attachments"][0]
                self.assertEqual(attachment["source_kind"], "user_upload")
                self.assertEqual(attachment["filename"], "sample.png")
                self.assertEqual(attachment["mime_type"], "image/png")
                image = client.get(attachment["url"])
                self.assertEqual(image.status_code, 200)
                self.assertEqual(image.content, png_bytes())

    def test_upload_count_and_actual_image_validation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ", {"SHUIYUAN_STATE_DIR": temp}
        ):
            store = SQLiteStateStore(Path(temp) / "state.sqlite3")
            asyncio.run(store.initialize())
            container = SimpleNamespace(
                state_store=store,
                bot_service=BotService(
                    SQLiteSessionRepository(store), HandlerRegistry([_EchoHandler()])
                ),
                aclose=lambda: asyncio.sleep(0),
            )

            async def factory():
                return container

            with TestClient(create_app(factory)) as client:
                conversation = client.post("/api/conversations", json={}).json()
                endpoint = f"/api/conversations/{conversation['id']}/messages/stream"
                too_many = client.post(
                    endpoint,
                    data={"message": "images"},
                    files=[
                        ("images", (f"{index}.png", png_bytes(), "image/png"))
                        for index in range(21)
                    ],
                )
                self.assertEqual(too_many.status_code, 400)
                invalid = client.post(
                    endpoint,
                    data={"message": "fake"},
                    files=[("images", ("fake.png", b"not-image", "image/png"))],
                )
                self.assertEqual(invalid.status_code, 400)
                self.assertIn("无法识别图片", invalid.json()["detail"])

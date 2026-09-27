import unittest
from types import SimpleNamespace

from shuiyuan_auto_reply.application.image_handles import TurnImageRegistry
from shuiyuan_auto_reply.application.tool_results import TurnResults


def artifact(artifact_id: str):
    return SimpleNamespace(artifact_id=artifact_id, uri=f"artifact://{artifact_id}")


class TurnImageRegistryTests(unittest.TestCase):
    def test_display_tokens_are_sequential_and_deduplicated(self):
        registry = TurnImageRegistry()
        self.assertEqual(registry.register(artifact("a"), "猫"), "[图1]")
        self.assertEqual(registry.register(artifact("b")), "[图2]")
        self.assertEqual(registry.register(artifact("a")), "[图1]")
        self.assertEqual(registry.resolve_display(2).artifact.artifact_id, "b")
        self.assertIsNone(registry.resolve_display(3))
        self.assertEqual(registry.display_summary(), "[图1]（猫）、[图2]")

    def test_history_tokens_resolve_to_original_url(self):
        registry = TurnImageRegistry()
        self.assertEqual(registry.register_history("upload://x.jpeg"), "#h1")
        self.assertEqual(registry.register_history("upload://x.jpeg"), "#h1")
        self.assertEqual(registry.resolve_reference("#h1"), "upload://x.jpeg")

    def test_reference_resolution(self):
        registry = TurnImageRegistry()
        registry.register(artifact("a"))
        for token in ("[图1]", "图1", "【图1】"):
            self.assertEqual(registry.resolve_reference(token).artifact_id, "a")
        self.assertIsNone(registry.resolve_reference("https://example.com/a.png"))
        with self.assertRaises(KeyError):
            registry.resolve_reference("[图2]")
        with self.assertRaises(KeyError):
            registry.resolve_reference("#h1")

    def test_each_turn_gets_a_fresh_registry(self):
        first, second = TurnResults(), TurnResults()
        first.images.register(artifact("a"))
        self.assertEqual(second.images.display, [])


if __name__ == "__main__":
    unittest.main()


class ReferenceHandleResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_handles_resolve_to_loadable_urls(self):
        import base64
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock

        from shuiyuan_auto_reply.application.tool_results import current_turn
        from shuiyuan_auto_reply.features.mention.image_generation import (
            ImageGenerationService,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.png"
            path.write_bytes(b"png-bytes")
            generated = SimpleNamespace(
                artifact_id="a", local_path=str(path), mime_type="image/png"
            )
            store = SimpleNamespace(
                get_artifact=AsyncMock(
                    return_value=SimpleNamespace(
                        local_path=str(path), mime_type="image/png"
                    )
                )
            )
            service = ImageGenerationService(object(), store)
            turn = TurnResults()
            turn.images.register(generated)
            turn.images.register_history("upload://old.jpeg")
            turn.images.register_history("artifact://web-1")
            token = current_turn.set(turn)
            try:
                resolved, unknown = await service._resolve_reference_handles(
                    [
                        {"key": "new", "url": "[图1]"},
                        {"key": "old", "url": "#h1"},
                        {"key": "web", "url": "#h2"},
                        {"key": "avatar", "url": "https://x/a.png"},
                        {"key": "ghost", "url": "[图5]"},
                    ]
                )
            finally:
                current_turn.reset(token)

        data_url = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode()
        self.assertEqual(unknown, ["[图5]"])
        self.assertEqual(
            [item["url"] for item in resolved],
            [data_url, "upload://old.jpeg", data_url, "https://x/a.png"],
        )
        store.get_artifact.assert_awaited_once_with("web-1")


class RenderImagePlaceholderTests(unittest.TestCase):
    def setUp(self):
        from shuiyuan_auto_reply.application.image_handles import (
            render_image_placeholders,
        )

        self.render = render_image_placeholders
        self.registry = TurnImageRegistry()
        self.generated = artifact("gen-1")
        self.registry.register(self.generated, "本轮生成图")

    def test_markdown_and_bare_tokens_render_to_turn_artifacts(self):
        for text in (
            "![豆豆眼头像]([图1])",
            "![豆豆眼头像](图1)",
            "看这张：[图1]",
            "【图1】",
        ):
            result = self.render(text, self.registry)
            self.assertIn("(artifact://gen-1)", result.text, text)
            self.assertEqual(result.used, [self.generated])
            self.assertEqual(result.rejected, [])

    def test_history_upload_link_is_dropped_but_turn_image_kept(self):
        # 814376ed: stale upload:// copied from history next to the real image.
        text = "帽子留住了。\n\n![豆豆眼浅金发帽子头像](upload://9nS5o.jpeg)\n\n[图1]"
        result = self.render(text, self.registry)
        self.assertNotIn("upload://", result.text)
        self.assertEqual(result.text.count("artifact://gen-1"), 1)
        self.assertEqual(len(result.rejected), 1)

    def test_fabricated_image_without_generation_is_removed(self):
        # e316ee20 / d2287bfc: no generate_image call, image link invented.
        empty = TurnImageRegistry()
        for text in (
            "背景加好了。\n\n![豆豆眼·背景版](artifact://made-up)\n\n想调再说。",
            "背景加好了。\n\n![豆豆眼·背景版](/api/artifacts/made-up)",
            '背景加好了。<img src="https://x/y.png" alt="a">',
            "背景加好了。\n\n![豆豆眼·背景版]([图1])",
            "背景加好了 upload://abc.jpeg",
        ):
            result = self.render(text, empty)
            self.assertNotRegex(result.text, r"!\[|<img|://|/api/artifacts|图1")
            self.assertTrue(result.text.startswith("背景加好了"))
            self.assertEqual(result.used, [])
            self.assertTrue(result.rejected)

    def test_unknown_token_is_dropped(self):
        result = self.render("[图1] 和 [图9]", self.registry)
        self.assertIn("artifact://gen-1", result.text)
        self.assertNotIn("图9", result.text)
        self.assertEqual(result.rejected, ["[图9]"])


class HistoryImageStrippingTests(unittest.TestCase):
    def test_history_links_become_reference_handles(self):
        from shuiyuan_auto_reply.application.image_handles import (
            strip_history_images,
        )

        registry = TurnImageRegistry()
        text = (
            "石壁版在这。\n\n![豆豆眼石雕鹰·石壁版](upload://pLt9.jpeg)\n\n"
            '<img src="/api/artifacts/abc" alt="x"> 还有 upload://raw.png'
        )
        cleaned = strip_history_images(text, registry)
        self.assertNotRegex(cleaned, r"upload://|/api/artifacts|!\[|<img")
        self.assertIn("[历史图 #h1：豆豆眼石雕鹰·石壁版]", cleaned)
        self.assertIn("[历史图 #h2]", cleaned)
        self.assertIn("#h3", cleaned)
        # a26ac482: editing the previous version still reaches the real image.
        self.assertEqual(registry.resolve_reference("#h1"), "upload://pLt9.jpeg")

    def test_history_messages_are_copied_not_mutated(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from shuiyuan_auto_reply.features.mention.context import ContextMixin

        registry = TurnImageRegistry()
        original = AIMessage(content="![旧图](upload://old.jpeg)")
        cleaned = ContextMixin._without_image_addresses(original, registry)
        self.assertEqual(original.content, "![旧图](upload://old.jpeg)")
        self.assertEqual(cleaned.content, "[历史图 #h1：旧图]")
        plain = HumanMessage(content="没有图")
        self.assertIs(ContextMixin._without_image_addresses(plain, registry), plain)
        parts = HumanMessage(
            content=[
                {"type": "text", "text": "![a](upload://p.png)"},
                {"type": "image"},
            ]
        )
        self.assertEqual(
            ContextMixin._without_image_addresses(parts, registry).content[0]["text"],
            "[历史图 #h2：a]",
        )

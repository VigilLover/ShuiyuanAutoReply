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

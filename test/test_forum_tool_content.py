import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.post_content import clean_raw, parse_content
from shuiyuan_auto_reply.features.mention.shuiyuan_tools_objects import PostShort
from shuiyuan_auto_reply.features.mention.shuiyuan_tools_wrapper import (
    ShuiyuanToolsWrapper,
)


def post(raw=None, cooked='<p>Hello <a class="mention" href="/u/alice">@Alice</a></p>'):
    return SimpleNamespace(
        id=100,
        topic_id=42,
        post_number=7,
        reply_to_post_number=3,
        user_id=1,
        username="author",
        name=None,
        raw=raw,
        cooked=cooked,
    )


class ForumToolContentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.turn = TurnResults()
        self.token = current_turn.set(self.turn)

    async def asyncTearDown(self):
        current_turn.reset(self.token)

    async def test_exact_lookup_fills_raw_and_preserves_reply(self):
        model = SimpleNamespace(
            get_post_details_by_post_number=AsyncMock(return_value=post()),
            get_post_details=AsyncMock(return_value=post("@Alice " * 1000)),
        )
        tools = ShuiyuanToolsWrapper(model)
        result = await tools.get_post_details_by_post_number(42, 7)
        self.assertEqual(result.to_dict()["reply_to_post_number"], 3)
        self.assertEqual(result.to_dict()["content"], ("@Alice " * 1000).strip())
        self.assertEqual(result.to_dict()["content_source"], "raw")
        self.assertIs(await tools.get_post_by_id(100), result)
        model.get_post_details.assert_awaited_once_with(100)

    async def test_long_post_can_be_reconstructed_and_empty_raw_is_valid(self):
        content = "用户原文\n" * 4000
        result = PostShort(post(content), full=True).to_dict()
        page = self.turn.read(result["result_id"], result["next_cursor"])
        self.assertEqual(result["content"] + page["content"], content.strip())
        self.assertTrue(result["truncated"])
        self.assertEqual(PostShort(post(""), full=True).to_dict()["content"], "")

    async def test_fallback_and_identity_errors(self):
        model = SimpleNamespace(
            get_post_details_by_post_number=AsyncMock(return_value=post()),
            get_post_details=AsyncMock(side_effect=TimeoutError("offline")),
        )
        result = await ShuiyuanToolsWrapper(model).get_post_details_by_post_number(
            42, 7
        )
        self.assertEqual(result.to_dict()["content_source"], "cooked_text")
        self.assertTrue(result.warnings)
        self.assertNotIn("<p>", result.to_dict()["content"])
        result = await ShuiyuanToolsWrapper(model).get_post_details_by_post_number(
            42, 8
        )
        self.assertEqual(result["status"], "error")

    async def test_exact_batch_preserves_inputs_and_retries_only_failures(self):
        user = SimpleNamespace(
            id=5, username="Alice", name="Someone", avatar_template="/avatar/{size}.png"
        )
        model = SimpleNamespace(
            get_user_by_username=AsyncMock(
                side_effect=lambda name: user if name.casefold() == "alice" else None
            )
        )
        tools = ShuiyuanToolsWrapper(model)
        result = await tools.get_users(["Alice", "@alice", "missing"], True)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["items"]), 3)
        self.assertEqual(model.get_user_by_username.await_count, 2)
        await tools.get_users(["Alice", "missing"], True)
        self.assertEqual(
            model.get_user_by_username.await_count, 2
        )  # deterministic missing user is cached
        await tools.get_user("Alice", True, refresh=True)
        self.assertEqual(model.get_user_by_username.await_count, 3)

    def test_cleanup_preserves_body_and_quoted_mentions(self):
        body = (
            "```html\n<div data-signature>example</div>\n```\n[quote]@Alice[/quote]\n"
        )
        self.assertEqual(clean_raw(body), body.strip())
        decorated = (
            body
            + "<div data-signature>bot</div>\n<!-- abcdefghijklmnopqrst -->\n<!-- 来自小狼的自动回复 -->"
        )
        self.assertEqual(clean_raw(decorated), body.strip())
        data = parse_content(
            None,
            '<blockquote><a class="mention" href="/u/alice">@Alice</a></blockquote><pre><a class="mention" href="/u/fake">@fake</a></pre>',
        )
        self.assertEqual(data["mentions"], [{"username": "alice", "source": "quote"}])

    async def test_cursor_on_exact_tool_and_alias_refresh(self):
        raw = "paragraph\n" * 2500
        model = SimpleNamespace(
            get_post_details=AsyncMock(return_value=post(raw)),
            get_post_details_by_post_number=AsyncMock(return_value=post(raw)),
        )
        tools = ShuiyuanToolsWrapper(model)
        first = await tools.get_post_by_id(100)
        rest = await tools.get_post_details_by_post_number(42, 7, cursor=12000)
        final = await tools.get_post_by_id(100, cursor=24000)
        self.assertEqual(
            first.to_dict()["content"] + rest["content"] + final["content"], raw.strip()
        )
        model.get_post_details.assert_awaited_once()
        model.get_post_details_by_post_number.assert_not_awaited()
        model.get_post_details.return_value = post("updated")
        await tools.get_post_by_id(100, refresh=True)
        reread = await tools.get_post_details_by_post_number(42, 7)
        self.assertEqual(reread.to_dict()["content"], "updated")

    def test_multiline_quote_mentions_are_distinct_from_body(self):
        result = parse_content(
            "[quote]\n@Alice\n@Bob\n[/quote]\n@Carol\n~~~\n@fake\n~~~", ""
        )
        self.assertEqual(
            result["mentions"],
            [
                {"username": "Alice", "source": "quote"},
                {"username": "Bob", "source": "quote"},
                {"username": "Carol", "source": "body"},
            ],
        )

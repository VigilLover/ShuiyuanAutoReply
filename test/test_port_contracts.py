import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from shuiyuan_auto_reply.domain import (
    ActorRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ForumContextRef,
    ReplyRequest,
)
from shuiyuan_auto_reply.infrastructure.forum import (
    EmptyChannelContextProvider,
    ForumOutputFormatter,
    ForumChannelContextProvider,
)
from shuiyuan_auto_reply.infrastructure.retrieval import Neo4jStyleRetriever
from shuiyuan_auto_reply.infrastructure.forum import ShuiyuanForumGateway
from shuiyuan_auto_reply.infrastructure.persistence import PostgresLongTermMemoryAdapter
from shuiyuan_auto_reply.application.ports.memory import MemoryCommand, MemoryScope


def forum_request() -> ReplyRequest:
    return ReplyRequest(
        "request",
        ConversationRef(Channel.FORUM, "topic:7", "bot", "wolf_lumine"),
        ActorRef(Channel.FORUM, "42", "user"),
        "query",
        DispatchMode.AUTO,
        ForumContextRef(7),
    )


class StyleRetrieverContract(unittest.IsolatedAsyncioTestCase):
    async def test_persona_query_and_limit_are_forwarded_unchanged(self):
        manager = SimpleNamespace(
            search_similar=AsyncMock(
                return_value=[SimpleNamespace(text="example", score=0.9)]
            )
        )

        async def factory():
            return manager

        result = await Neo4jStyleRetriever(factory).search("persona", "query", 8)
        manager.search_similar.assert_awaited_once_with(
            "query", top_k=8, userid="persona"
        )
        self.assertEqual(result[0].text, "example")
        self.assertEqual(result[0].score, 0.9)


class ForumGatewayContract(unittest.IsolatedAsyncioTestCase):
    async def test_reply_arguments_are_forwarded_without_format_changes(self):
        model = SimpleNamespace(reply_to_post=AsyncMock())
        await ShuiyuanForumGateway(model).reply("text", 12, 3)
        model.reply_to_post.assert_awaited_once_with("text", 12, 3)


class LongTermMemoryContract(unittest.IsolatedAsyncioTestCase):
    async def test_namespace_query_limit_and_command_are_forwarded(self):
        delegate = SimpleNamespace(
            search_mention_memory=AsyncMock(return_value="memory"),
            manage_mention_memory=AsyncMock(return_value="managed"),
        )
        adapter = PostgresLongTermMemoryAdapter(delegate)
        self.assertEqual(
            await adapter.search(MemoryScope("api:session"), "query", 5),
            "memory",
        )
        delegate.search_mention_memory.assert_awaited_once_with(
            target_user_id="api:session", query="query", limit=5
        )
        command = MemoryCommand({"action": "delete", "memory_id": "id"})
        self.assertEqual(await adapter.manage(command), "managed")
        delegate.manage_mention_memory.assert_awaited_once_with(
            action="delete", memory_id="id"
        )


class ChannelContextContract(unittest.IsolatedAsyncioTestCase):
    async def test_empty_provider_never_loads_a_forum_topic(self):
        context = await EmptyChannelContextProvider().load(forum_request())
        self.assertEqual(context.recent_messages, "无近期回帖记录")

    async def test_forum_provider_uses_real_topic_id(self):
        loader = AsyncMock(return_value="recent")
        context = await ForumChannelContextProvider(loader).load(forum_request())
        loader.assert_awaited_once_with(7)
        self.assertEqual(context.recent_messages, "recent")


class ForumOutputContract(unittest.TestCase):
    def test_signature_random_comment_and_tag_are_unchanged(self):
        with patch(
            "shuiyuan_auto_reply.shuiyuan.user_action_model.BaseUserActionModel._generate_random_string",
            return_value="abcdefghijklmnopqrst",
        ):
            result = ForumOutputFormatter().format_chat("正文", "小狼bot")
        self.assertEqual(
            result,
            "正文\n<div data-signature>\n\n---\n"
            "[right]这里是AI小狼<small>(Pumpkin Edition)</small> :robot: [/right]\n"
            "</div>\n\n<!-- abcdefghijklmnopqrst -->\n<!-- 来自小狼的自动回复 -->",
        )

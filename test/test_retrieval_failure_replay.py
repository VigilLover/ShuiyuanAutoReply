import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, ToolMessage
from test_forum_agent_flow import OfflineChat

from shuiyuan_auto_reply.shuiyuan.objects import User


def _post(number: int):
    return SimpleNamespace(
        id=10_000 + number,
        topic_id=515777,
        post_number=number,
        reply_to_post_number=None,
        user_id=number,
        username=f"user-{number}",
        name=None,
        raw=f"第 {number} 楼内容",
        cooked=f"<p>第 {number} 楼内容</p>",
        created_at=f"2026-09-15T00:{number % 60:02d}:00Z",
    )


class RetrievalReplayChat(OfflineChat):
    def __init__(self, model):
        super().__init__(model)
        self.retrieval_prompts = []
        self.final_prompts = []
        self.llm_with_tools = SimpleNamespace(ainvoke=self.retrieve)
        self.llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=[
                    AIMessage(
                        content=(
                            '<｜｜DSML｜｜ invoke name="forum_search">'
                            '<｜｜DSML｜｜ parameter name="cursor">stale</｜｜DSML｜｜ parameter>'
                        ),
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                        },
                    ),
                    AIMessage(
                        content="话题总结完成。",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                        },
                    ),
                ]
            )
        )

    async def retrieve(self, prompt):
        self.retrieval_prompts.append(prompt.to_messages())
        turn = len(self.retrieval_prompts)
        if turn == 1:
            call = {
                "id": "find-topic",
                "name": "forum_search",
                "args": {"kind": "topics", "query": "抽象日记 3.0", "limit": 8},
            }
        elif turn == 2:
            call = {
                "id": "read-start",
                "name": "forum_read",
                "args": {"topic_id": 515777, "order": "oldest", "limit": 20},
            }
        elif turn <= 5:
            tool_message = next(
                message
                for message in reversed(prompt.to_messages())
                if isinstance(message, ToolMessage)
            )
            cursor = json.loads(tool_message.content)["next_cursor"]
            call = {
                "id": f"read-{turn}",
                "name": "forum_read",
                "args": {"cursor": cursor, "limit": 20},
            }
        else:
            # Reproduce the unnecessary restart from the failed production run.
            call = {
                "id": "redundant-restart",
                "name": "forum_read",
                "args": {"topic_id": 515777, "order": "oldest", "limit": 20},
            }
        return AIMessage(
            content="",
            tool_calls=[call],
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )


async def _replay_large_topic_retrieval_and_finalizer_repair():
    posts = [_post(number) for number in range(1, 72)]

    async def search_forum(_query, *, page):
        assert page == 1
        return {
            "posts": [],
            "topics": [
                {
                    "id": 515777,
                    "title": "中杯小狼的抽象日记 3.0",
                    "posts_count": 71,
                }
            ],
            "more_posts": False,
        }

    async def read_topic(topic_id, *, offset, limit, username=None, ascending=False):
        assert topic_id == 515777
        assert username is None
        assert ascending is True
        page = posts[offset : offset + limit]
        next_offset = offset + len(page)
        return (
            "中杯小狼的抽象日记 3.0",
            page,
            next_offset,
            next_offset < len(posts),
        )

    model = SimpleNamespace(
        search_forum=AsyncMock(side_effect=search_forum),
        read_topic_post_page=AsyncMock(side_effect=read_topic),
    )
    runtime = RetrievalReplayChat(model)
    with patch(
        "shuiyuan_auto_reply.features.mention.mention_chat_model.emit_event",
        new_callable=AsyncMock,
    ) as emit:
        result = await runtime.get_pumpkin_response(
            None,
            None,
            "总结中杯小狼的抽象日记 3.0",
            User(id=1, username="requester", name=None),
            session_id="replay",
            load_forum_context=False,
        )

    assert result == "话题总结完成。"
    assert len(runtime.retrieval_prompts) == 6
    assert runtime.llm.ainvoke.await_count == 2
    assert model.search_forum.await_count == 1
    assert model.read_topic_post_page.await_count == 4
    assert all(
        call.kwargs["limit"] == 20
        for call in model.read_topic_post_page.await_args_list
    )
    assert not any(
        call[0][0] == "tool.execution"
        and "after_date" in call[0][1].get("arguments", {})
        for call in emit.await_args_list
    )

    usage = [
        call.args[1]
        for call in emit.await_args_list
        if call.args and call.args[0] == "usage.recorded"
    ]
    assert len(usage) == 8
    assert sum(item["total_tokens"] for item in usage) == 96
    for call in runtime.llm.ainvoke.await_args_list:
        messages = call.args[0].to_messages()
        assert not any(isinstance(message, ToolMessage) for message in messages)
        assert "stale" not in "\n".join(str(message.content) for message in messages)


def test_replay_large_topic_retrieval_and_finalizer_repair():
    import asyncio

    asyncio.run(_replay_large_topic_retrieval_and_finalizer_repair())

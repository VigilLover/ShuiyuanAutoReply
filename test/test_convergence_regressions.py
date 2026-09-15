import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from test_forum_agent_flow import OfflineChat

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.infrastructure.persistence.state import SQLiteStateStore
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository


class ConvergenceRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_migration_keeps_draft_separate_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStateStore(Path(tmp) / "state.sqlite3")
            await store.initialize()
            prompt = (
                FilePromptRepository()
                .load("wolf_lumine", set(), PromptScope.FORUM)
                .system_prompt
            )
            defaults = {
                "system_prompt": prompt,
                "api_format": "responses",
                "enabled_tools": ["get_post"],
            }
            profile = await store.get_profile("forum", defaults)
            self.assertEqual(profile["active"]["prompt_mode"], "managed")
            draft = {**defaults, "system_prompt": "Custom legacy persona {username}"}
            await store.save_profile_draft("forum", draft)
            for _ in range(3):
                profile = await store.get_profile("forum", defaults)
                self.assertEqual(profile["active_revision"], 1)
                self.assertEqual(profile["active"]["system_prompt"], prompt)
                self.assertEqual(profile["draft"]["prompt_mode"], "legacy")
                self.assertEqual(profile["draft"]["enabled_tools"], ["get_post"])
                self.assertEqual(profile["draft"]["api_format"], "responses")
            await store.apply_profile("forum")
            profile = await store.get_profile("forum", defaults)
            self.assertEqual(profile["active_revision"], 2)
            self.assertEqual(profile["active"]["system_prompt"], draft["system_prompt"])

    async def test_changed_keywords_with_same_posts_stop_after_no_new_evidence(
        self,
    ):
        post = SimpleNamespace(
            id=10,
            topic_id=42,
            post_number=1,
            reply_to_post_number=None,
            user_id=1,
            username="Alice",
            name="Alice",
            raw="I like song A",
            cooked="<p>I like song A</p>",
        )
        model = SimpleNamespace(
            search_forum=AsyncMock(
                return_value={
                    "posts": [
                        {
                            "id": 10,
                            "topic_id": 42,
                            "post_number": 1,
                            "username": "Alice",
                            "blurb": "I like song A",
                        }
                    ],
                    "topics": [{"id": 42, "title": "title"}],
                    "more_posts": False,
                }
            )
        )
        runtime = OfflineChat(model)
        turn = TurnResults()
        turn.progress.topic_id = 42
        token = current_turn.set(turn)
        try:
            for i, term in enumerate(["music", "song", "favourite", "playlist"]):
                await runtime._execute_tools(
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": str(i),
                                        "name": "forum_search",
                                        "args": {"query": term, "topic_id": 42},
                                    }
                                ],
                            )
                        ]
                    }
                )
            self.assertEqual(turn.progress.phase, "final")
            self.assertEqual(turn.control.stop_reason, "no_new_evidence")
            self.assertEqual(model.search_forum.await_count, 4)
        finally:
            current_turn.reset(token)

    async def test_completed_topic_short_circuits_restarted_unfiltered_read(self):
        post = SimpleNamespace(
            id=10,
            topic_id=42,
            post_number=1,
            reply_to_post_number=None,
            user_id=1,
            username="Alice",
            name="Alice",
            raw="完整内容",
            cooked="<p>完整内容</p>",
            created_at="2026-09-15T00:00:00Z",
        )
        model = SimpleNamespace(
            read_topic_post_page=AsyncMock(return_value=("Topic", [post], 1, False))
        )
        runtime = OfflineChat(model)
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            call = lambda call_id: {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": call_id,
                                "name": "forum_read",
                                "args": {"topic_id": 42, "order": "oldest"},
                            }
                        ],
                    )
                ]
            }
            first = await runtime._execute_tools(call("first"))
            self.assertTrue(json.loads(first["messages"][0].content)["complete"])
            self.assertEqual(turn.topic_coverage[42], {1})
            self.assertEqual(turn.progress.phase, "investigate")

            second = await runtime._execute_tools(call("second"))
            payload = json.loads(second["messages"][0].content)
            self.assertEqual(payload["items"][0]["ref"], "forum:42/1")
            self.assertTrue(payload["complete"])
            self.assertEqual(turn.progress.phase, "final")
            self.assertEqual(turn.control.stop_reason, "source_complete")
            model.read_topic_post_page.assert_awaited_once()
        finally:
            current_turn.reset(token)

    async def test_removed_progress_tool_is_rejected_while_search_runs(self):
        model = SimpleNamespace(
            search_forum=AsyncMock(return_value={"posts": [], "topics": []})
        )
        runtime = OfflineChat(model)
        turn = TurnResults()
        turn.progress.topic_id = 42
        token = current_turn.set(turn)
        try:
            result = await runtime._execute_tools(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "s",
                                    "name": "forum_search",
                                    "args": {"query": "music"},
                                },
                                {
                                    "id": "p",
                                    "name": "update_task_progress",
                                    "args": {
                                        "goal": "music",
                                        "gaps": {"music": "identify songs"},
                                        "findings": [],
                                        "authors": [],
                                    },
                                },
                            ],
                        )
                    ]
                }
            )
            self.assertEqual(
                [message.status for message in result["messages"]],
                ["success", "error"],
            )
            model.search_forum.assert_awaited_once()
        finally:
            current_turn.reset(token)

    def test_invalid_or_index_read_does_not_count_as_new_evidence(self):
        turn = TurnResults()
        result = turn.save("hello")
        self.assertEqual(turn.read(result, -1)["status"], "error")
        index = turn.save([{"result_id": result}], index=True)
        turn.read(index)
        self.assertEqual(len(turn.read_pages), 0)
        turn.read(result)
        turn.read(result)
        self.assertEqual(len(turn.read_pages), 1)

    async def test_removed_result_reader_is_rejected_without_exposing_storage(self):
        runtime = OfflineChat(SimpleNamespace())
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            ids = [turn.save(str(i) + "x" * 15000) for i in range(4)]
            response = await runtime._execute_tools(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": str(i),
                                    "name": "read_tool_result",
                                    "args": {"result_id": key, "limit": 12000},
                                }
                                for i, key in enumerate(ids)
                            ],
                        )
                    ]
                }
            )
            self.assertTrue(
                all(message.status == "error" for message in response["messages"])
            )
            self.assertTrue(
                all(
                    "read_tool_result" in message.content
                    for message in response["messages"]
                )
            )
        finally:
            current_turn.reset(token)

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
from shuiyuan_auto_reply.application.task_progress import update_task_progress
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

    async def test_changed_keywords_with_same_posts_trigger_review_without_rewriting_query(
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
            search_post_details_by_optional_username_topic=AsyncMock(
                return_value={"title": [post]}
            )
        )
        runtime = OfflineChat(model)
        turn = TurnResults()
        turn.progress.topic_id = 42
        turn.progress.authors = ["Alice"]
        turn.progress.gaps = {"music": "find explicit music preferences"}
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
                                        "name": "search_posts",
                                        "args": {"term": term, "gap_id": "music"},
                                    }
                                ],
                            )
                        ]
                    }
                )
            self.assertEqual(turn.progress.phase, "review")
            self.assertEqual(len(turn.evidence), 1)
            # The controller must not add an author filter the model never asked for:
            # doing so silently changed which posts came back and the model then
            # distrusted its own query scope.
            for (
                call
            ) in model.search_post_details_by_optional_username_topic.await_args_list:
                self.assertEqual(call.args[2:4], (None, 42))
        finally:
            current_turn.reset(token)

    async def test_progress_update_and_search_can_share_batch(self):
        model = SimpleNamespace(
            search_post_details_by_optional_username_topic=AsyncMock(return_value={})
        )
        runtime = OfflineChat(model)
        runtime.tools.append(
            StructuredTool.from_function(coroutine=update_task_progress)
        )
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
                                    "name": "search_posts",
                                    "args": {"term": "music", "gap_id": "music"},
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
            self.assertTrue(all(m.status == "success" for m in result["messages"]))
            model.search_post_details_by_optional_username_topic.assert_awaited_once()
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

    async def test_parallel_readbacks_have_explicit_bounded_pages(self):
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
            outputs = [json.loads(message.content) for message in response["messages"]]
            self.assertLessEqual(sum(len(p["content"]) for p in outputs), 6000)
            self.assertTrue(
                all(
                    p["truncated"] and p["page_limit"] == p["next_cursor"]
                    for p in outputs
                )
            )
        finally:
            current_turn.reset(token)

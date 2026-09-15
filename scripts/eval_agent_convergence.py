"""Opt-in real text-model evaluation against synthetic forum and image tools.

Default only lists fixtures. --live-model explicitly permits paid text-model calls;
no real forum reads, forum writes, image generation or MCP calls are made.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))


def scenarios():
    return json.loads(
        (ROOT / "test/fixtures/agent_convergence/scenarios.json").read_text()
    )


async def evaluate(scenario, model_name):
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from openai import AsyncOpenAI
    from test_forum_agent_flow import OfflineChat

    from shuiyuan_auto_reply.application.task_progress import update_task_progress
    from shuiyuan_auto_reply.application.tool_results import current_turn
    from shuiyuan_auto_reply.domain.tool_error import ReadFailure
    from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
    from shuiyuan_auto_reply.shuiyuan.objects import User

    class Forum:
        def __init__(self):
            self.posts = [
                SimpleNamespace(
                    id=10000 + p["number"],
                    topic_id=42,
                    post_number=p["number"],
                    reply_to_post_number=None,
                    user_id={"Alice": 1, "Bob": 2, "Carol": 3}[p["author"]],
                    username=p["author"],
                    name=p["author"],
                    raw=p["text"],
                    cooked="",
                )
                for p in scenario["posts"]
            ]

        async def get_post_details_by_post_number(self, topic_id, post_number):
            for p in self.posts:
                if p.topic_id == topic_id and p.post_number == post_number:
                    return p
            raise ReadFailure(404)

        async def get_post_details(self, post_id):
            return await self.get_post_details_by_post_number(42, post_id - 10000)

        async def query_recent_posts_by_topic_id(self, topic_id, limit):
            return "Synthetic topic", self.posts[-limit:]

        async def search_post_details_by_optional_username_topic(
            self, term="", latest=False, username=None, topic_id=None
        ):
            # Deterministic closed corpus: changing keywords cannot invent new evidence.
            return {
                "Synthetic topic": [
                    p
                    for p in self.posts
                    if not username or p.username.casefold() == username.casefold()
                ]
            }

        async def get_user_by_username(self, username):
            names = {"alice": 1, "bob": 2, "carol": 3}
            if username.casefold() not in names:
                return None
            return SimpleNamespace(
                id=names[username.casefold()],
                username=username,
                name=username,
                avatar_template="/user_avatar/shuiyuan.sjtu.edu.cn/"
                + username
                + "/{size}/avatar.png",
            )

    class Runtime(OfflineChat):
        async def _finalize_response(self, state):
            turn = current_turn.get()
            self.metrics = {
                **turn.control.metrics(),
                "external_queries": turn.external_requests,
                "evidence_count": len(turn.evidence),
            }
            return await super()._finalize_response(state)

    runtime = Runtime(Forum())
    runtime.tools = [
        t
        for t in runtime.tools
        if t.name
        in {
            "get_post",
            "get_post_by_id",
            "get_user",
            "get_users",
            "search_posts",
            "recent_posts",
            "read_tool_result",
        }
    ]
    runtime.tools.append(StructuredTool.from_function(coroutine=update_task_progress))

    async def prepare_image_references(references: list[dict]):
        """Synthetic image preparation; accepts key/url/label and never downloads images."""
        return {
            "reference_set_id": "fixture-set",
            "status": "ok",
            "items": [
                {
                    "key": p["key"],
                    "label": p.get("label", p["key"]),
                    "index": i + 1,
                    "status": "ok",
                }
                for i, p in enumerate(references)
            ],
        }

    async def generate_image(prompt: str, reference_set_id: str):
        """Synthetic generation: only fixture-set exists; no real image is generated."""
        if reference_set_id != "fixture-set":
            return {"status": "error", "error": "unknown_set"}
        return {
            "status": "ok",
            "fixture_only": True,
            "message": "Simulated image generation completed; do not claim a real image exists.",
        }

    runtime.tools += [
        StructuredTool.from_function(coroutine=prepare_image_references),
        StructuredTool.from_function(coroutine=generate_image),
    ]
    runtime.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FilePromptRepository().load("wolf_lumine", set()).system_prompt),
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("messages"),
        ]
    )
    async with AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    ) as client:

        class Model:
            def __init__(self, tools):
                self.tools = tools

            async def ainvoke(self, prompt):
                messages = []
                for m in prompt.to_messages():
                    item = {
                        "role": {"human": "user", "ai": "assistant"}.get(
                            m.type, m.type
                        ),
                        "content": m.content or "",
                    }
                    if m.type == "tool":
                        item["tool_call_id"] = m.tool_call_id
                    if getattr(m, "tool_calls", None):
                        item["tool_calls"] = [
                            {
                                "id": t["id"],
                                "type": "function",
                                "function": {
                                    "name": t["name"],
                                    "arguments": json.dumps(t["args"]),
                                },
                            }
                            for t in m.tool_calls
                        ]
                    messages.append(item)
                args = {"model": model_name, "messages": messages}
                if self.tools:
                    args["tools"] = [convert_to_openai_tool(t) for t in self.tools]
                response = await client.chat.completions.create(**args)
                m = response.choices[0].message
                return AIMessage(
                    content=m.content or "",
                    tool_calls=[
                        {
                            "id": t.id,
                            "name": t.function.name,
                            "args": json.loads(t.function.arguments),
                        }
                        for t in m.tool_calls or []
                    ],
                )

        runtime.llm_with_tools = Model(runtime.tools)
        runtime.llm = Model([])
        answer = await runtime.get_pumpkin_response(
            42,
            None,
            scenario["request"],
            User(id=9, username="Requester", name="Requester"),
        )
    return {
        "scenario": scenario["id"],
        "model": model_name,
        "metrics": runtime.metrics,
        "answer": answer,
        "expected": scenario["expected"],
        "fixture_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-model", action="store_true")
    parser.add_argument(
        "--scenario", choices=[s["id"] for s in scenarios()], default="music"
    )
    parser.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    args = parser.parse_args()
    if not args.live_model:
        print(json.dumps(scenarios(), ensure_ascii=False, indent=2))
        return
    print(
        json.dumps(
            asyncio.run(
                evaluate(
                    next(s for s in scenarios() if s["id"] == args.scenario), args.model
                )
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

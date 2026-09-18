import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.shuiyuan_tools_wrapper import (
    ShuiyuanToolsWrapper,
)
from shuiyuan_auto_reply.features.mention.tool_catalog import migrate_tool_names


def _post(*, post_id=10, number=3, raw="正文", image=""):
    if image:
        raw += f"\n![图]({image})"
    return SimpleNamespace(
        id=post_id,
        topic_id=42,
        post_number=number,
        reply_to_post_number=None,
        user_id=7,
        username="alice",
        name="Alice",
        raw=raw,
        cooked=f"<p>{raw}</p>",
        created_at="2026-09-01T00:00:00Z",
    )


async def _forum_search_uses_native_snippets_and_opaque_cursor():
    model = SimpleNamespace(
        search_forum=AsyncMock(
            return_value={
                "posts": [
                    {
                        "id": index,
                        "topic_id": 42,
                        "post_number": index,
                        "username": "alice",
                        "blurb": "<span class='search-highlight'>命中</span> 内容",
                    }
                    for index in range(1, 4)
                ],
                "topics": [{"id": 42, "title": "测试话题"}],
                "more_posts": True,
            }
        )
    )
    token = current_turn.set(TurnResults())
    try:
        tools = ShuiyuanToolsWrapper(model)
        first = await tools.forum_search(
            query="关键词",
            username="alice",
            after_date="2026-08-01",
            before_date="2026-09-01",
            sort="latest",
            limit=2,
        )
        assert first["status"] == "ok"
        assert [item["text"] for item in first["items"]] == ["命中 内容"] * 2
        assert first["next_cursor"].startswith("c_")
        query = model.search_forum.await_args.args[0]
        assert "user:alice" in query
        assert "after:2026-08-01" in query
        assert "before:2026-09-01" in query
        assert "order:latest" in query

        second = await tools.forum_search(cursor=first["next_cursor"], limit=2)
        assert [item["ref"] for item in second["items"]] == ["forum:42/3"]
        assert model.search_forum.await_count == 1
    finally:
        current_turn.reset(token)


async def _forum_search_rejects_conflicting_structured_filter():
    tools = ShuiyuanToolsWrapper(SimpleNamespace(search_forum=AsyncMock()))
    result = await tools.forum_search(query="user:bob", username="alice")
    assert result["status"] == "error"
    assert result["code"] == "invalid_arguments"
    assert result["message"] == "Conflicting user filter"
    assert result["retryable"] is False
    assert "hint" in result


async def _cursor_accepts_matching_arguments_and_rejects_conflicts():
    model = SimpleNamespace(
        search_forum=AsyncMock(
            return_value={
                "posts": [
                    {
                        "topic_id": 42,
                        "post_number": number,
                        "username": "alice",
                        "blurb": f"正文 {number}",
                    }
                    for number in range(1, 4)
                ],
                "topics": [{"id": 42, "title": "测试话题"}],
                "more_posts": False,
            }
        )
    )
    token = current_turn.set(TurnResults())
    try:
        tools = ShuiyuanToolsWrapper(model)
        first = await tools.forum_search(
            kind="posts", query="测试", sort="latest", limit=2
        )
        continued = await tools.forum_search(
            cursor=first["next_cursor"],
            kind="posts",
            query="测试",
            sort="latest",
            limit=60,
        )
        assert [item["ref"] for item in continued["items"]] == ["forum:42/3"]
        conflict = await tools.forum_search(
            cursor=first["next_cursor"], query="另一个查询"
        )
        assert conflict["code"] == "cursor_conflict"
    finally:
        current_turn.reset(token)


async def _forum_read_cursor_accepts_matching_topic_locator():
    model = SimpleNamespace(
        read_topic_post_page=AsyncMock(
            side_effect=[
                ("Topic", [_post(number=1)], 1, True),
                ("Topic", [_post(post_id=11, number=2)], 2, False),
            ]
        )
    )
    token = current_turn.set(TurnResults())
    try:
        tools = ShuiyuanToolsWrapper(model)
        content, _ = await tools.forum_read(topic_id=42, order="oldest")
        cursor = json.loads(content)["next_cursor"]
        continued, _ = await tools.forum_read(
            cursor=cursor, topic_id=42, order="oldest", limit=60
        )
        assert json.loads(continued)["items"][0]["ref"] == "forum:42/2"
        assert model.read_topic_post_page.await_args.kwargs["limit"] == 20
        conflict, _ = await tools.forum_read(cursor=cursor, topic_id=43)
        assert json.loads(conflict)["code"] == "cursor_conflict"
    finally:
        current_turn.reset(token)


async def _dates_report_the_supported_format_and_topic_read_path():
    tools = ShuiyuanToolsWrapper(SimpleNamespace(search_forum=AsyncMock()))
    result = await tools.forum_search(topic_id=42, after_date="2026-09-14T16:03:20")
    assert result["code"] == "invalid_arguments"
    assert "YYYY-MM-DD" in result["message"]
    assert "forum_read" in result["message"]


async def _forum_read_attaches_images_only_for_exact_reads():
    exact = _post(image="upload://one.png")
    model = SimpleNamespace(
        get_post_details=AsyncMock(return_value=exact),
        read_topic_post_page=AsyncMock(
            return_value=("话题", [_post(post_id=11, number=4)], 1, False)
        ),
    )
    tools = ShuiyuanToolsWrapper(model)

    content, artifacts = await tools.forum_read(post_id=10)
    payload = json.loads(content)
    assert "media" not in payload["items"][0]
    assert artifacts == []

    content, artifacts = await tools.forum_read(post_id=10, images="auto")
    payload = json.loads(content)
    assert payload["items"][0]["media"][0]["url"] == "upload://one.png"
    assert artifacts[0].image_urls == ["upload://one.png"]

    content, artifacts = await tools.forum_read(topic_id=42)
    assert json.loads(content)["topic"] == "话题"
    assert artifacts == []


async def _users_preserves_batch_order_and_item_status():
    async def get_user(name):
        if name.casefold() == "missing":
            return None
        return SimpleNamespace(
            id=1,
            username=name.strip("@"),
            name=None,
            avatar_template="/user_avatar/site/name/{size}/1.png",
        )

    tools = ShuiyuanToolsWrapper(
        SimpleNamespace(get_user_by_username=AsyncMock(side_effect=get_user))
    )
    token = current_turn.set(TurnResults())
    try:
        result = await tools.users(
            usernames=["Alice", "@Alice", "missing"], include_avatar=True
        )
    finally:
        current_turn.reset(token)
    assert result["status"] == "partial"
    assert [item["input"] for item in result["items"]] == [
        "Alice",
        "@Alice",
        "missing",
    ]
    assert result["items"][0]["avatar"].startswith("https://")
    assert result["items"][2]["code"] == "not_found"


def test_forum_search_uses_native_snippets_and_opaque_cursor():
    asyncio.run(_forum_search_uses_native_snippets_and_opaque_cursor())


def test_forum_search_rejects_conflicting_structured_filter():
    asyncio.run(_forum_search_rejects_conflicting_structured_filter())


def test_cursor_accepts_matching_arguments_and_rejects_conflicts():
    asyncio.run(_cursor_accepts_matching_arguments_and_rejects_conflicts())


def test_forum_read_cursor_accepts_matching_topic_locator():
    asyncio.run(_forum_read_cursor_accepts_matching_topic_locator())


def test_dates_report_the_supported_format_and_topic_read_path():
    asyncio.run(_dates_report_the_supported_format_and_topic_read_path())


def test_forum_read_attaches_images_only_for_exact_reads():
    asyncio.run(_forum_read_attaches_images_only_for_exact_reads())


def test_users_preserves_batch_order_and_item_status():
    asyncio.run(_users_preserves_batch_order_and_item_status())


def test_legacy_tool_names_map_to_unified_tools():
    names = [
        "search_posts",
        "search_user_by_id",
        "get_users",
        "prepare_image_references",
    ]
    assert migrate_tool_names(names) == ["forum_search", "users", "generate_image"]

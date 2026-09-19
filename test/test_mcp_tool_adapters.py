import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.features.mention.tools_runtime import mcp_text_content


def _model():
    model = MentionChatModel.__new__(MentionChatModel)
    model._web_search_kinds = {"text", "news", "images"}
    return model


def test_mcp_text_content_unwraps_langchain_blocks():
    assert (
        mcp_text_content(
            [
                {"type": "text", "text": "first", "id": "lc_1"},
                SimpleNamespace(type="text", text="second"),
            ]
        )
        == "first\nsecond"
    )


def test_web_search_decodes_mcp_text_block_json():
    upstream = SimpleNamespace(
        name="web_search",
        ainvoke=AsyncMock(
            return_value=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "results": [
                                {
                                    "title": "Example",
                                    "url": "https://example.com/article",
                                    "snippet": "Useful result",
                                }
                            ]
                        }
                    ),
                    "id": "lc_search",
                }
            ]
        ),
    )
    tool = _model()._consolidate_mcp_tools([upstream])[0]

    result = asyncio.run(tool.coroutine(query="example", max_results=5))

    assert result == {
        "status": "ok",
        "items": [
            {
                "ref": "https://example.com/article",
                "url": "https://example.com/article",
                "title": "Example",
                "text": "Useful result",
            }
        ],
    }


def test_web_read_decodes_legacy_mcp_text_block_without_repr_noise():
    upstream = SimpleNamespace(
        name="fetch_webpage_content",
        ainvoke=AsyncMock(
            return_value=[
                {
                    "type": "text",
                    "text": "clean page body",
                    "id": "lc_fetch",
                }
            ]
        ),
    )
    tool = _model()._consolidate_mcp_tools([upstream])[0]

    content, artifact = asyncio.run(
        tool.coroutine(url="https://example.com", max_length=6000, images="none")
    )

    assert json.loads(content)["items"][0]["content"] == "clean page body"
    assert "lc_fetch" not in content
    assert artifact is None


def test_web_read_direct_image_is_loaded_only_when_requested():
    upstream = SimpleNamespace(name="fetch_webpage_content", ainvoke=AsyncMock())
    tool = _model()._consolidate_mcp_tools([upstream])[0]

    content, artifact = asyncio.run(tool.coroutine(url="https://example.com/a.png"))
    assert json.loads(content)["items"][0]["media"][0]["loaded"] is False
    assert artifact is None

    content, artifact = asyncio.run(
        tool.coroutine(url="https://example.com/a.png", images="auto")
    )
    assert json.loads(content)["items"][0]["media"][0]["loaded"] is True
    assert artifact.image_urls == ["https://example.com/a.png"]


def test_web_read_uses_structured_envelope_and_exact_cursor():
    first = {
        "status": "ok",
        "url": "https://example.com/menu",
        "content_type": "application/json",
        "mode": "json",
        "content": '{"items":[{"name":"香草冰淇淋"}]}',
        "total_chars": 40,
        "start_index": 0,
        "truncated": True,
        "next_start_index": 37,
        "matched_count": 1,
        "warnings": [],
    }
    second = {
        **first,
        "content": "end",
        "start_index": 37,
        "truncated": False,
        "next_start_index": None,
    }
    upstream = SimpleNamespace(
        name="fetch_webpage_content",
        ainvoke=AsyncMock(
            side_effect=[
                [{"type": "text", "text": json.dumps(first)}],
                [{"type": "text", "text": json.dumps(second)}],
            ]
        ),
    )
    tool = _model()._consolidate_mcp_tools([upstream])[0]
    token = current_turn.set(TurnResults())
    try:
        content, _ = asyncio.run(
            tool.coroutine(
                url=first["url"],
                query="冰淇淋",
                json_path="dataList",
                fields=["name"],
                max_results=10,
            )
        )
        payload = json.loads(content)
        cursor = payload["next_cursor"]
        assert payload["items"][0] == {
            "ref": first["url"],
            "url": first["url"],
            "content": first["content"],
            "page_start": 0,
        }

        continued, _ = asyncio.run(
            tool.coroutine(url=first["url"], cursor=cursor, max_length=12000)
        )

        assert json.loads(continued)["items"][0]["page_start"] == 37
        continued_args = upstream.ainvoke.call_args_list[-1].args[0]
        assert continued_args["start_index"] == 37
        assert continued_args["query"] == "冰淇淋"
    finally:
        current_turn.reset(token)


def test_web_read_rejects_conflicting_cursor_options():
    envelope = {
        "status": "ok",
        "url": "https://example.com",
        "content_type": "text/plain",
        "mode": "document",
        "content": "page",
        "total_chars": 10,
        "start_index": 0,
        "truncated": True,
        "next_start_index": 4,
        "warnings": [],
    }
    upstream = SimpleNamespace(
        name="fetch_webpage_content",
        ainvoke=AsyncMock(
            return_value=[{"type": "text", "text": json.dumps(envelope)}]
        ),
    )
    tool = _model()._consolidate_mcp_tools([upstream])[0]
    token = current_turn.set(TurnResults())
    try:
        content, _ = asyncio.run(tool.coroutine(url=envelope["url"], query="first"))
        cursor = json.loads(content)["next_cursor"]

        conflict, _ = asyncio.run(tool.coroutine(cursor=cursor, query="different"))

        assert json.loads(conflict)["code"] == "invalid_arguments"
    finally:
        current_turn.reset(token)


def test_web_read_pages_are_distinct_full_evidence_for_finalizer():
    turn = TurnResults()
    first = {
        "status": "ok",
        "items": [
            {
                "ref": "https://example.com/long",
                "url": "https://example.com/long",
                "content": "first verified page",
                "page_start": 0,
            }
        ],
    }
    second = {
        "status": "ok",
        "items": [
            {
                "ref": "https://example.com/long",
                "url": "https://example.com/long",
                "content": "second verified page",
                "page_start": 6000,
            }
        ],
    }

    assert len(turn.observe(json.dumps(first), tool="web_read")) == 1
    assert len(turn.observe(json.dumps(second), tool="web_read")) == 1
    assert all(item["kind"] == "full" for item in turn.evidence.values())
    final = turn.final_evidence_text(6000)
    assert "first verified page" in final
    assert "second verified page" in final


def test_web_read_changed_content_at_same_offset_is_new_evidence():
    turn = TurnResults()

    def page(content):
        return {
            "status": "ok",
            "items": [
                {
                    "ref": "https://example.com/live",
                    "url": "https://example.com/live",
                    "content": content,
                    "page_start": 0,
                }
            ],
        }

    assert len(turn.observe(json.dumps(page("old content")), tool="web_read")) == 1
    assert len(turn.observe(json.dumps(page("new content")), tool="web_read")) == 1
    assert len(turn.observe(json.dumps(page("new content")), tool="web_read")) == 0


def test_chuangka_menu_decodes_envelope_and_uses_exact_cursor():
    source = "https://m.yk.fkw.com/api/product/list?aid=32677668&page=1"
    first = {
        "status": "ok",
        "fetched_at": "2026-09-16T12:00:00+08:00",
        "location": "zhutu",
        "category": "ice_cream",
        "query": None,
        "total_products": 188,
        "total_by_location": {"zhutu": 188},
        "matched_count": 9,
        "content": "# 创咖当前冰淇淋菜单\n- 香草圣代｜¥2.50",
        "start_index": 0,
        "truncated": True,
        "next_start_index": 24,
        "source_urls": [source],
        "failed_locations": [],
        "warnings": [],
    }
    second = {
        **first,
        "content": "- 香草吐冰｜¥3.50",
        "start_index": 24,
        "truncated": False,
        "next_start_index": None,
    }
    upstream = SimpleNamespace(
        name="get_chuangka_menu",
        ainvoke=AsyncMock(
            side_effect=[
                [{"type": "text", "text": json.dumps(first, ensure_ascii=False)}],
                [{"type": "text", "text": json.dumps(second, ensure_ascii=False)}],
            ]
        ),
    )
    tool = next(
        item
        for item in _model()._consolidate_mcp_tools([upstream])
        if item.name == "get_chuangka_menu"
    )
    token = current_turn.set(TurnResults())
    try:
        result = asyncio.run(tool.coroutine(location="zhutu", category="ice_cream"))
        assert result["items"][0] == {
            "ref": source,
            "url": source,
            "content": first["content"],
            "page_start": 0,
            "source_urls": [source],
        }
        cursor = result["next_cursor"]

        continued = asyncio.run(tool.coroutine(cursor=cursor, max_length=12000))

        assert continued["items"][0]["page_start"] == 24
        continued_args = upstream.ainvoke.call_args_list[-1].args[0]
        assert continued_args == {
            "location": "zhutu",
            "category": "ice_cream",
            "query": None,
            "max_length": 12000,
            "start_index": 24,
        }
    finally:
        current_turn.reset(token)


def test_chuangka_menu_is_full_final_evidence():
    turn = TurnResults()
    payload = {
        "status": "ok",
        "items": [
            {
                "ref": "https://m.yk.fkw.com/api/product/list?aid=32677668",
                "url": "https://m.yk.fkw.com/api/product/list?aid=32677668",
                "content": "- 香草圣代｜¥2.50",
                "page_start": 0,
            }
        ],
    }

    added = turn.observe(json.dumps(payload), tool="get_chuangka_menu")

    assert len(added) == 1
    evidence = turn.evidence[next(iter(added))]
    assert evidence["kind"] == "full"
    assert "香草圣代" in turn.final_evidence_text(1000)


def test_web_read_refuses_forum_urls_with_a_hint():
    upstream = SimpleNamespace(name="fetch_webpage_content", ainvoke=AsyncMock())
    tool = _model()._consolidate_mcp_tools([upstream])[0]

    content, artifact = asyncio.run(
        tool.coroutine(
            url="https://shuiyuan.sjtu.edu.cn/secure-uploads/original/4X/a/b.jpeg"
        )
    )

    payload = json.loads(content)
    assert payload["status"] == "error"
    assert payload["code"] == "use_forum_tools"
    assert "forum_read" in payload["hint"]
    assert artifact is None
    upstream.ainvoke.assert_not_awaited()


def test_web_read_carries_page_title_and_date():
    envelope = {
        "status": "ok",
        "url": "https://example.com/post",
        "content_type": "text/html",
        "mode": "document",
        "content": "# 标题\n\n正文",
        "total_chars": 8,
        "start_index": 0,
        "truncated": False,
        "next_start_index": None,
        "title": "标题",
        "published_at": "2026-09-01T08:00:00+08:00",
        "warnings": [],
    }
    upstream = SimpleNamespace(
        name="fetch_webpage_content",
        ainvoke=AsyncMock(
            return_value=[{"type": "text", "text": json.dumps(envelope)}]
        ),
    )
    tool = _model()._consolidate_mcp_tools([upstream])[0]

    content, _ = asyncio.run(tool.coroutine(url=envelope["url"]))

    item = json.loads(content)["items"][0]
    assert item["title"] == "标题"
    assert item["published_at"] == "2026-09-01T08:00:00+08:00"
    assert item["content"].startswith("# 标题")

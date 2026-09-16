import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from shuiyuan_auto_reply.features.mention.mention_chat_model import (
    MentionChatModel,
    mcp_text_content,
)


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

    assert json.loads(content)["items"][0]["text"] == "clean page body"
    assert "lc_fetch" not in content
    assert artifact is None

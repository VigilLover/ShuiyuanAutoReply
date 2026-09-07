import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from shuiyuan_auto_reply.infrastructure.embedding.openai import OpenAIEmbeddings


def make():
    client = Mock()
    async_client = Mock()
    settings = dict(
        api_key="fake",
        base_url="http://invalid",
        timeout=1,
        concurrency=2,
        dims=2,
        model="fake",
        batch_size=2,
        attempts=1,
    )
    return OpenAIEmbeddings(settings, client=client, async_client=async_client)


def test_order_and_dimensions():
    model = make()
    result = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[1.0, 0.0]),
            SimpleNamespace(index=0, embedding=[0.0, 1.0]),
        ]
    )
    assert model._validate(result, 2) == [[0.0, 1.0], [1.0, 0.0]]
    with pytest.raises(ValueError):
        model._validate(result, 1)
    result.data[0].embedding = [float("nan"), 0]
    with pytest.raises(ValueError):
        model._validate(result, 2)


def test_async_batches_raw_text():
    model = make()

    async def create(**kwargs):
        assert all(isinstance(text, str) for text in kwargs["input"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[1.0, 0.0])
                for i, _ in enumerate(kwargs["input"])
            ]
        )

    model.async_client.embeddings.create = AsyncMock(side_effect=create)
    assert len(asyncio.run(model.aembed_documents(["a", "b", "c"]))) == 3
    assert model.async_client.embeddings.create.call_count == 2


def test_retry_after_and_authentication_failure():
    from unittest.mock import patch

    import httpx
    from openai import APIStatusError

    model = make()
    model.settings["attempts"] = 3
    request = httpx.Request("POST", "https://example.com/v1/embeddings")
    limited = APIStatusError(
        "limited",
        response=httpx.Response(429, request=request, headers={"Retry-After": "2"}),
        body=None,
    )
    ok = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])

    async def run():
        model.async_client.embeddings.create = AsyncMock(side_effect=[limited, ok])
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            assert await model.aembed_query("hello") == [1.0, 0.0]
            sleep.assert_awaited_once_with(2)
        denied = APIStatusError(
            "denied", response=httpx.Response(401, request=request), body=None
        )
        model.async_client.embeddings.create = AsyncMock(side_effect=denied)
        with pytest.raises(APIStatusError):
            await model.aembed_query("hello")
        assert model.async_client.embeddings.create.await_count == 1

    asyncio.run(run())

import asyncio
import importlib.util
import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def module():
    pytest.importorskip("markdownify")
    path = Path(__file__).resolve().parents[1] / "deploy/mcp/fetch.py"
    spec = importlib.util.spec_from_file_location("bounded_fetch", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_literal_addresses_and_ports():
    fetch = module()
    for url in [
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "https://example.com:8080/",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
    ]:
        with pytest.raises(ValueError):
            fetch.validate_url(url)
    fetch.validate_url("https://example.com/path")


def test_dns_validated_at_connection():
    fetch = module()

    async def run():
        resolver = fetch.PublicResolver()
        loop = asyncio.get_running_loop()
        with patch.object(
            loop,
            "getaddrinfo",
            AsyncMock(
                return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
                ]
            ),
        ):
            with pytest.raises(ValueError):
                await resolver.resolve("example.com", 443)

    asyncio.run(run())


def test_chunked_download_limit_and_redirect_policy():
    fetch = module()
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    class Response:
        def __init__(self, *, status=200, headers=None, blocks=()):
            self.status = status
            self.headers = headers or {}
            self.content_length = None
            self.charset = "utf-8"

            async def chunks(size):
                for block in blocks:
                    yield block

            self.content = SimpleNamespace(iter_chunked=chunks)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

    class Client:
        def __init__(self, responses):
            self.responses = iter(responses)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return next(self.responses)

    async def run():
        with patch.object(fetch.aiohttp, "TCPConnector", return_value=object()):
            with patch.object(
                fetch.aiohttp,
                "ClientSession",
                return_value=Client([Response(blocks=[b"x" * fetch.MAX_BYTES, b"x"])]),
            ):
                assert (
                    await fetch.fetch_webpage_content("https://example.com")
                    == "Fetch failed: ValueError"
                )
            with patch.object(
                fetch.aiohttp,
                "ClientSession",
                return_value=Client(
                    [
                        Response(
                            status=302, headers={"Location": "http://169.254.169.254/"}
                        )
                    ]
                ),
            ):
                assert (
                    await fetch.fetch_webpage_content("https://example.com")
                    == "Fetch failed: ValueError"
                )
            with patch.object(
                fetch.aiohttp,
                "ClientSession",
                return_value=Client(
                    [
                        Response(
                            headers={"Content-Type": "text/plain"}, blocks=[b"abcdef"]
                        )
                    ]
                ),
            ):
                assert (
                    await fetch.fetch_webpage_content(
                        "https://example.com", max_length=3, start_index=2
                    )
                    == "cde"
                )

    asyncio.run(run())

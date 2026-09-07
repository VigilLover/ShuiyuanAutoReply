"""Bounded public-only fetch; DNS validation happens in the actual connector."""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from bs4 import BeautifulSoup
from markdownify import markdownify

MAX_BYTES = 5 * 1024 * 1024
_gate = asyncio.Semaphore(1)


def public_address(address):
    value = ipaddress.ip_address(address)
    return value.is_global and not (
        getattr(value, "ipv4_mapped", None) and not value.ipv4_mapped.is_global
    )


def validate_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Only public HTTP(S) URLs without credentials are allowed")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Only ports 80 and 443 are allowed")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return
    if not public_address(str(address)):
        raise ValueError("Non-public address rejected")


class PublicResolver(AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        rows = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM, family=family
        )
        if not rows or any(not public_address(row[4][0]) for row in rows):
            raise ValueError("DNS resolved to a non-public address")
        return [
            dict(
                hostname=host,
                host=row[4][0],
                port=port,
                family=row[0],
                proto=row[2],
                flags=socket.AI_NUMERICHOST,
            )
            for row in rows
        ]

    async def close(self):
        pass


def convert_page(content):
    soup = BeautifulSoup(content, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return markdownify(str(soup), heading_style="ATX")


async def fetch_webpage_content(
    url: str, max_length: int = 5000, start_index: int = 0, raw: bool = False
) -> str:
    """Read a public page as Markdown with a hard 5 MiB download limit."""
    if not 1 <= max_length < 1_000_000 or start_index < 0:
        return "Invalid max_length or start_index"
    async with _gate:
        try:
            async with asyncio.timeout(30):
                connector = aiohttp.TCPConnector(
                    resolver=PublicResolver(), use_dns_cache=False, limit=1
                )
                async with aiohttp.ClientSession(
                    connector=connector,
                    trust_env=False,
                    auto_decompress=False,
                    timeout=aiohttp.ClientTimeout(total=25),
                    headers={
                        "Accept-Encoding": "identity",
                        "User-Agent": "ShuiyuanMCP/1.0",
                    },
                ) as client:
                    for redirect in range(6):
                        validate_url(url)
                        async with client.get(url, allow_redirects=False) as response:
                            if response.status in {301, 302, 303, 307, 308}:
                                if redirect == 5 or not response.headers.get(
                                    "Location"
                                ):
                                    raise ValueError(
                                        "Too many redirects or missing Location"
                                    )
                                url = urljoin(url, response.headers["Location"])
                                continue
                            response.raise_for_status()
                            if response.headers.get(
                                "Content-Encoding", "identity"
                            ).lower() not in {"identity", ""}:
                                raise ValueError(
                                    "Compressed responses are rejected by the bounded fetch policy"
                                )
                            if (
                                response.content_length is not None
                                and response.content_length > MAX_BYTES
                            ):
                                raise ValueError("Response exceeds 5 MiB")
                            data = bytearray()
                            async for block in response.content.iter_chunked(16384):
                                if len(data) + len(block) > MAX_BYTES:
                                    raise ValueError("Response exceeds 5 MiB")
                                data.extend(block)
                            content = data.decode(
                                response.charset or "utf-8", errors="replace"
                            )
                            if not raw and "html" in response.headers.get(
                                "Content-Type", ""
                            ):
                                content = await asyncio.to_thread(convert_page, content)
                            return content[start_index : start_index + max_length]
                return "No response"
        except (aiohttp.ClientError, ValueError, TimeoutError, LookupError) as exc:
            # Return type and fixed policy messages only; never upstream response headers.
            return f"Fetch failed: {type(exc).__name__}"

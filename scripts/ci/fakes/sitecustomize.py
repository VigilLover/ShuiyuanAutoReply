"""Test-only HTTP transport redirect; mounted only in isolated integration containers."""

from urllib.parse import urlsplit

import aiohttp

_original = aiohttp.ClientSession._request


async def _request(self, method, url, **kwargs):
    parsed = urlsplit(str(url))
    if parsed.hostname == "shuiyuan.sjtu.edu.cn":
        url = "http://fake:8080" + parsed.path
        if parsed.query:
            url += "?" + parsed.query
    return await _original(self, method, url, **kwargs)


aiohttp.ClientSession._request = _request

"""Deployment entry point reusing pinned upstream tools without uvx subprocesses."""

import asyncio
import functools
import inspect

from fetch import fetch_webpage_content
from mcp.server.fastmcp import FastMCP
from tools.hardware_status import get_hardware_status
from tools.system_time import get_system_time
from tools.web_search import image_search, web_search

_gate = asyncio.Semaphore(2)


def bounded(function):
    @functools.wraps(function)
    async def invoke(*args, **kwargs):
        async with _gate:
            async with asyncio.timeout(30):
                if inspect.iscoroutinefunction(function):
                    return await function(*args, **kwargs)
                return await asyncio.to_thread(function, *args, **kwargs)

    return invoke


def main():
    server = FastMCP("AutoReplyToolServer", host="0.0.0.0", port=58000)
    for function in (
        get_system_time,
        get_hardware_status,
        web_search,
        image_search,
        fetch_webpage_content,
    ):
        server.tool()(bounded(function))
    server.run("sse")


if __name__ == "__main__":
    main()

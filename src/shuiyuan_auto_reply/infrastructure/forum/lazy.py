"""Allow management startup while community login is unavailable."""

import asyncio

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel


class LazyForum:
    def __init__(self, cookie_file):
        self.cookie_file = cookie_file
        self.instance = None
        self.lock = asyncio.Lock()

    async def _get(self):
        async with self.lock:
            if self.instance is None:
                self.instance = await ShuiyuanModel.create(self.cookie_file)
            return self.instance

    def __getattr__(self, name):
        async def invoke(*args, **kwargs):
            return await getattr(await self._get(), name)(*args, **kwargs)

        return invoke

    async def close(self):
        if self.instance is not None:
            await self.instance.close()


class LazyChat:
    def __init__(self, factory):
        self.factory = factory
        self.instance = None

    def __getattr__(self, name):
        if self.instance is None:
            self.instance = self.factory()
        return getattr(self.instance, name)

    async def aclose(self):
        if self.instance is not None:
            await self.instance.aclose()

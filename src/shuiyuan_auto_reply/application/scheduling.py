"""One bounded, conversation-ordered scheduler shared by all channels."""

import asyncio
import contextvars
import weakref
from contextlib import asynccontextmanager

_held = contextvars.ContextVar("scheduler_held", default=False)
_schedulers = weakref.WeakKeyDictionary()


class BusyError(RuntimeError):
    pass


class ReplyScheduler:
    def __init__(self, concurrency=3, queue_limit=100, timeout=900):
        self.slots = asyncio.Semaphore(concurrency)
        self.queue_limit = queue_limit
        self.timeout = timeout
        self.waiting = 0
        self.active = 0
        self.locks = {}

    @asynccontextmanager
    async def admission(self, key):
        if _held.get():
            yield
            return
        if self.waiting >= self.queue_limit:
            raise BusyError("Reply queue is full; try again later")
        self.waiting += 1
        entry = self.locks.setdefault(key, [asyncio.Lock(), 0])
        entry[1] += 1
        started = False
        try:
            async with entry[0]:
                async with self.slots:
                    self.waiting -= 1
                    started = True
                    self.active += 1
                    token = _held.set(True)
                    try:
                        async with asyncio.timeout(self.timeout):
                            yield
                    finally:
                        _held.reset(token)
                        self.active -= 1
        finally:
            if not started:
                self.waiting -= 1
            entry[1] -= 1
            if entry[1] == 0:
                self.locks.pop(key, None)


def get_scheduler():
    loop = asyncio.get_running_loop()
    if loop not in _schedulers:
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

        config = get_deployment().section("runtime")
        _schedulers[loop] = ReplyScheduler(
            config["concurrency"], config["queue_limit"], config["timeout"]
        )
    return _schedulers[loop]

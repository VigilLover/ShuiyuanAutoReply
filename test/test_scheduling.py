import asyncio

import pytest

from shuiyuan_auto_reply.application.scheduling import BusyError, ReplyScheduler


def test_parallel_conversations_and_serial_order():
    async def run():
        scheduler = ReplyScheduler(3, 100, 2)
        active = maximum = 0
        order = []

        async def work(key, number):
            nonlocal active, maximum
            async with scheduler.admission(key):
                active += 1
                maximum = max(maximum, active)
                order.append(number)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(work("a", 1), work("a", 2), work("b", 3), work("c", 4))
        assert maximum == 3
        assert order.index(1) < order.index(2)
        assert scheduler.waiting == scheduler.active == 0
        assert not scheduler.locks

    asyncio.run(run())


def test_timeout_releases_capacity():
    async def run():
        scheduler = ReplyScheduler(1, 1, 0.01)
        with pytest.raises(TimeoutError):
            async with scheduler.admission("a"):
                await asyncio.sleep(1)
        async with scheduler.admission("a"):
            pass
        assert not scheduler.locks

    asyncio.run(run())


def test_queue_recovers_uncertain_publication(tmp_path):
    async def run():
        import aiosqlite

        from shuiyuan_auto_reply.infrastructure.persistence.work_queue import ForumQueue

        queue = ForumQueue(tmp_path / "state.sqlite3", "bot")
        await queue.initialize()
        db = await queue.connect()
        try:
            await db.execute(
                "INSERT INTO forum_jobs VALUES ('bot',1,'{}','sending',NULL,0)"
            )
            await db.execute(
                "INSERT INTO forum_jobs VALUES ('bot',2,'{}','running',NULL,0)"
            )
            await db.commit()
        finally:
            await db.close()
        await queue.initialize()
        assert await queue.state(1) == "needs_review"
        assert await queue.state(2) == "pending"

    asyncio.run(run())

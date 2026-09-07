"""Read-only health views, independent of community login and chat admission."""

import asyncio
import time
from pathlib import Path

from shuiyuan_auto_reply.bootstrap.deployment import get_deployment


async def runtime_health(store):
    result = {"process": "ok", "database": "unknown", "forum": "unknown", "jobs": {}}
    try:
        db = await store._connect()
        try:
            tables = await (
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE name='forum_cursor'"
                )
            ).fetchone()
            if tables:
                row = await (
                    await db.execute(
                        "SELECT last_poll FROM forum_cursor WHERE username=?",
                        (get_deployment().section("forum")["bot_username"],),
                    )
                ).fetchone()
                result["last_poll"] = row[0] if row else None
                result["forum"] = "ok" if row and time.time() - row[0] < 90 else "stale"
                rows = await (
                    await db.execute(
                        "SELECT status,count(*) FROM forum_jobs GROUP BY status"
                    )
                ).fetchall()
                result["jobs"] = {row[0]: row[1] for row in rows}
        finally:
            await db.close()
    except Exception:
        result["state"] = "unavailable"
    if get_deployment().profile == "remote":
        from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
            check_vector_space,
            engine_for,
        )

        engine = engine_for()
        try:
            async with asyncio.timeout(3):
                async with engine.connect() as connection:
                    await check_vector_space(connection)
            result["database"] = "ok"
        except Exception:
            result["database"] = "unavailable"
        finally:
            await engine.dispose()
    else:
        result["database"] = "not_probed"
    return result

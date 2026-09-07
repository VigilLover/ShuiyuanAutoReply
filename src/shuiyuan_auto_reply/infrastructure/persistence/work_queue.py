"""Single-worker-process durable forum inbox and publication uncertainty tracking."""

import contextvars
import json
import time
from dataclasses import asdict

import aiosqlite

_current_job = contextvars.ContextVar("forum_job", default=None)


async def publication_status(status, reply_id=None):
    job = _current_job.get()
    if job:
        queue, post_id = job
        await queue.status(post_id, status, reply_id)


class ForumQueue:
    def __init__(self, path, username):
        self.path, self.username = path, username

    async def connect(self):
        db = await aiosqlite.connect(self.path)
        await db.execute("PRAGMA busy_timeout=5000")
        return db

    async def initialize(self):
        db = await self.connect()
        try:
            await db.executescript(
                """CREATE TABLE IF NOT EXISTS forum_jobs (
                username TEXT NOT NULL, post_id INTEGER NOT NULL, payload TEXT NOT NULL,
                status TEXT NOT NULL, reply_id INTEGER, updated REAL NOT NULL,
                PRIMARY KEY(username,post_id));
                CREATE TABLE IF NOT EXISTS forum_cursor (username TEXT PRIMARY KEY, post_id INTEGER, last_poll REAL);"""
            )
            await db.execute(
                "UPDATE forum_jobs SET status=CASE WHEN status='sending' THEN 'needs_review' ELSE 'pending' END WHERE username=? AND status IN ('running','sending')",
                (self.username,),
            )
            await db.commit()
        finally:
            await db.close()

    async def cursor(self):
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    "SELECT post_id FROM forum_cursor WHERE username=?",
                    (self.username,),
                )
            ).fetchone()
            return row[0] if row else None
        finally:
            await db.close()

    async def enqueue(self, actions, cursor):
        db = await self.connect()
        try:
            await db.execute("BEGIN IMMEDIATE")
            for action in actions:
                await db.execute(
                    "INSERT OR IGNORE INTO forum_jobs VALUES (?,?,?,'pending',NULL,?)",
                    (
                        self.username,
                        action.post_id,
                        json.dumps(asdict(action)),
                        time.time(),
                    ),
                )
            await db.execute(
                "INSERT INTO forum_cursor VALUES (?,?,?) ON CONFLICT(username) DO UPDATE SET post_id=excluded.post_id,last_poll=excluded.last_poll",
                (self.username, cursor, time.time()),
            )
            await db.commit()
        finally:
            await db.close()

    async def pending(self):
        db = await self.connect()
        try:
            rows = await (
                await db.execute(
                    "SELECT post_id,payload FROM forum_jobs WHERE username=? AND status='pending' ORDER BY updated,post_id",
                    (self.username,),
                )
            ).fetchall()
            return rows
        finally:
            await db.close()

    async def status(self, post_id, status, reply_id=None):
        db = await self.connect()
        try:
            await db.execute(
                "UPDATE forum_jobs SET status=?,reply_id=COALESCE(?,reply_id),updated=? WHERE username=? AND post_id=?",
                (status, reply_id, time.time(), self.username, post_id),
            )
            await db.commit()
        finally:
            await db.close()

    async def state(self, post_id):
        db = await self.connect()
        try:
            row = await (
                await db.execute(
                    "SELECT status FROM forum_jobs WHERE username=? AND post_id=?",
                    (self.username, post_id),
                )
            ).fetchone()
            return row[0]
        finally:
            await db.close()

import asyncio
import logging
import random
import traceback
from abc import abstractmethod
from typing import List

from ..constants import settings
from .objects import UserActionDetails
from .shuiyuan_model import ShuiyuanModel


class BaseUserActionModel:
    """
    A class to represent a mention model.
    """

    def __init__(self, model: ShuiyuanModel, username: str, action_type: List[int]):
        """
        Initialize the MentionModel with a ShuiyuanModel instance.

        :param model: An instance of ShuiyuanModel.
        :param username: The username to be managed.
        :param action_type: The list of action types to monitor.
        """
        self.model = model
        self.username = username
        self.action_type = action_type
        self.stream_list = []
        self._bg_tasks = set()

    @staticmethod
    def _generate_random_string(length: int) -> str:
        """
        Generate a random string of a given length.

        :param length: The length of the random string to generate.
        :return: A random string of the specified length.
        """
        return "".join(
            random.sample(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                k=length,
            )
        )

    @staticmethod
    def _make_unique_reply(base: str) -> str:
        """
        Append a random string to the base reply to make it unique.

        :param base: The base reply string.
        :return: The unique reply string.
        """
        return (
            f"{base}\n\n"
            f"<!-- {BaseUserActionModel._generate_random_string(20)} -->\n"
            f"{settings.auto_reply_tag}"
        )

    @abstractmethod
    async def _new_action_routine(self, action: UserActionDetails) -> None:
        """
        A routine to handle new actions.
        NOTE: no exception should be raised in this method.

        :param action: The details of the action notification.
        :return: None
        """
        pass

    async def _prepare_action(self, action):
        return action

    async def _accept_action(self, prepared):
        return prepared

    async def _execute_action(self, prepared):
        await self._new_action_routine(prepared)

    async def _interrupt_action(self, prepared):
        pass

    async def _fail_action(self, prepared, error):
        pass

    async def watch_new_action_routine(self, interval: int = 1) -> None:
        """
        A routine to watch for new actions.
        """
        import json
        from collections import deque

        from dacite import from_dict

        from shuiyuan_auto_reply.application.scheduling import get_scheduler
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
        from shuiyuan_auto_reply.infrastructure.persistence.state import state_directory
        from shuiyuan_auto_reply.infrastructure.persistence.work_queue import (
            ForumQueue,
            _current_job,
        )

        config = get_deployment().section("runtime")
        queue = ForumQueue(state_directory() / "state.sqlite3", self.username)
        await queue.initialize()
        active = set()
        prechecks = asyncio.Semaphore(3)
        store = getattr(self, "state_store", None)
        if store is not None:
            await store.recover_forum_runs(self.username)

        async def execute(post_id, action):
            token = _current_job.set((queue, post_id))
            prepared = None
            try:
                # Prechecks stay bounded; the scheduler bounds execution.
                async with prechecks:
                    async with asyncio.timeout(config["timeout"]):
                        prepared = await self._prepare_action(action)
                if prepared is not None:
                    prepared = await self._accept_action(prepared)
                if prepared is None:
                    await queue.status(post_id, "done")
                    return
                # Key by post, not topic: mentions that arrive while an earlier
                # one is still unanswered are independent questions, so they may
                # be handled concurrently instead of queueing behind each other.
                async with get_scheduler().admission(
                    ("forum", self.username, action.post_id)
                ):
                    await queue.status(post_id, "running")
                    await self._execute_action(prepared)
                    if await queue.state(post_id) not in {"sent", "needs_review"}:
                        await queue.status(post_id, "done")
            except asyncio.CancelledError:
                state = await queue.state(post_id)
                await queue.status(
                    post_id,
                    (
                        "needs_review"
                        if state in {"sending", "needs_review"}
                        else "sent" if state == "sent" else "pending"
                    ),
                )
                raise
            except Exception as exc:
                if prepared is not None:
                    await self._fail_action(prepared, exc)
                state = await queue.state(post_id)
                if state not in {"sent", "needs_review"}:
                    await queue.status(
                        post_id, "needs_review" if state == "sending" else "failed"
                    )
                logging.exception("Forum job failed: %s", post_id)
            finally:
                try:
                    if prepared is not None:
                        await self._interrupt_action(prepared)
                finally:
                    _current_job.reset(token)
                    active.discard(post_id)

        while True:
            try:
                pending = await queue.pending()
                for post_id, payload in pending:
                    if post_id not in active and len(active) < config["queue_limit"]:
                        try:
                            action = from_dict(UserActionDetails, json.loads(payload))
                        except Exception:
                            await queue.status(post_id, "failed")
                            logging.exception(
                                "Invalid persisted forum action: %s", post_id
                            )
                            continue
                        active.add(post_id)
                        task = asyncio.create_task(execute(post_id, action))
                        self._bg_tasks.add(task)
                        task.add_done_callback(self._on_background_task_done)
                cursor = await queue.cursor()
                free = config["queue_limit"] - len(active)
                if free > 0:
                    offset = 0
                    collected = deque(maxlen=free)
                    while True:
                        page = (
                            await self.model.get_actions(
                                self.username, self.action_type, offset=offset
                            )
                        ).user_actions
                        if cursor is None:
                            await queue.enqueue([], page[0].post_id if page else 0)
                            break
                        stop = False
                        for action in page:
                            if action.post_id == cursor:
                                stop = True
                                break
                            collected.append(action)
                        if stop or not page:
                            ordered = list(reversed(collected))
                            await queue.enqueue(
                                ordered, ordered[-1].post_id if ordered else cursor
                            )
                            break
                        offset += len(page)
            except Exception:
                logging.exception("Forum polling failed")
            await asyncio.sleep(config["poll_interval"])

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logging.exception("User-action background task failed")

    async def aclose(self) -> None:
        """Cancel and observe every in-flight action before shutdown."""
        tasks = tuple(self._bg_tasks)
        if tasks:
            from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

            _, pending = await asyncio.wait(
                tasks, timeout=get_deployment().section("runtime")["shutdown_timeout"]
            )
        else:
            pending = ()
        for task in pending:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bg_tasks.clear()

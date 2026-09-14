import asyncio
import inspect
import json
from functools import wraps
from typing import List, Optional

from shuiyuan_auto_reply.application.tool_results import (
    cached_query,
    current_turn,
    tool_error,
)
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .shuiyuan_tools_objects import PostSearchResults, PostShort, UserShort


def cached_read(func):
    @wraps(func)
    async def wrapped(self, *args, **kwargs):
        bound = inspect.signature(func).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        refresh = values.pop("refresh", False)
        cursor = values.pop("cursor", 0)
        if cursor < 0 or any(
            values.get(name, 1) <= 0
            for name in ("post_id", "post_number")
            if values.get(name) is not None
        ):
            return {
                "status": "error",
                "error": "invalid_post_locator_or_cursor",
                "retryable": False,
            }
        for name in ("username", "term"):
            if isinstance(values.get(name), str):
                values[name] = values[name].strip()
        if func.__name__ == "get_post_by_id":
            key = "post:" + str(values["post_id"])
        elif func.__name__ == "get_post_details_by_post_number":
            key = f"post_number:{values['topic_id']}:{values['post_number']}"
        else:
            key = func.__name__ + json.dumps(values, sort_keys=True, ensure_ascii=False)
        turn = current_turn.get()
        old = turn.cache.get(key) if turn else None
        if refresh and isinstance(old, PostShort):
            turn.cache.pop(f"post:{old.id}", None)
            turn.cache.pop(f"post_number:{old.topic_id}:{old.post_number}", None)
        result = await cached_query(
            key, lambda: func(self, *args, **kwargs), refresh=refresh
        )
        if cursor and isinstance(result, PostShort):
            if cursor < 0 or cursor > result.to_dict()["total_chars"]:
                return {
                    "status": "error",
                    "error": "invalid_cursor",
                    "retryable": False,
                }
            return result.to_dict(cursor=cursor)
        return result

    return wrapped


class ShuiyuanToolsWrapper:
    """
    A wrapper around the ShuiyuanModel to provide tool functions for LLM agents.
    """

    def __init__(self, shuiyuan_model: ShuiyuanModel):
        self.shuiyuan_model = shuiyuan_model

    @cached_read
    async def search_user_by_term(
        self,
        term: str,
        include_avatar: bool = False,
        refresh: bool = False,
    ) -> List[UserShort] | str:
        """
        Search for users by a search term.

        :param term: The search term to use for finding users. It has to be NON-EMPTY.
        :param include_avatar: Whether to include each user's avatar. Default is False.
            Set to True only if the avatar is needed for image generation or editing.
        :return: A list of UserShort instances matching the search term or error message.
        """
        try:
            users = await self.shuiyuan_model.search_user_by_term(term)
            return [UserShort(user, include_avatar=include_avatar) for user in users]
        except Exception as e:
            return tool_error(e)

    @cached_read
    async def search_user_by_user_id(
        self,
        user_id: int,
        include_avatar: bool = False,
        refresh: bool = False,
    ) -> UserShort | None | str:
        """
        Resolve a user ID through their post history; no result does NOT prove the user is absent. Prefer get_user when the username is known.

        :param user_id: The ID of the user to search for.
        :param include_avatar: Whether to include each user's avatar. Default is False.
            Set to True only if the avatar is needed for image generation or editing.
        :return: An instance of UserShort for the user with the given ID or error message.
        """
        try:
            user = await self.shuiyuan_model.search_user_by_user_id(user_id)
            if user and include_avatar and user.avatar_template is None:
                full_user = await self.shuiyuan_model.get_user_by_username(
                    user.username
                )
                if full_user:
                    user = full_user
            return UserShort(user, include_avatar=include_avatar) if user else None
        except Exception as e:
            return tool_error(e)

    @cached_read
    async def search_post_details_by_optional_username_topic(
        self,
        term: str = "",
        latest: bool = False,
        username: Optional[str] = None,
        topic_id: Optional[int] = None,
        refresh: bool = False,
    ) -> List[PostShort] | str:
        """
        Search posts and return summaries (up to 800 characters each). Use get_post/get_post_by_id for full content. Results may not exhaust the topic; do not assume complete coverage. refresh=True explicitly bypasses cached results.

        :param term: Optional search term to use for finding posts. Default is empty.
        :param latest: Whether to sort the results by created_at in descending order. Default is False.
        :param username: An optional username to filter posts by. Default is None.
        :param topic_id: An optional topic ID to filter posts by. Default is None.
        :return: A list of PostShort instances matching the search criteria or error message.
        """
        try:
            posts_dict = await self.shuiyuan_model.search_post_details_by_optional_username_topic(
                term,
                latest,
                username,
                topic_id,
            )
            return PostSearchResults(
                [
                    PostShort(post, title)
                    for title, post_list in posts_dict.items()
                    for post in post_list
                ]
            )
        except Exception as e:
            return tool_error(e)

    @cached_read
    async def query_recent_posts_by_topic_id(
        self,
        topic_id: int,
        limit: int = 10,
        refresh: bool = False,
    ) -> List[PostShort] | str:
        """
        Read recent post summaries, up to 800 characters each. Use get_post for full text and reply relations. Start small; expand only for a specific information gap. refresh=True bypasses cached results.

        :param topic_id: The ID of the topic to query.
        :param limit: The maximum number of recent posts to retrieve. Default is 10.
        :return: A list of PostShort instances for the recent posts in the topic or error message.
        """
        try:
            title, posts = await self.shuiyuan_model.query_recent_posts_by_topic_id(
                topic_id, limit
            )
            return PostSearchResults([PostShort(post, title) for post in posts])
        except Exception as e:
            return tool_error(e)

    @cached_read
    async def get_post_details_by_post_number(
        self, topic_id: int, post_number: int, refresh: bool = False, cursor: int = 0
    ) -> PostShort | str:
        """
        Read full raw text by topic ID and topic-local floor number. Cursor pages contain 12000 characters; use next_cursor to continue, or read_tool_result with result_id. refresh=True explicitly bypasses cached results.
        If a user give you a url like "https://shuiyuan.sjtu.edu.cn/t/topic_id/post_number",
        you can extract the topic_id and post_number from the url and use this function to get the post details.
        Also, for any post you've retrieved using tool, if the `topic_id` and `reply_to_post_number` are both not None,
        you can use this function to get the details of the post being replied to.

        :param topic_id: The ID of the topic the post belongs to.
        :param post_number: The post number within the topic.
        :return: An instance of PostShort containing the post information or error message.
        """
        try:
            turn = current_turn.get()
            key = f"post_number:{topic_id}:{post_number}"
            if turn and not refresh and key in turn.cache:
                return turn.cache[key]
            post = await self.shuiyuan_model.get_post_details_by_post_number(
                topic_id,
                post_number,
            )
            if post.topic_id != topic_id or post.post_number != post_number:
                raise ValueError("Post identity mismatch")
            return await self._full_post(post, refresh=refresh)
        except Exception as e:
            return tool_error(e)

    @cached_read
    async def search_post_details_by_time_range_and_topic(
        self,
        topic_id: int,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        refresh: bool = False,
    ) -> List[PostShort] | str:
        """
        Search post summaries within a topic and date range; results may not exhaust the range. Use get_post for full text. refresh=True bypasses cached results.

        :param topic_id: The ID of the topic to search in.
        :param after_date: An optional start date (format: YYYY-MM-DD).
        :param before_date: An optional end date (format: YYYY-MM-DD).
        :return: A list of PostShort instances matching the criteria or error message.
        """
        try:
            posts_dict = (
                await self.shuiyuan_model.search_post_details_by_time_range_and_topic(
                    topic_id, after_date, before_date
                )
            )
            return PostSearchResults(
                [
                    PostShort(post, title)
                    for title, post_list in posts_dict.items()
                    for post in post_list
                ]
            )
        except Exception as e:
            return tool_error(e)

    async def _full_post(self, post, *, refresh=False, supplement=True):
        warnings = []
        if post.raw is None and supplement:
            try:
                original = post
                post = await self.shuiyuan_model.get_post_details(post.id)
                if (post.id, post.topic_id, post.post_number) != (
                    original.id,
                    original.topic_id,
                    original.post_number,
                ):
                    raise ValueError("Post identity mismatch")
            except Exception as exc:
                post = original
                warnings.append(tool_error(exc))
        result = PostShort(post, full=True)
        result.warnings = warnings
        turn = current_turn.get()
        if turn and not warnings and post.raw is not None:
            turn.cache["post:" + str(post.id)] = result
            turn.cache[f"post_number:{post.topic_id}:{post.post_number}"] = result
        return result

    @cached_read
    async def get_post_by_id(
        self, post_id: int, refresh: bool = False, cursor: int = 0
    ):
        """Read a complete post by GLOBAL post ID (not topic-local floor number).

        Returns raw text, reply relation, mentions and media. Long text includes
        result_id/next_cursor for read_tool_result or cursor on this tool. refresh bypasses this turn's cache.
        """
        try:
            turn = current_turn.get()
            if turn and not refresh and "post:" + str(post_id) in turn.cache:
                return turn.cache["post:" + str(post_id)]
            post = await self.shuiyuan_model.get_post_details(post_id)
            if post.id != post_id:
                raise ValueError("Post identity mismatch")
            return await self._full_post(post, refresh=refresh, supplement=False)
        except Exception as exc:
            return tool_error(exc)

    async def get_user(
        self, username: str, include_avatar: bool = False, refresh: bool = False
    ):
        """Get exactly one user by username. Never substitutes nickname/fuzzy matches.

        Set include_avatar only when needed. refresh requests current data again.
        Not found is an explicit error; use search_user only to resolve ambiguity.
        """
        username = username.strip().lstrip("@")

        async def fetch():
            try:
                if not username or any(c in username for c in "/?#"):
                    raise ValueError("Invalid username")
                user = await self.shuiyuan_model.get_user_by_username(username)
                if user is None:
                    return {
                        "status": "error",
                        "error": "not_found",
                        "username": username,
                        "retryable": False,
                    }
                if user.username.casefold() != username.casefold():
                    raise ValueError("User identity mismatch")
                value = UserShort(user, include_avatar=True)
                return {
                    "status": "ok",
                    "user_id": value.id,
                    "username": value.username,
                    "name": value.name,
                    "avatar": value.avatar,
                }
            except Exception as exc:
                return tool_error(exc)

        result = await cached_query(
            "user:" + username.casefold(), fetch, refresh=refresh
        )
        return (
            dict(result)
            if include_avatar
            else {k: v for k, v in result.items() if k != "avatar"}
        )

    async def get_users(
        self, usernames: list[str], include_avatar: bool = False, refresh: bool = False
    ):
        """Resolve up to 50 exact usernames, preserving input order and per-item errors.

        Prefer this to separate calls for a list of known usernames. Successful
        items are cached within this turn; retries only refetch failed items.
        """
        if not usernames or len(usernames) > 50:
            return {
                "status": "error",
                "error": "Provide 1 to 50 usernames",
                "retryable": False,
            }
        gate = asyncio.Semaphore(4)
        tasks = {}

        async def fetch(name):
            async with gate:
                return await self.get_user(name, include_avatar, refresh)

        for name in usernames:
            key = name.strip().lstrip("@").casefold()
            if key not in tasks:
                tasks[key] = asyncio.create_task(fetch(name))
        await asyncio.gather(*tasks.values())
        items = [
            {"input": name, **tasks[name.strip().lstrip("@").casefold()].result()}
            for name in usernames
        ]
        return {
            "status": "ok" if all(i["status"] == "ok" for i in items) else "partial",
            "items": items,
        }

    async def read_tool_result(self, result_id: str, cursor: int = 0):
        """Read the next 12000-character page of a result saved in this execution turn.

        Use the exact result_id and next_cursor returned by a tool or context summary.
        Results are not available in later turns.
        """
        turn = current_turn.get()
        return (
            turn.read(result_id, cursor)
            if turn
            else {"status": "error", "error": "no_active_turn"}
        )

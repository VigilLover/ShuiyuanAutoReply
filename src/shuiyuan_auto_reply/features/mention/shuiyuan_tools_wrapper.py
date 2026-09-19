import asyncio
import json
import re
import secrets
from datetime import date
from typing import Literal

from bs4 import BeautifulSoup

from shuiyuan_auto_reply.application.tool_results import (
    TurnResults,
    cached_query,
    current_turn,
    tool_error,
)
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_multimodal import extract_image_urls
from .shuiyuan_tools_objects import PostShort, UserShort


class CursorConflictError(ValueError):
    code = "cursor_conflict"
    retryable = False


class ShuiyuanToolsWrapper:
    """
    A wrapper around the ShuiyuanModel to provide tool functions for LLM agents.
    """

    def __init__(self, shuiyuan_model: ShuiyuanModel):
        self.shuiyuan_model = shuiyuan_model

    @staticmethod
    def _ok(items: list[dict], **metadata) -> dict:
        return {
            "status": "ok",
            "items": items,
            **{
                key: value
                for key, value in metadata.items()
                if value not in (None, "", [], {})
            },
        }

    @staticmethod
    def _error(exc: Exception) -> dict:
        value = tool_error(exc)
        code = value["error"]
        if getattr(exc, "code", None):
            code = exc.code
        elif isinstance(exc, ValueError):
            code = "invalid_arguments"
        hints = {
            "not_found": "确认 ref/楼层号或用户名是否正确；不要用相邻楼层猜测",
            "invalid_arguments": "按工具说明修正参数后重试一次；不要重复同样的调用",
            "cursor_conflict": "继续翻页时只传 cursor，不要再带其他筛选参数",
            "search_limit_reached": "换更具体的关键词或加 topic_id/username 缩小范围",
            "forbidden": "该内容当前账号不可见，忽略它继续作答",
        }
        result = {
            "status": "error",
            "code": code,
            "message": value["message"],
            "retryable": value["retryable"],
        }
        if code in hints:
            result["hint"] = hints[code]
        return result

    @staticmethod
    def _cursor(value: dict) -> str | None:
        turn = current_turn.get()
        if turn is None:
            return None
        token = "c_" + secrets.token_urlsafe(12)
        turn.cursors[token] = value
        return token

    @staticmethod
    def _resume(cursor: str, kind: str) -> dict:
        turn = current_turn.get()
        value = turn.cursors.get(cursor) if turn else None
        if not value or value.get("kind") != kind:
            raise ValueError("Unknown or incompatible cursor")
        return value

    @staticmethod
    def _check_cursor_arguments(
        state: dict, supplied: dict, *, defaults: dict | None = None
    ) -> None:
        """Accept repeated cursor locators when they match the bound request."""
        request = state.get("request", {})
        defaults = defaults or {}
        for key, value in supplied.items():
            if value in (None, "", []):
                continue
            if key in defaults and value == defaults[key] and request.get(key) != value:
                # Function defaults are indistinguishable from omitted arguments.
                continue
            expected = request.get(key)
            if key == "username" and isinstance(value, str):
                value = value.strip().lstrip("@").casefold()
                expected = str(expected or "").strip().lstrip("@").casefold()
            elif key == "query" and isinstance(value, str):
                value, expected = value.strip(), str(expected or "").strip()
            if value != expected:
                raise CursorConflictError(
                    f"cursor is bound to a different {key}; continue with the cursor alone"
                )

    @staticmethod
    def _search_query(
        query: str,
        *,
        topic_id: int | None,
        username: str | None,
        after_date: str | None,
        before_date: str | None,
        sort: str,
    ) -> str:
        query = query.strip()
        structured = {
            "topic": str(topic_id) if topic_id else None,
            "user": username.strip().lstrip("@") if username else None,
            "after": after_date,
            "before": before_date,
        }
        for value in (after_date, before_date):
            if value:
                try:
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(
                        "Dates must use YYYY-MM-DD, for example 2026-09-14; "
                        "use forum_read cursor for sequential topic reading"
                    ) from exc
        if after_date and before_date and after_date >= before_date:
            raise ValueError("after_date must be earlier than before_date")
        for name, wanted in structured.items():
            matches = re.findall(rf"(?<!\S){name}:([^\s]+)", query, re.I)
            if (
                wanted
                and matches
                and any(value.casefold() != wanted.casefold() for value in matches)
            ):
                raise ValueError(f"Conflicting {name} filter")
            if wanted and not matches:
                query += f" {name}:{wanted}"
        if sort != "relevance":
            query += f" order:{sort}"
        if not query.strip():
            raise ValueError("Provide a query or at least one filter")
        return query.strip()

    @staticmethod
    def _snippet(value: str, limit: int = 400) -> str:
        soup = BeautifulSoup(value or "", "html.parser")
        text = " ".join(soup.get_text(" ", strip=True).split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    @staticmethod
    def _text_page(content: str, offset: int, limit: int = 8000) -> tuple[str, int]:
        end = min(len(content), offset + limit)
        if end < len(content):
            boundary = content.rfind("\n\n", offset + limit // 2, end)
            if boundary < 0:
                boundary = content.rfind("\n", offset + limit // 2, end)
            if boundary >= 0:
                end = boundary
        return content[offset:end], end

    async def forum_search(
        self,
        kind: Literal["posts", "topics"] | None = None,
        query: str = "",
        topic_id: int | None = None,
        username: str | None = None,
        after_date: str | None = None,
        before_date: str | None = None,
        sort: Literal["relevance", "latest", "oldest"] = "relevance",
        limit: int = 8,
        cursor: str | None = None,
    ) -> dict:
        """搜索水源社区的帖子或话题。

        何时用：需要找到讨论某件事的帖子、某人说过的话，或按时间范围回顾时。
        参数要点：query 支持 Discourse 语法（`in:title`、`@用户名`、`#分类`）；
        topic_id 限定在一个话题内；username 限定作者；after_date/before_date 用
        YYYY-MM-DD；kind="topics" 只找话题标题；sort 默认按相关度。
        返回：items 为命中的帖子（ref、author、text 摘要≤400字、created_at、topic）
        或话题（ref、title、posts）；命中不保证穷尽，摘要可能截断，需要完整正文时按
        ref 用 forum_read 精读；有 next_cursor 时只传 cursor 继续翻页。
        """
        try:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            limit = min(limit, 20)
            if topic_id is not None and topic_id <= 0:
                raise ValueError("topic_id must be positive")
            if cursor:
                state = self._resume(cursor, "forum_search")
                self._check_cursor_arguments(
                    state,
                    {
                        "kind": kind,
                        "query": query,
                        "topic_id": topic_id,
                        "username": username,
                        "after_date": after_date,
                        "before_date": before_date,
                        "sort": sort,
                    },
                    defaults={"sort": "relevance"},
                )
                search_query = state["query"]
                page = state["page"]
                offset = state["offset"]
                kind = state["result_kind"]
            else:
                kind = kind or "posts"
                request = {
                    "kind": kind,
                    "query": query.strip(),
                    "topic_id": topic_id,
                    "username": username,
                    "after_date": after_date,
                    "before_date": before_date,
                    "sort": sort,
                }
                search_query = self._search_query(
                    query,
                    topic_id=topic_id,
                    username=username,
                    after_date=after_date,
                    before_date=before_date,
                    sort=sort,
                )
                page, offset = 1, 0
            turn = current_turn.get()
            scope = state.get("request", {}) if cursor else request
            completed_topic = TurnResults._topic_only_search(scope) if turn else None
            if completed_topic in (turn.completed_topics if turn else set()):
                return self._ok(
                    [],
                    topic=turn.topic_titles.get(completed_topic),
                    complete=True,
                )
            data = await cached_query(
                f"forum_search:{search_query}:{page}",
                lambda: self.shuiyuan_model.search_forum(search_query, page=page),
            )
            topics = {
                int(row["id"]): row for row in data.get("topics", []) if row.get("id")
            }
            if kind == "topics":
                source = list(topics.values())
                items = [
                    {
                        "ref": f"topic:{row['id']}",
                        "title": self._snippet(str(row.get("title", "")), 160),
                        "posts": row.get("posts_count"),
                        "replies": row.get("reply_count"),
                        "last_posted_at": row.get("last_posted_at"),
                    }
                    for row in source[offset : offset + limit]
                ]
            else:
                source = data.get("posts", [])
                items = []
                for row in source[offset : offset + limit]:
                    topic = topics.get(int(row.get("topic_id", 0)), {})
                    blurb = str(row.get("blurb", ""))
                    item = {
                        "ref": f"forum:{row.get('topic_id')}/{row.get('post_number')}",
                        "author": row.get("username"),
                        "text": self._snippet(blurb),
                        "created_at": row.get("created_at"),
                        "media": [
                            {
                                "ref": f"{row.get('topic_id')}/{row.get('post_number')}#image-{index}",
                                "url": url,
                            }
                            for index, url in enumerate(extract_image_urls(blurb), 1)
                        ],
                    }
                    if topic_id is None and topic.get("title"):
                        item["topic"] = self._snippet(str(topic["title"]), 160)
                    items.append(
                        {k: v for k, v in item.items() if v not in (None, "", [], {})}
                    )
            next_cursor = None
            next_offset = offset + len(items)
            more = next_offset < len(source) or bool(data.get("more_posts"))
            if more and page >= 10 and next_offset >= len(source):
                return {
                    "status": "partial",
                    "items": items,
                    "query": search_query,
                    "error": {
                        "code": "search_limit_reached",
                        "message": "Discourse search page limit reached",
                        "retryable": False,
                    },
                }
            if more:
                next_cursor = self._cursor(
                    {
                        "kind": "forum_search",
                        "query": search_query,
                        "page": page if next_offset < len(source) else page + 1,
                        "offset": next_offset if next_offset < len(source) else 0,
                        "result_kind": kind,
                        "request": state.get("request", {}) if cursor else request,
                    }
                )
            return self._ok(items, query=search_query, next_cursor=next_cursor)
        except Exception as exc:
            return self._error(exc)

    async def forum_read(
        self,
        post_id: int | None = None,
        topic_id: int | None = None,
        post_number: int | None = None,
        username: str | None = None,
        order: Literal["latest", "oldest"] = "latest",
        limit: int = 10,
        cursor: str | None = None,
        images: Literal["auto", "none", "selected"] = "none",
        image_refs: list[str] | None = None,
    ) -> tuple[str, list[PostShort]]:
        """精读一个帖子，或按顺序读取一个话题的楼层。

        何时用：已经知道要看哪一楼（forum_search 返回的 ref、用户给的链接楼层、
        reply_to），或需要顺着话题从头/从尾读一段时。
        参数要点：精读用 post_id 或 topic_id+post_number；顺序读用 topic_id 加
        order/limit，可用 username 只看某人的楼层；同一帖只需读一次，重复读取会直接
        复用结果。图片默认不加载，只有需要看图时才传 images="auto"（该帖全部图）或
        images="selected" 并给出 image_refs。
        返回：items 为帖子（ref、post_id、author、text、created_at、reply_to、media
        引用）；正文过长时给出 next_cursor，只传 cursor 继续读下一页；complete=true
        表示话题已读完。
        """
        try:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            limit = min(limit, 20)
            for locator in (post_id, topic_id, post_number):
                if locator is not None and locator <= 0:
                    raise ValueError("post and topic locators must be positive")
            if cursor:
                state = self._resume(cursor, "forum_read")
                self._check_cursor_arguments(
                    state,
                    {
                        "post_id": post_id,
                        "topic_id": topic_id,
                        "post_number": post_number,
                        "username": username,
                        "order": order,
                    },
                    defaults={"order": "latest"},
                )
                post_id, topic_id, post_number = (
                    state.get("post_id"),
                    state.get("topic_id"),
                    state.get("post_number"),
                )
                username, order = state.get("username"), state.get("order", order)
                offset = state.get("offset", 0)
            else:
                offset = 0
                request = {
                    "post_id": post_id,
                    "topic_id": topic_id,
                    "post_number": post_number,
                    "username": username,
                    "order": order,
                }
            exact = post_id is not None or post_number is not None
            if exact and username is not None:
                raise ValueError("username is available only for topic list reads")
            if images == "selected" and not image_refs:
                raise ValueError("selected image mode requires image_refs")
            if images != "selected" and image_refs:
                raise ValueError("image_refs requires images=selected")
            if post_id is not None and (
                topic_id is not None or post_number is not None
            ):
                raise ValueError("Use post_id or topic_id + post_number, not both")
            if post_number is not None and topic_id is None:
                raise ValueError("post_number requires topic_id")
            if post_id is not None:
                post = await cached_query(
                    f"forum_read:post:{post_id}",
                    lambda: self.shuiyuan_model.get_post_details(post_id),
                )
                posts, title = [post], ""
            elif post_number is not None:
                post = await cached_query(
                    f"forum_read:floor:{topic_id}:{post_number}",
                    lambda: self.shuiyuan_model.get_post_details_by_post_number(
                        topic_id, post_number
                    ),
                )
                posts, title = [post], ""
            elif topic_id is not None:
                turn = current_turn.get()
                if not cursor and turn and topic_id in turn.completed_topics:
                    payload = self._ok(
                        [], topic=turn.topic_titles.get(topic_id), complete=True
                    )
                    return json.dumps(payload, ensure_ascii=False), []
                fetch_limit = limit
                (
                    title,
                    posts,
                    next_offset,
                    has_more,
                ) = await cached_query(
                    f"forum_read:topic:{topic_id}:{username}:{order}:{offset}:{fetch_limit}",
                    lambda: self.shuiyuan_model.read_topic_post_page(
                        topic_id,
                        offset=offset,
                        limit=fetch_limit,
                        username=username,
                        ascending=order == "oldest",
                    ),
                )
            else:
                raise ValueError("Provide post_id, topic_id, or a cursor")
            if exact:
                has_more = False
            short = [PostShort(post, title, full=True) for post in posts]
            items = []
            for value in short:
                text_offset = offset if exact else 0
                text_limit = 1400
                page_end = text_offset + text_limit
                if exact:
                    page, page_end = self._text_page(
                        value._data["content"], text_offset
                    )
                    text_limit = len(page)
                data = value.to_compact_dict(
                    text_limit=text_limit,
                    text_offset=text_offset,
                )
                if images == "none":
                    data.pop("media", None)
                elif images == "selected" and image_refs:
                    data["media"] = [
                        m for m in data.get("media", []) if m["ref"] in image_refs
                    ]
                    value.image_urls = [m["url"] for m in data["media"]]
                items.append(data)
            next_cursor = None
            if exact and short and len(short[0]._data["content"]) > page_end:
                next_cursor = self._cursor(
                    {
                        "kind": "forum_read",
                        "post_id": short[0].id,
                        "offset": page_end,
                        "request": state.get("request", {}) if cursor else request,
                    }
                )
            elif not exact and has_more:
                next_cursor = self._cursor(
                    {
                        "kind": "forum_read",
                        "topic_id": topic_id,
                        "username": username,
                        "order": order,
                        "offset": next_offset,
                        "request": state.get("request", {}) if cursor else request,
                    }
                )
            payload = self._ok(items, topic=title, next_cursor=next_cursor)
            if not exact:
                turn = current_turn.get()
                if turn:
                    turn.note_topic_page(
                        topic_id,
                        items,
                        complete=not has_more,
                        title=title,
                    )
                if not has_more:
                    payload["complete"] = True
            artifacts = short if exact and images != "none" else []
            return json.dumps(payload, ensure_ascii=False), artifacts
        except Exception as exc:
            return json.dumps(self._error(exc), ensure_ascii=False), []

    async def users(
        self,
        query: str | None = None,
        username: str | None = None,
        usernames: list[str] | None = None,
        user_id: int | None = None,
        include_avatar: bool = False,
    ) -> dict:
        """查询水源用户资料，可选带头像。

        何时用：需要确认用户存在、拿到 user_id、昵称或头像时。
        参数要点：四种模式只能选一种——username 精确查一个人；usernames 一次精确查
        多个人（最多 50，多个用户名一律用这个，不要逐个调用）；query 按名字模糊搜索；
        user_id 反查已知 ID。只在需要头像（例如生成合照）时传 include_avatar=true。
        返回：items 为用户（user_id、username、name、可选 avatar）；批量模式下每项带
        input 和 status，查不到的项 status 为 error。
        """
        try:
            modes = sum(
                value not in (None, "", [])
                for value in (query, username, usernames, user_id)
            )
            if modes != 1:
                raise ValueError(
                    "Provide exactly one of query, username, usernames, or user_id"
                )
            if user_id is not None and user_id <= 0:
                raise ValueError("user_id must be positive")
            if usernames is not None:
                if not usernames or len(usernames) > 50:
                    raise ValueError("Provide 1 to 50 usernames")
                gate = asyncio.Semaphore(4)

                async def resolve(name: str) -> dict:
                    async with gate:
                        return await self._exact_user(name, include_avatar)

                tasks: dict[str, asyncio.Task] = {}
                for name in usernames:
                    key = name.strip().lstrip("@").casefold()
                    if key not in tasks:
                        tasks[key] = asyncio.create_task(resolve(name))
                await asyncio.gather(*tasks.values())
                items = [
                    {
                        "input": name,
                        **tasks[name.strip().lstrip("@").casefold()].result(),
                    }
                    for name in usernames
                ]
                return {
                    "status": (
                        "ok"
                        if all(item.get("status") == "ok" for item in items)
                        else "partial"
                    ),
                    "items": items,
                }
            if username:
                result = await self._exact_user(username, include_avatar)
                if result.get("status") == "ok":
                    return self._ok(
                        [
                            {
                                key: value
                                for key, value in result.items()
                                if key != "status" and value not in (None, "", [], {})
                            }
                        ]
                    )
                return {
                    "status": "error",
                    "code": result.get("error", "not_found"),
                    "message": result.get("message", "User was not found"),
                    "retryable": bool(result.get("retryable")),
                }
            if user_id is not None:
                user = await self.shuiyuan_model.search_user_by_user_id(user_id)
                if user is None:
                    return {
                        "status": "error",
                        "code": "unresolved",
                        "message": "No public post establishes this user ID",
                        "retryable": False,
                    }
                value = UserShort(user, include_avatar=include_avatar)
                return self._ok([value.to_compact_dict()])
            found = await self.shuiyuan_model.search_user_by_term(query or "")
            return self._ok(
                [UserShort(user, include_avatar).to_compact_dict() for user in found]
            )
        except Exception as exc:
            return self._error(exc)

    async def _exact_user(self, username: str, include_avatar: bool) -> dict:
        username = username.strip().lstrip("@")
        if not username or any(char in username for char in "/?#"):
            raise ValueError("Invalid username")

        async def fetch() -> dict:
            user = await self.shuiyuan_model.get_user_by_username(username)
            if user is None:
                return {
                    "status": "error",
                    "code": "not_found",
                    "message": "User was not found",
                    "retryable": False,
                }
            if user.username.casefold() != username.casefold():
                raise ValueError("User identity mismatch")
            return {
                "status": "ok",
                **UserShort(user, include_avatar=include_avatar).to_compact_dict(),
            }

        return await cached_query("user:" + username.casefold(), fetch)

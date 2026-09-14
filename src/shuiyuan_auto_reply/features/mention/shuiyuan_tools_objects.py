"""
Only important information for LLM is kept here.
"""

import json
from typing import Optional

from shuiyuan_auto_reply.application.tool_results import PAGE_CHARS, current_turn
from shuiyuan_auto_reply.shuiyuan.constants import base_url
from shuiyuan_auto_reply.shuiyuan.objects import PostDetails, User
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_multimodal import extract_image_urls
from .post_content import parse_content


class UserShort:
    """
    Represents a short version of a user, with only the most important information.
    """

    id: int
    username: str
    name: Optional[str]
    avatar: Optional[str] = None

    def __init__(self, user: User, include_avatar: bool = False):
        self.id = user.id
        self.username = user.username
        self.name = user.name
        self.avatar = None
        if include_avatar and user.avatar_template:
            avatar_path = user.avatar_template.replace("{size}", "288")
            if avatar_path.startswith(("http://", "https://")):
                self.avatar = avatar_path
            else:
                self.avatar = f"{base_url}{avatar_path}"

    def __str__(self):
        text = f"ID: 【{self.id}】 Username: 【{self.username}】"
        if self.name:
            text += f" Name: 【{self.name}】"
        if self.avatar:
            text += f" avatar: 【{self.avatar}】"
        return text + "\n"

    def __repr__(self):
        return self.__str__()


class PostShort:
    """
    Represents a short version of a post, with only the most important information.
    """

    id: int
    post_number: int
    topic_id: int
    name: Optional[str]
    user_id: int
    username: str
    cooked: str
    raw: Optional[str]
    reply_to_post_number: Optional[int]
    title: str
    image_urls: list[str]

    def __init__(self, post: PostDetails, title: str = "", *, full: bool = False):
        self.id = post.id
        self.post_number = post.post_number
        self.topic_id = post.topic_id
        self.name = post.name
        self.user_id = post.user_id
        self.username = post.username
        image_urls: list[str] = []
        seen: set[str] = set()
        for text in (post.raw, post.cooked):
            for image_url in extract_image_urls(text):
                if image_url in seen:
                    continue
                seen.add(image_url)
                image_urls.append(image_url)
        self.image_urls = image_urls
        self.cooked = post.cooked
        self.raw = post.raw
        self.full = full
        self.warnings = []
        self._data = parse_content(post.raw, post.cooked)
        self.result_id = None
        turn = current_turn.get()
        if turn:
            self.result_id = turn.save(self._data["content"])
        self.reply_to_post_number = post.reply_to_post_number
        self.title = title

    def to_dict(self, cursor: int = 0) -> dict:
        content = self._data["content"]
        limit = PAGE_CHARS if self.full else 800
        return {
            "post_id": self.id,
            "topic_id": self.topic_id,
            "post_number": self.post_number,
            "reply_to_post_number": self.reply_to_post_number,
            "author": {
                "user_id": self.user_id,
                "username": self.username,
                "name": self.name,
            },
            "title": self.title,
            **self._data,
            "content": content[cursor : cursor + limit],
            "image_urls": self.image_urls,
            "truncated": len(content) > cursor + limit,
            "total_chars": len(content),
            "result_id": self.result_id,
            "next_cursor": cursor + limit if len(content) > cursor + limit else None,
            "read_full": {"tool": "get_post_by_id", "post_id": self.id},
            "warnings": self.warnings,
        }

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


class PostSearchResults(list):
    """List-compatible search result with explicit coverage metadata for the model."""

    def __init__(self, items=(), *, query=None, truncated=False):
        super().__init__(items)
        self.query = query or {}
        self.truncated = truncated

    def __str__(self):
        return json.dumps(
            {
                "query_scope": self.query,
                "truncated": self.truncated,
                "pagination_supported": False,
                "posts": [post.to_dict() for post in self],
                "returned_count": len(self),
                "coverage": "not_guaranteed_complete",
                "continuation": "Use get_post/get_post_by_id for full content; refine query for additional matches.",
            },
            ensure_ascii=False,
        )

    __repr__ = __str__

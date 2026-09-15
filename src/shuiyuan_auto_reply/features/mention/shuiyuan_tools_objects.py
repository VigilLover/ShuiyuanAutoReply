"""
Only important information for LLM is kept here.
"""

import json
from typing import Optional

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

    def to_compact_dict(self) -> dict:
        return {
            key: value
            for key, value in {
                "user_id": self.id,
                "username": self.username,
                "name": self.name,
                "avatar": self.avatar,
            }.items()
            if value not in (None, "")
        }


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
    source = "forum_read"

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
        self.reply_to_post_number = post.reply_to_post_number
        self.created_at = getattr(post, "created_at", None)
        self.title = title

    def __str__(self):
        return json.dumps(self.to_compact_dict(text_limit=6000), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_compact_dict(self, *, text_limit: int = 1200, text_offset: int = 0) -> dict:
        content = self._data["content"]
        item = {
            "ref": f"forum:{self.topic_id}/{self.post_number}",
            "post_id": self.id,
            "author": self.username,
            "text": content[text_offset : text_offset + text_limit],
            "created_at": str(self.created_at) if self.created_at else None,
            "reply_to": (
                f"forum:{self.topic_id}/{self.reply_to_post_number}"
                if self.reply_to_post_number
                else None
            ),
            "media": [
                {
                    "ref": f"{self.topic_id}/{self.post_number}#image-{index}",
                    "url": url,
                }
                for index, url in enumerate(self.image_urls, 1)
            ],
        }
        if len(content) > text_offset + text_limit:
            item["text_truncated"] = True
        return {
            key: value for key, value in item.items() if value not in (None, "", [], {})
        }

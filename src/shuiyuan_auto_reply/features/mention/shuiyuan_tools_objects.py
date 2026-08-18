"""
Only important information for LLM is kept here.
"""

from typing import Optional

from shuiyuan_auto_reply.shuiyuan.constants import base_url
from shuiyuan_auto_reply.shuiyuan.objects import PostDetails, User
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_multimodal import extract_image_urls


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

    def __init__(self, post: PostDetails, title: str):
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
        self.cooked = post.cooked[:384]
        self.raw = post.raw[:384] if post.raw else None
        self.reply_to_post_number = post.reply_to_post_number
        self.title = title

    def __str__(self):
        text = (
            f"PostMeta: id={self.id}, post_number={self.post_number}, topic_id={self.topic_id}\n"
            f"FromUser: {UserShort(User(id=self.user_id, username=self.username, name=self.name))}"
            f"TopicTitle: {self.title}\n"
            f"Content: {ShuiyuanModel.remove_shuiyuan_signature(self.raw) if self.raw else self.cooked}\n"
        )
        if self.image_urls:
            text += f"Images: {', '.join(self.image_urls)}\n"
        return text

    def __repr__(self):
        return self.__str__()

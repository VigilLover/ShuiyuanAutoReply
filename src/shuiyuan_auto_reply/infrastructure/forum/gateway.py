"""Shuiyuan API adapter."""

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel


class ShuiyuanForumGateway:
    def __init__(self, model: ShuiyuanModel) -> None:
        self.model = model

    async def get_post(self, post_id: int):
        return await self.model.get_post_details(post_id)

    async def get_topic(self, topic_id: int):
        return await self.model.get_topic_details(topic_id)

    async def get_actions(self, username: str, action_types: list[int]):
        return await self.model.get_actions(username, action_types)

    async def reply(
        self, text: str, topic_id: int, reply_to_post_number: int | None
    ) -> None:
        await self.model.reply_to_post(text, topic_id, reply_to_post_number)

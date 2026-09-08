"""Forum-only decorations kept out of the application result."""

from shuiyuan_auto_reply.shuiyuan.user_action_model import BaseUserActionModel


class ForumOutputFormatter:
    @staticmethod
    def signature(nickname: str) -> str:
        return (
            "\n<div data-signature>\n\n---\n"
            f"[right]这里是AI{nickname.strip('bot')}<small>(Pumpkin Edition)</small> :robot: [/right]\n"
            "</div>"
        )

    @staticmethod
    def make_unique(text: str) -> str:
        return BaseUserActionModel._make_unique_reply(text)

    def format_chat(self, text: str, nickname: str) -> str:
        return self.make_unique(f"{text}{self.signature(nickname)}")

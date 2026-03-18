import re
import logging
import traceback
from typing import Optional
from src.shuiyuan.objects import User, UserActionDetails
from src.shuiyuan.shuiyuan_model import ShuiyuanModel
from src.shuiyuan.user_action_model import BaseUserActionModel
from src.constants import auto_reply_tag
from .mention_tongyi_model import MentionTongyiModel


class MentionModel(BaseUserActionModel):
    """
    A class to represent a mention model for robot auto-replies.
    """

    def __init__(self, model: ShuiyuanModel, bot_username: str, persona: str):
        """
        Initialize the TopicModel with a ShuiyuanModel instance.

        :param model: An instance of ShuiyuanModel.
        :param bot_username: The username of the robot account.
        :param persona: The name of the character model to emulate.
        """
        super().__init__(model, bot_username, [5, 7])
        self.persona = persona
        
        # 预先定义各个角色的触发词和昵称
        self.persona_configs = {
            "wolf_lumine": {"trigger": "【小狼】", "nickname": "小狼bot"},
            "存档读取": {"trigger": "【存读】", "nickname": "存读bot"}, # 可扩展
        }
        self.config = self.persona_configs.get(persona, self.persona_configs["wolf_lumine"])
        self.trigger_word = self.config["trigger"]
        self.nickname = self.config["nickname"]
        
        self.mention_tongyi_model = MentionTongyiModel(model, username=persona)
        
    @staticmethod
    def _remove_shuiyuan_signature(text: str) -> str:
        """
        Remove the Shuiyuan signature from the given text.

        :param text: The text from which to remove the signature.
        :return: The text without the signature.
        """
        sig_re = r"<div data-signature>.*?</div>"
        return re.sub(sig_re, "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _parse_prompt_text(raw: str, prompt: str) -> Optional[str]:
        """
        Return text after the first occurrence of the prompt in raw.
        And remove the prompt itself and Shuiyuan signature.

        :param raw: The raw content of the post.
        :param prompt: The prompt string to look for.
        :return: The parsed text after the prompt or None if prompt not found.
        """
        # Get the text after the first occurrence of the prompt
        irst_occurrence = raw.find(prompt)
        if irst_occurrence == -1:
            return None
        raw = raw[irst_occurrence:]

        # Remove the keyword itself
        raw = MentionModel._remove_shuiyuan_signature(raw.replace(prompt, "")).strip()
        return raw

    async def _pumpkin_condition(self, raw: str, user: User, topic_id: int) -> Optional[str]:
        """
        Check if the raw content of a post contains the target trigger word.

        :param raw: The raw content of the post.
        :param user: The user who posted the message.
        :param topic_id: The ID of the topic.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # Check if the mention actually exists
        # r = re.search(r"@wolf_lumine", raw, re.IGNORECASE)
        # if r is None:
        #     return None

        # If the raw content does not contain the trigger word, we return None
        raw = MentionModel._parse_prompt_text(raw, self.trigger_word)
        if raw is None:
            logging.info(f"==> [MentionModel] post did not contain keyword {self.trigger_word}, skipping AI spawn.")
            return None

        logging.info(f"==> [MentionModel] Triggered AI spawn with prompt: '{raw}' for user: {user.username}")
        # Let the Tongyi model respond based on conversation and similar responses
        reply = await self.mention_tongyi_model.get_pumpkin_response(topic_id, raw, user)
        logging.info(f"==> [MentionModel] AI replied with length {len(reply)}.")
        signature = (
            "\n"
            "<div data-signature>\n"
            "\n"
            "---\n"
            f"[right]这里是AI{self.nickname.strip('bot')} :robot: [/right]\n"
            "</div>"
        )
        reply = f"{reply}{signature}"
        return MentionModel._make_unique_reply(reply)

    async def _clear_condition(self, raw: str, user: User) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【清除历史】".

        :param raw: The raw content of the post.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # If the raw content does not contain "清除历史", we return None
        if "【清除历史】" not in raw:
            return None

        # Clear the session history for the user
        self.mention_tongyi_model.clear_session_history(user.id)

        return MentionModel._make_unique_reply(f"已清除与{self.nickname}的对话历史记录")

    def _help_condition(self, raw: str) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【帮助】".

        :param raw: The raw content of the post.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # If the raw content does not contain "帮助", we return None
        if "【帮助】" not in raw:
            return None

        return MentionModel._make_unique_reply(
            f"欢迎和{self.nickname}对话o(｀ω´ )o\n"
            "帮助信息如下：\n"
            f"1. 输入{self.trigger_word}+对话，与{self.nickname}聊天 :wolf:\n"
            f"2. 输入【清除历史】，清除与{self.nickname}的对话历史记录 :broom:\n"
            "3. 输入【帮助】，查看该帮助信息 :question:"
        )

    async def _new_action_routine(self, action: UserActionDetails) -> None:
        """
        A routine to handle new actions for a specific user.
        NOTE: no exception should be raised in this method.

        :param action: The details of the user action (mention).
        :return: None
        """
        logging.info(f"==> [MentionModel] Event triggered for action_type={action.action_type} on post_id={action.post_id}")
        
        # This is the text to reply to the post
        text: Optional[str] = None

        try:
            # First let's try to get the post details
            post_details = await self.model.get_post_details(action.post_id)
            post_user = User(
                post_details.user_id,
                post_details.username,
                post_details.name,
            )
            logging.info(f"==> [MentionModel] Fetched post details successfully. User={post_user.username}")

            # If the member "raw" is not present, we should skip it
            if post_details.raw is None:
                logging.warning(
                    f"Post {action.post_id} does not have raw content, skipping."
                )
                return

        except Exception:
            logging.error(
                f"Failed to get post details for {action.post_id}, "
                f"traceback is as follows:\n{traceback.format_exc()}"
            )
            return

        try:
            # If the post is an auto-reply, we should skip it
            if auto_reply_tag in post_details.raw:
                logging.info(f"==> [MentionModel] Post {action.post_id} is an auto-reply. Skipping.")
                return

            # Check help condition
            logging.info(f"==> [MentionModel] Checking _help_condition...")
            text = self._help_condition(post_details.raw)
            if text is not None:
                logging.info(f"==> [MentionModel] _help_condition matched.")
                return

            # Check clear condition
            logging.info(f"==> [MentionModel] Checking _clear_condition...")
            text = await self._clear_condition(post_details.raw, post_user)
            if text is not None:
                logging.info(f"==> [MentionModel] _clear_condition matched.")
                return

            # Check pumpkin condition
            logging.info(f"==> [MentionModel] Checking _pumpkin_condition...")
            text = await self._pumpkin_condition(post_details.raw, post_user, action.topic_id)
            if text is not None:
                logging.info(f"==> [MentionModel] _pumpkin_condition matched.")
                return

            logging.info(f"==> [MentionModel] No conditions matched for post {action.post_id}.")

        except Exception:
            # If we failed to get the post details or any other error occurred
            logging.error(
                f"Failed to process post {action.post_id}, "
                f"traceback is as follows:\n{traceback.format_exc()}"
            )
            # We should reply to the post with an error message
            text = MentionModel._make_unique_reply(
                "抱歉，小狼bot遇到了一个错误，暂时无法处理您的请求，请稍后再试 :crying_cat:"
            )

        finally:
            if text is not None:
                logging.info(f"==> [MentionModel] Replying to topic {action.topic_id} at post {action.post_number}...")
                await self.model.reply_to_post(
                    text,
                    action.topic_id,
                    action.post_number,
                )
                logging.info(f"==> [MentionModel] Reply successfully sent to post {action.post_id}.")

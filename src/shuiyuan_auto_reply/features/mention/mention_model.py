import logging
import asyncio
import random
import re
import traceback
from typing import Any, Dict, List, Optional

from shuiyuan_auto_reply.application import BotContext, BotService, HandlerRegistry
from shuiyuan_auto_reply.application.handlers import (
    CallbackChatHandler,
    ClearHandler,
    DiceHandler,
    HelpHandler,
    PetHandler,
    PollHandler,
)
from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings
from shuiyuan_auto_reply.constants import settings
from shuiyuan_auto_reply.domain import (
    AttachmentRef,
    ActorRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ForumContextRef,
    ReplyRequest,
    ReplyResult,
)
from shuiyuan_auto_reply.infrastructure.persistence import InMemorySessionRepository, SQLiteExecutionObserver, SQLiteSessionRepository
from shuiyuan_auto_reply.infrastructure.forum import (
    ForumMediaUploader,
    ForumOutputFormatter,
    ForumReplyMediaPublisher,
)
from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.shuiyuan.objects import User, UserActionDetails
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
from shuiyuan_auto_reply.shuiyuan.user_action_model import BaseUserActionModel

from .mention_pet_model import MentionPetModel

MENTION_HANDLER_PRIORITIES = {
    "help": 10,
    "rua": 20,
    "clear": 30,
    "chat": 40,
    "dice": 50,
    "poll": 60,
}


class MentionModel(BaseUserActionModel):
    """
    A class to represent a mention model for robot auto-replies.
    """

    def __init__(
        self,
        model: ShuiyuanModel,
        bot_username: str,
        persona: str,
        chat_model: Any,
        provider_settings: ProviderSettings | None = None,
        state_store=None,
        runtime_refresher=None,
    ):
        """
        Initialize the TopicModel with a ShuiyuanModel instance.

        :param model: An instance of ShuiyuanModel.
        :param bot_username: The username of the robot account.
        :param persona: The name of the character model to emulate.
        """
        super().__init__(model, bot_username, [5, 7])
        self.persona = persona
        self.username = bot_username

        # 预先定义各个角色的触发词和昵称
        self.persona_configs = {
            "wolf_lumine": {"trigger": "【小狼】", "nickname": "小狼bot"},
            "存档读取": {"trigger": "【存读】", "nickname": "存读bot"},
            "MonkeysPumpkin": {"trigger": "【小南瓜】", "nickname": "南瓜bot"},
        }
        self.config = self.persona_configs.get(persona, self.persona_configs["wolf_lumine"])
        self.trigger_word = self.config["trigger"]
        self.nickname = self.config["nickname"]

        self.pumpkin = chat_model
        self._runtime_lock = asyncio.Lock()
        self._runtime_counts: dict[Any, int] = {chat_model: 0}
        self._retired_runtimes: set[Any] = set()
        self._closed_runtimes: set[Any] = set()
        self.pet_model = MentionPetModel(
            persona=persona, provider_settings=provider_settings
        )
        self.output_formatter = ForumOutputFormatter()
        self.state_store = state_store
        self.runtime_refresher = runtime_refresher
        self.media_uploader = ForumMediaUploader(model, state_store)
        self.media_publisher = ForumReplyMediaPublisher(
            self.media_uploader, state_store
        )
        self.bot_service = BotService(
            SQLiteSessionRepository(state_store) if state_store else InMemorySessionRepository(),
            HandlerRegistry(
                [
                    HelpHandler(
                        lambda raw: "【帮助】" in raw,
                        self._handle_help,
                        MENTION_HANDLER_PRIORITIES["help"],
                    ),
                    PetHandler(
                        lambda raw: "【rua】" in raw,
                        self._handle_rua,
                        MENTION_HANDLER_PRIORITIES["rua"],
                    ),
                    ClearHandler(
                        lambda raw: "【清除历史】" in raw,
                        self._handle_clear,
                        MENTION_HANDLER_PRIORITIES["clear"],
                    ),
                    CallbackChatHandler(
                        lambda raw: self._parse_prompt_text(raw, self.trigger_word)
                        is not None,
                        self._handle_chat,
                        MENTION_HANDLER_PRIORITIES["chat"],
                    ),
                    DiceHandler(
                        lambda raw: "【投掷】" in raw,
                        self._handle_dice,
                        MENTION_HANDLER_PRIORITIES["dice"],
                    ),
                    PollHandler(
                        lambda raw: "【抽选】" in raw,
                        self._handle_poll,
                        MENTION_HANDLER_PRIORITIES["poll"],
                    ),
                ]
            ),
            observer_factory=(lambda: SQLiteExecutionObserver(state_store)) if state_store else None,
        )

    async def swap_chat_model(self, candidate: Any) -> None:
        close_now = None
        async with self._runtime_lock:
            old = self.pumpkin
            self.pumpkin = candidate
            self._runtime_counts.setdefault(candidate, 0)
            if self._runtime_counts.get(old, 0) == 0:
                self._runtime_counts.pop(old, None)
                self._closed_runtimes.add(old)
                close_now = old
            else:
                self._retired_runtimes.add(old)
        if close_now is not None:
            await self._close_chat_runtime(close_now)

    async def _acquire_chat_runtime(self):
        async with self._runtime_lock:
            runtime = self.pumpkin
            self._runtime_counts[runtime] = self._runtime_counts.get(runtime, 0) + 1
            return runtime

    async def _release_chat_runtime(self, runtime) -> None:
        close_now = False
        async with self._runtime_lock:
            remaining = self._runtime_counts.get(runtime, 1) - 1
            self._runtime_counts[runtime] = remaining
            if remaining == 0 and runtime in self._retired_runtimes:
                self._retired_runtimes.discard(runtime)
                self._runtime_counts.pop(runtime, None)
                self._closed_runtimes.add(runtime)
                close_now = True
        if close_now:
            await self._close_chat_runtime(runtime)

    @staticmethod
    async def _close_chat_runtime(runtime) -> None:
        try:
            await runtime.aclose()
        except Exception:
            logging.exception("Failed to close retired forum chat runtime")

    async def aclose(self) -> None:
        await super().aclose()
        async with self._runtime_lock:
            runtimes = tuple(
                runtime
                for runtime in self._runtime_counts
                if runtime not in self._closed_runtimes
            )
            self._runtime_counts.clear()
            self._retired_runtimes.clear()
            self._closed_runtimes.update(runtimes)
        for runtime in runtimes:
            await self._close_chat_runtime(runtime)

    async def _handle_help(self, context: BotContext) -> str | None:
        return self._help_condition(context.request.content)

    async def _handle_rua(self, context: BotContext) -> str | None:
        actor = context.request.actor
        return await self._rua_condition(
            context.request.content, actor.username, actor.display_name or ""
        )

    async def _handle_clear(self, context: BotContext) -> str | None:
        forum = context.request.forum_context
        if forum is None:
            return None
        return await self._clear_condition(context.request.content, forum.topic_id)

    async def _handle_chat(self, context: BotContext) -> str | ReplyResult | None:
        forum = context.request.forum_context
        if forum is None:
            return None
        actor = context.request.actor
        user_id = int(actor.external_id)
        return await self._pumpkin_condition(
            forum.topic_id,
            forum.reply_to_post_number,
            context.request.content,
            User(user_id, actor.username, actor.display_name),
        )

    async def _handle_dice(self, context: BotContext) -> str | None:
        return self._random_condition(context.request.content)

    async def _handle_poll(self, context: BotContext) -> str | None:
        forum = context.request.forum_context
        if forum is None:
            return None
        return await self._poll_condition(
            context.request.content, forum.topic_id, forum.reply_to_post_number
        )
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
        return ShuiyuanModel.remove_shuiyuan_signature(raw.replace(prompt, "")).strip()

    async def _pumpkin_condition(
        self, topic_id: int, reply_to_post_number: Optional[int], raw: str, user: User
    ) -> Optional[str | ReplyResult]:
        """
        Check if the raw content of a post contains the target trigger word.

        :param topic_id: The ID of the topic where the post is located.
        :param reply_to_post_number: The post number this post is replying to.
        :param raw: The raw content of the post.
        :param user: The user who posted the message.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # If the raw content does not contain the trigger word, we return None
        raw = MentionModel._parse_prompt_text(raw, self.trigger_word)
        if raw is None:
            logging.info(f"==> [MentionModel] post did not contain keyword {self.trigger_word}, skipping AI spawn.")
            return None

        logging.info(f"==> [MentionModel] Triggered AI spawn with prompt: '{raw}' for user: {user.username}")
        # Let the Tongyi model respond based on conversation and similar responses
        artifacts = ()
        input_artifacts = ()
        runtime = await self._acquire_chat_runtime()
        try:
            response = await runtime.get_pumpkin_response(
                topic_id,
                reply_to_post_number,
                raw,
                user,
                conversation_ref=ConversationRef(
                    Channel.FORUM,
                    f"topic:{topic_id}",
                    self.username,
                    self.persona,
                ),
                include_artifacts=True,
            )
            if len(response) == 2:
                reply, artifacts = response
            else:
                reply, artifacts, input_artifacts = response
        except ValueError as e:
            if "DataInspectionFailed" in str(e):
                reply = "抱歉，您的输入包含不当内容，无法处理。"
                logging.error(f"==> [MentionModel] AI replied with DataInspectionFailed: {str(e)}")
            else:
                reply = "抱歉，遇到了一些错误。"
                logging.error(f"==> [MentionModel] AI replied with ValueError: {str(e)}")
        except Exception as e:
            reply = "抱歉，遇到了一些未知错误。"
            logging.error(f"==> [MentionModel] AI replied with Exception: {str(e)}")
        finally:
            await self._release_chat_runtime(runtime)

        publication = await self.media_publisher.publish(reply, artifacts)
        reply = publication.text
        for media in publication.published:
            event_type = (
                "forum.image_reused" if media.reused else "forum.image_uploaded"
            )
            await emit_event(
                event_type,
                {
                    "artifact_id": media.artifact.artifact_id,
                    "short_path": media.short_path,
                    "source_kind": media.source_kind,
                },
            )
        for failure in publication.failures:
            await emit_event(
                "forum.image_upload_failed",
                {
                    "artifact_id": failure.artifact.artifact_id,
                    "source_kind": failure.source_kind,
                    "error": failure.error[:500],
                },
            )

        logging.info(f"==> [MentionModel] AI replied with length {len(reply)}.")
        formatted = self.output_formatter.format_chat(reply, self.nickname)
        if artifacts or input_artifacts:
            def attachment(artifact):
                return AttachmentRef(
                    artifact.uri,
                    artifact.mime_type,
                    artifact.artifact_id,
                    getattr(artifact, "source_kind", "generated"),
                    getattr(artifact, "source_url", None),
                    getattr(artifact, "filename", None),
                    getattr(artifact, "width", None),
                    getattr(artifact, "height", None),
                )
            return ReplyResult(
                formatted,
                tuple(attachment(artifact) for artifact in artifacts),
                tuple(attachment(artifact) for artifact in input_artifacts),
            )
        return formatted

    async def _clear_condition(self, raw: str, topic_id: int) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【清除历史】".

        :param raw: The raw content of the post.

        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # If the raw content does not contain "清除历史", we return None
        if "【清除历史】" not in raw:
            return None

        # Clear the session history for the user
        self.pumpkin.clear_session_history(topic_id)

        return self.output_formatter.make_unique("已清除当前话题中的对话历史记录")

    def _random_condition(self, raw: str) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【投掷】".

        :param raw: The raw content of the post.
        :return: A random number between 1 and max (given in the post).
        """
        # If the raw content does not contain "投掷", we return None
        if "【投掷】" not in raw:
            return None

        # Use regular expression to extract the parameters for the random number generation
        r = re.search(r"【投掷】\s*(\d*)d\s*(\d+)", raw, re.IGNORECASE)
        if r is None:
            return self.output_formatter.make_unique(
                "请按照格式`【投掷】ndm`来投掷随机数，"
                "例如`【投掷】3d6`表示投掷3个1到6之间的随机数\n"
            )

        n = int(r.group(1)) if r.group(1) else 1
        m = int(r.group(2))

        # Check the validity of n and m
        if n <= 0 or m <= 0:
            return self.output_formatter.make_unique(
                "n和m必须都是正整数，请检查你的输入\n"
                "例如`【投掷】3d6`表示投掷3个1到6之间的随机数\n"
            )
        if n > 100:
            return self.output_formatter.make_unique(
                "n太大了，请限制在100以内\n"
                "例如`【投掷】3d6`表示投掷3个1到6之间的随机数\n"
            )

        # Generate n random numbers between 1 and m
        random_numbers = [random.randint(1, m) for _ in range(n)]
        random_numbers_str = ", ".join(str(num) for num in random_numbers)
        return self.output_formatter.make_unique(
            f"你投掷了 {n} 个 1 到 {m} 之间的随机数，结果是：\n"
            f"> {random_numbers_str}\n"
        )

    async def _poll_condition(
        self, raw: str, topic_id: int, reply_to_post_number: Optional[int]
    ) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【抽选】".

        :param raw: The raw content of the post.

        :param reply_to_post_number: The post number of the post to which current post is replying to.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # If the raw content does not contain "抽选", we return None
        if "【抽选】" not in raw:
            return None

        # Now we check if the post is replying to another post
        if reply_to_post_number is None:
            # Flow B, use regular expression to extract the topic_id/post_number
            r = re.search(r"【抽选】\s*(\d+)(?:/(\d+))?", raw, re.IGNORECASE)
            if r is None:
                return self.output_formatter.make_unique(
                    "请按照格式`【抽选】topic_id/post_number`或`【抽选】post_number`来抽选，"
                    "或者直接回复包含了投票的帖子来进行抽选，"
                    "例如`【抽选】2026/1896`表示选取ID为2026话题的第1896层里的投票进行随机抽选\n"
                )

            if r.group(2):
                topic_id = int(r.group(1))
                reply_to_post_number = int(r.group(2))
            else:
                reply_to_post_number = int(r.group(1))

        # Flow A, let's get the post details
        try:
            post_details = await self.model.get_post_details_by_post_number(
                topic_id, reply_to_post_number
            )
        except Exception:
            logging.error(
                f"Failed to get post details for topic_id {topic_id} and "
                f"post_number {reply_to_post_number}, "
                f"traceback is as follows:\n{traceback.format_exc()}"
            )
            return self.output_formatter.make_unique(
                "无法获取被抽选的帖子详情，请检查你的输入是否正确，或者稍后再试\n"
                "请按照格式`【抽选】topic_id/post_number`或`【抽选】post_number`来抽选，"
                "或者直接回复包含了投票的帖子来进行抽选，"
                "例如`【抽选】2026/1896`表示选取ID为2026话题的第1896层里的投票进行随机抽选\n"
            )

        # Check if the post contains a poll
        if post_details.polls is None:
            return self.output_formatter.make_unique(
                "被抽选的帖子中不包含投票，无法进行抽选，请检查你的输入或者稍后再试\n"
                "请按照格式`【抽选】topic_id/post_number`或`【抽选】post_number`来抽选，"
                "或者直接回复包含了投票的帖子来进行抽选，"
                "例如`【抽选】2026/1896`表示选取ID为2026话题的第1896层里的投票进行随机抽选\n"
            )

        # Check the visibility of the polls
        visible_polls = [
            poll
            for poll in post_details.polls
            if (poll.type == "regular" or poll.type == "multiple")
            and poll.public
            and (poll.results != "on_close" or poll.status == "closed")
        ]
        visible_poll_ids = {poll.id for poll in visible_polls}
        if not visible_poll_ids:
            return self.output_formatter.make_unique(
                "被抽选的帖子中的所有投票均不可见或类型不支持，"
                "当前仅支持单选或多选且公开的投票，请检查你的输入或者稍后再试\n"
            )

        # Try to get the full list of voters
        voters = await self.model.get_voters_by_post_id(post_details.id)

        # Now we randomly select one of the options for all polls
        selected_options: Dict[str, User | str] = {}
        for poll_id, users in voters.voters.items():
            # If there are no users who voted for this option, we should skip it
            if not users:
                selected_options[poll_id] = "参与投票人数为0，无法抽选"
                continue

            # Now randomly select one user from the list of users
            selected_user = random.choice(users)
            selected_options[poll_id] = selected_user

        # Now we have to match poll_id with their contents in post_details
        results: Dict[str, Dict[str, User | str] | str] = {}
        for idx, poll in enumerate(post_details.polls):
            current_poll_title = poll.title or f"投票 {idx + 1}"

            # Check whether the poll is visible or not
            if poll.id not in visible_poll_ids:
                results[current_poll_title] = "该投票不可见或类型不支持，无法抽选"
                continue

            current_poll_result: Dict[str, List[User] | str] = {}
            for option in poll.options:
                if option.id in selected_options:
                    current_poll_result[option.html] = selected_options[option.id]

            # Add titles for each poll
            results[current_poll_title] = current_poll_result

        # Finally we format the reply text
        reply_lines = ["抽选结果如下：\n"]
        for poll_title, options in results.items():
            # Poll title line
            reply_lines.append(f"## {poll_title}")
            # Error reported for this poll
            if isinstance(options, str):
                reply_lines.append(f"错误：{options}")
                continue
            # Now we add the result for each option in the poll
            for option_html, selected_user in options.items():
                # Option content line
                reply_lines.append(f"{option_html}")
                # If any error occurs, we report it here
                if isinstance(selected_user, str):
                    reply_lines.append(f"错误：{selected_user}")
                    continue
                reply_lines.append(f"抽选结果：@{selected_user.username}")
                reply_lines.append("")

        # Join all lines into a single reply text
        return self.output_formatter.make_unique("\n".join(reply_lines))

    def _help_condition(self, raw: str) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【帮助】".

        :param raw: The raw content of the post.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        # If the raw content does not contain "帮助", we return None
        if "【帮助】" not in raw:
            return None

        return self.output_formatter.make_unique(
            f"欢迎和{self.nickname}对话o(｀ω´ )o\n"
            "帮助信息如下：\n"
            f"1. 输入{self.trigger_word}+对话，与{self.nickname}聊天 :wolf:\n"
            f"2. 输入【清除历史】，清除当前话题中的对话历史记录 :broom:\n"
            "3. 输入【投掷】+ndm，投掷 n 个 1 到 m 之间的随机数 :game_die:\n"
            "4. 输入【抽选】+topic_id/post_number，随机抽选该帖中的投票结果，或者直接回复包含投票的帖子进行抽选 :ballot_box:\n"
            "5. 输入【帮助】，查看该帮助信息 :question:\n"
            "6. 输入【rua】，可以rua一下小狼哦 :kissing_cat:"
        )

    async def _rua_condition(self, raw: str, username: str, name: str) -> Optional[str]:
        """
        Check if the raw content of a post contains the string "【rua】".
        :param raw: The raw content of the post.
        :param username: The username of the user who posts.
        :param name: The display name of the user who posts.
        :return: A string to reply to the post if the condition is met, otherwise None.
        """
        if "【rua】" not in raw:
            return None

        rua_text = MentionModel._parse_prompt_text(raw, "【rua】")
        if rua_text is not None:
            rua_text = rua_text.strip()

        reply = await self.pet_model.get_rua_response(
            username=username,
            name=name,
            user_text=rua_text,
        )
        return self.output_formatter.format_chat(reply, self.nickname)

    async def _new_action_routine(self, action: UserActionDetails) -> None:
        """
        A routine to handle new actions for a specific user.
        NOTE: no exception should be raised in this method.

        :param action: The details of the user action (mention).
        :return: None
        """
        logging.info(f"==> [MentionModel] Event triggered for action_type={action.action_type} on post_id={action.post_id}")

        if self.runtime_refresher is not None:
            try:
                await self.runtime_refresher()
            except Exception:
                logging.exception(
                    "Failed to apply the latest forum runtime; keeping the previous runtime"
                )
        
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
            # If the post is an auto-reply send by the bot, we should skip it
            if settings.contains_auto_reply_tag(post_details.raw) and post_details.username == self.username:
                logging.info(f"==> [MentionModel] Post {action.post_id} is an auto-reply. Skipping.")
                return

            # Check if the mention actually exists
            r = re.search(rf"@{self.username}", post_details.raw, re.IGNORECASE)
            if r is None:
                return

            request = ReplyRequest(
                request_id=f"forum:{action.post_id}",
                conversation=ConversationRef(
                    Channel.FORUM,
                    f"topic:{post_details.topic_id}",
                    self.username,
                    self.persona,
                ),
                actor=ActorRef(
                    Channel.FORUM,
                    str(post_user.id),
                    post_user.username,
                    post_user.name,
                ),
                content=post_details.raw,
                dispatch_mode=DispatchMode.AUTO,
                forum_context=ForumContextRef(
                    topic_id=post_details.topic_id,
                    post_id=post_details.id,
                    post_number=post_details.post_number,
                    reply_to_post_number=post_details.reply_to_post_number,
                ),
            )
            if self.state_store is not None:
                try:
                    topic = await self.model.get_topic_details(post_details.topic_id)
                    await self.state_store.update_title_for_ref(
                        request.conversation, topic.title
                    )
                except Exception:
                    logging.exception("Failed to persist forum topic title")
            try:
                result = await self.bot_service.reply(request)
            except LookupError:
                logging.info(
                    "==> [MentionModel] No conditions matched for post %s.",
                    action.post_id,
                )
                return
            text = result.text

        except Exception:
            # If we failed to get the post details or any other error occurred
            logging.error(
                f"Failed to process post {action.post_id}, "
                f"traceback is as follows:\n{traceback.format_exc()}"
            )
            # We should reply to the post with an error message
            text = self.output_formatter.make_unique(
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
                await emit_event(
                    "forum.reply_published",
                    {"topic_id": action.topic_id, "post_number": action.post_number},
                )
                if self.state_store is not None:
                    await self.state_store.append_event_for_request(
                        f"forum:{action.post_id}",
                        "forum.reply_published",
                        {"topic_id": action.topic_id, "post_number": action.post_number},
                    )
                logging.info(f"==> [MentionModel] Reply successfully sent to post {action.post_id}.")

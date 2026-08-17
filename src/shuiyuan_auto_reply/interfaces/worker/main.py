"""Formal forum worker entry point."""

import asyncio
import logging

from dotenv import load_dotenv

from shuiyuan_auto_reply.application.handlers import ChatHandler
from shuiyuan_auto_reply.bootstrap.container import ApplicationContainer
from shuiyuan_auto_reply.bootstrap.providers import MentionProviderFactory
from shuiyuan_auto_reply.bootstrap.settings import AppSettings
from shuiyuan_auto_reply.features.mention import MentionModel
from shuiyuan_auto_reply.infrastructure.llm import LegacyMentionChatBackend
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel


async def run_worker(persona: str = "wolf_lumine") -> None:
    load_dotenv()
    settings = AppSettings()
    model = await ShuiyuanModel.create(settings.forum.cookie_file)
    chat_model = None
    mention = None
    container = None
    try:
        chat_model = MentionProviderFactory.create(model, persona, settings.providers)
        mention = MentionModel(
            model,
            bot_username=settings.forum.bot_username,
            persona=persona,
            chat_model=chat_model,
            provider_settings=settings.providers,
        )
        container = ApplicationContainer(
            settings,
            model,
            mention.bot_service,
            ChatHandler(LegacyMentionChatBackend(chat_model)),
            managed=[mention],
        )
        await mention.watch_new_action_routine()
    finally:
        if container is not None:
            await container.aclose()
        else:
            try:
                if chat_model is not None:
                    await chat_model.aclose()
            finally:
                await model.close()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

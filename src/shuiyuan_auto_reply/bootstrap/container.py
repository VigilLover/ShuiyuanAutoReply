"""Composition root and idempotent lifecycle owner."""

import asyncio
import logging
from dataclasses import dataclass, field

from shuiyuan_auto_reply.application import BotService, HandlerRegistry
from shuiyuan_auto_reply.application.handlers import ChatHandler
from shuiyuan_auto_reply.database.neo4j_mgr import close_global_async_neo4j_manager
from shuiyuan_auto_reply.database.postgres_memory_mgr import (
    close_global_async_postgres_memory_manager,
)
from shuiyuan_auto_reply.database.postgres_record_mgr import (
    close_global_async_postgres_record_manager,
)
from shuiyuan_auto_reply.features.mention.image_generation import close_shared_session
from shuiyuan_auto_reply.infrastructure.llm import LegacyMentionChatBackend
from shuiyuan_auto_reply.infrastructure.persistence import InMemorySessionRepository
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .providers import MentionProviderFactory
from .settings import AppSettings


@dataclass(slots=True)
class ApplicationContainer:
    settings: AppSettings
    forum_model: ShuiyuanModel
    bot_service: BotService
    chat_handler: ChatHandler
    managed: list[object] = field(default_factory=list)
    _closed: bool = False
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    async def for_api(cls, settings: AppSettings | None = None) -> "ApplicationContainer":
        current = settings or AppSettings()
        current.providers.validate_api()
        forum_model = await ShuiyuanModel.create(current.forum.cookie_file)
        try:
            chat_model = MentionProviderFactory.create_api(
                forum_model, "wolf_lumine", current.providers
            )
            chat_handler = ChatHandler(LegacyMentionChatBackend(chat_model))
            service = BotService(
                InMemorySessionRepository(), HandlerRegistry([chat_handler])
            )
            return cls(current, forum_model, service, chat_handler)
        except BaseException:
            await forum_model.close()
            raise

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            closers = []
            for resource in reversed(self.managed):
                if (close := getattr(resource, "aclose", None)) is not None:
                    closers.append(close)
            closers.extend(
                [
                    self.chat_handler.aclose,
                    close_global_async_postgres_memory_manager,
                    close_global_async_postgres_record_manager,
                    close_global_async_neo4j_manager,
                    close_shared_session,
                    self.forum_model.close,
                ]
            )
            for close in closers:
                try:
                    await close()
                except Exception:
                    logging.exception("Failed to close application resource")

"""Composition root and idempotent lifecycle owner."""

import asyncio
import logging
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any

from shuiyuan_auto_reply.application import BotService, HandlerRegistry
from shuiyuan_auto_reply.application.handlers import ChatHandler, HelpHandler, PetHandler
from shuiyuan_auto_reply.database.neo4j_mgr import close_global_async_neo4j_manager
from shuiyuan_auto_reply.database.postgres_memory_mgr import (
    close_global_async_postgres_memory_manager,
)
from shuiyuan_auto_reply.database.postgres_record_mgr import (
    close_global_async_postgres_record_manager,
)
from shuiyuan_auto_reply.features.mention.image_generation import close_shared_session
from shuiyuan_auto_reply.infrastructure.llm import LegacyMentionChatBackend
from shuiyuan_auto_reply.infrastructure.persistence import (
    LocalSecretVault,
    SQLiteExecutionObserver,
    SQLiteSessionRepository,
    SQLiteStateStore,
)
from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.domain import ConversationRef, ReplyRequest, ReplyResult
from shuiyuan_auto_reply.features.mention.mention_pet_model import MentionPetModel
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .providers import MentionProviderFactory
from .settings import AppSettings, ProviderSettings

DEEPSEEK_VISION_MODEL = "deepseek-v4-flash-vision-exp"


class _UnavailableChatBackend:
    """Keeps the management UI usable until a web Provider is configured."""

    async def reply(self, _request: ReplyRequest, _history=()) -> ReplyResult:
        raise RuntimeError("网页对话 Provider 尚未配置，请先在设置页保存并应用 API Key。")

    async def clear(self, _conversation: ConversationRef) -> None:
        return None

    async def aclose(self) -> None:
        return None


@dataclass(eq=False)
class _RuntimeBinding:
    service: BotService
    handler: ChatHandler
    active_requests: int = 0
    retired: bool = False
    closed: bool = False


class _SwappableBotService:
    """Pins each request to one runtime and retires old runtimes after their last request."""

    owns_handlers = True

    def __init__(self, service: BotService, handler: ChatHandler) -> None:
        self._current = _RuntimeBinding(service, handler)
        self._bindings = {self._current}
        self._lock = asyncio.Lock()

    @staticmethod
    async def _close_binding(binding: _RuntimeBinding) -> None:
        try:
            await binding.handler.aclose()
        except Exception:
            logging.exception("Failed to close a retired web runtime")

    async def _acquire(self) -> _RuntimeBinding:
        async with self._lock:
            binding = self._current
            binding.active_requests += 1
            return binding

    async def _release(self, binding: _RuntimeBinding) -> None:
        close = False
        async with self._lock:
            binding.active_requests -= 1
            if binding.retired and binding.active_requests == 0 and not binding.closed:
                binding.closed = True
                self._bindings.discard(binding)
                close = True
        if close:
            await self._close_binding(binding)

    async def reply(self, request):
        binding = await self._acquire()
        try:
            return await binding.service.reply(request)
        finally:
            await self._release(binding)

    async def clear_conversation(self, conversation) -> None:
        binding = await self._acquire()
        try:
            await binding.service.clear_conversation(conversation)
        finally:
            await self._release(binding)

    async def swap(self, service: BotService, handler: ChatHandler) -> None:
        old_to_close = None
        async with self._lock:
            old = self._current
            old.retired = True
            current = _RuntimeBinding(service, handler)
            self._current = current
            self._bindings.add(current)
            if old.active_requests == 0 and not old.closed:
                old.closed = True
                self._bindings.discard(old)
                old_to_close = old
        if old_to_close is not None:
            await self._close_binding(old_to_close)

    async def aclose(self) -> None:
        async with self._lock:
            bindings = tuple(self._bindings)
            self._bindings.clear()
            for binding in bindings:
                binding.closed = True
        for binding in bindings:
            await self._close_binding(binding)


@dataclass(slots=True)
class ApplicationContainer:
    settings: AppSettings
    forum_model: ShuiyuanModel
    bot_service: BotService
    chat_handler: ChatHandler
    managed: list[object] = field(default_factory=list)
    state_store: SQLiteStateStore | None = None
    secret_vault: LocalSecretVault | None = None
    _closed: bool = False
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @staticmethod
    def _build_web_service(chat_model, provider_settings, state_store):
        backend = (
            LegacyMentionChatBackend(chat_model)
            if chat_model is not None
            else _UnavailableChatBackend()
        )
        chat_handler = ChatHandler(backend)
        pet_model = None

        async def web_help(_context):
            return (
                "这是本地网页对话。直接输入内容即可聊天；输入【rua】可以与小狼互动，"
                "输入【帮助】可以再次查看本说明。需要论坛信息时，我会按需使用只读搜索工具。"
            )

        async def web_rua(context):
            nonlocal pet_model
            if pet_model is None:
                pet_model = MentionPetModel(
                    persona="wolf_lumine", provider_settings=provider_settings
                )
            text = context.request.content.split("【rua】", 1)[-1].strip()
            actor = context.request.actor
            return await pet_model.get_rua_response(
                username=actor.username,
                name=actor.display_name or "",
                user_text=text,
            )

        service = BotService(
            SQLiteSessionRepository(state_store),
            HandlerRegistry(
                [
                    HelpHandler(lambda raw: "【帮助】" in raw, web_help, 10),
                    PetHandler(lambda raw: "【rua】" in raw, web_rua, 20),
                    chat_handler,
                ]
            ),
            observer_factory=lambda: SQLiteExecutionObserver(
                state_store,
                provider=provider_settings.mention_provider,
                model={
                    "openrouter": provider_settings.openrouter_mention_model,
                    "deepseek": provider_settings.deepseek_model,
                    "tongyi": provider_settings.dashscope_model,
                    "mimo": provider_settings.mimo_model,
                }.get(provider_settings.mention_provider),
            ),
        )
        return chat_handler, service

    @staticmethod
    def _profile_defaults(settings: AppSettings) -> dict[str, Any]:
        prompt = FilePromptRepository().load(
            "wolf_lumine", set(), PromptScope.WEB
        ).system_prompt
        return {
            "provider": "deepseek",
            "model": DEEPSEEK_VISION_MODEL,
            "fallback_model": None,
            "system_prompt": prompt,
            "enabled_tools": None,
            "disabled_mcp_tools": [],
        }

    async def _settings_for_profile(self, scope: str, profile: dict) -> ProviderSettings:
        provider = "deepseek"
        settings = replace(
            self.settings.providers,
            mention_provider=provider,
            deepseek_model=DEEPSEEK_VISION_MODEL,
        )
        secret = (
            await self.secret_vault.get(f"{scope}:{provider}")
            if self.secret_vault
            else None
        )
        key_fields = {
            "openrouter": "openrouter_api_key",
            "deepseek": "deepseek_api_key",
            "tongyi": "dashscope_api_key",
            "mimo": "mimo_api_key",
        }
        if secret:
            settings = replace(settings, **{key_fields[provider]: secret})
        return settings

    @classmethod
    async def for_api(cls, settings: AppSettings | None = None) -> "ApplicationContainer":
        current = settings or AppSettings()
        state_store = SQLiteStateStore()
        await state_store.initialize()
        secret_vault = LocalSecretVault(state_store)
        forum_model = await ShuiyuanModel.create(current.forum.cookie_file)
        try:
            container = cls(
                current,
                forum_model,
                None,  # type: ignore[arg-type]
                None,  # type: ignore[arg-type]
                state_store=state_store,
                secret_vault=secret_vault,
            )
            profile = await state_store.get_profile(
                "web", cls._profile_defaults(current)
            )
            effective = await container._settings_for_profile("web", profile["active"])
            configured_keys = {
                "openrouter": effective.openrouter_api_key,
                "deepseek": effective.deepseek_api_key,
                "tongyi": effective.dashscope_api_key,
                "mimo": effective.mimo_api_key,
            }
            if not configured_keys.get(effective.mention_provider):
                logging.warning(
                    "Web runtime is waiting for a %s API key",
                    effective.mention_provider,
                )
                chat_model = None
            else:
                factory_method = (
                    MentionProviderFactory.create_api
                    if effective.mention_provider == "openrouter"
                    else MentionProviderFactory.create
                )
                chat_model = factory_method(
                    forum_model,
                    "wolf_lumine",
                    effective,
                    prompt_scope=PromptScope.WEB,
                    enabled_tools=(
                        set(profile["active"]["enabled_tools"])
                        if profile["active"].get("enabled_tools") is not None
                        else None
                    ),
                    disabled_mcp_tools=set(
                        profile["active"].get("disabled_mcp_tools", [])
                    ),
                    state_store=state_store,
                    system_prompt_override=profile["active"].get("system_prompt"),
                )
            chat_handler, service = cls._build_web_service(
                chat_model, effective, state_store
            )
            container.bot_service = _SwappableBotService(service, chat_handler)  # type: ignore[assignment]
            container.chat_handler = chat_handler
            return container
        except BaseException:
            await forum_model.close()
            raise

    async def prepare_runtime_profile(self, scope: str, profile: dict):
        if scope != "web" or self.state_store is None:
            raise ValueError("Only the web runtime can be switched in this process")
        settings = await self._settings_for_profile(scope, profile)
        enabled = profile.get("enabled_tools")
        candidate = MentionProviderFactory.create(
            self.forum_model,
            "wolf_lumine",
            settings,
            prompt_scope=PromptScope.WEB,
            enabled_tools=set(enabled) if enabled is not None else None,
            disabled_mcp_tools=set(profile.get("disabled_mcp_tools", [])),
            state_store=self.state_store,
            system_prompt_override=profile.get("system_prompt"),
        )
        new_handler, new_service = self._build_web_service(candidate, settings, self.state_store)
        return new_handler, new_service

    async def prepare_forum_runtime_profile(self, profile: dict):
        if self.state_store is None:
            raise RuntimeError("Local state is not configured")
        settings = await self._settings_for_profile("forum", profile)
        enabled = profile.get("enabled_tools")
        return MentionProviderFactory.create(
            self.forum_model,
            "wolf_lumine",
            settings,
            prompt_scope=PromptScope.FORUM,
            enabled_tools=set(enabled) if enabled is not None else None,
            disabled_mcp_tools=set(profile.get("disabled_mcp_tools", [])),
            state_store=self.state_store,
            system_prompt_override=profile.get("system_prompt"),
        )

    async def activate_prepared_runtime(self, prepared) -> None:
        new_handler, new_service = prepared
        swap = getattr(self.bot_service, "swap", None)
        if swap is not None:
            await swap(new_service, new_handler)
        else:
            self.managed.append(self.chat_handler)
            self.bot_service = new_service
        self.chat_handler = new_handler

    async def apply_runtime_profile(self, scope: str, profile: dict) -> None:
        prepared = await self.prepare_runtime_profile(scope, profile)
        await self.activate_prepared_runtime(prepared)

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            closers = []
            for resource in reversed(self.managed):
                if (close := getattr(resource, "aclose", None)) is not None:
                    closers.append(close)
            service_close = getattr(self.bot_service, "aclose", None)
            if service_close is not None and getattr(self.bot_service, "owns_handlers", False):
                closers.append(service_close)
            else:
                closers.append(self.chat_handler.aclose)
            closers.extend(
                [
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

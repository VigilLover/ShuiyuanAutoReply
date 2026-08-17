"""Single channel-independent application entry point."""

from shuiyuan_auto_reply.domain import DispatchMode, ReplyRequest, ReplyResult

from .dispatch import BotContext, HandlerRegistry
from .events import reset_execution_context, set_execution_context
from .ports.session import SessionRepository


class BotService:
    def __init__(
        self,
        sessions: SessionRepository,
        handlers: HandlerRegistry,
        *,
        chat_handler_name: str = "chat",
        observer_factory=None,
    ) -> None:
        self._sessions = sessions
        self._handlers = handlers
        self._chat_handler_name = chat_handler_name
        self._observer_factory = observer_factory

    async def reply(self, request: ReplyRequest) -> ReplyResult:
        if not request.request_id or not request.content.strip():
            raise ValueError("request_id and non-empty content are required")

        observer = self._observer_factory() if self._observer_factory else None
        run_id = await observer.start(request) if observer else None
        tokens = set_execution_context(observer, run_id, request.actor.memory_id)
        try:
            history = tuple(await self._sessions.load(request.conversation))
            context = BotContext(request=request, history=history)

            if request.dispatch_mode is DispatchMode.CHAT_ONLY:
                handler = self._handlers.by_name(self._chat_handler_name)
                result = await handler.handle(context)
            else:
                handler, result = await self._dispatch_auto(context)

            if handler.name == "clear":
                await self._sessions.clear(request.conversation)
            else:
                await self._sessions.append(request.conversation, request, result)
            if observer:
                await observer.finish(result)
            return result
        except Exception as exc:
            if observer:
                await observer.finish(None, exc)
            raise
        finally:
            reset_execution_context(tokens)

    async def _dispatch_auto(self, context: BotContext):
        for handler in self._handlers.handlers:
            if await handler.matches(context):
                return handler, await handler.handle(context)
        raise LookupError("No message handler matched the request")

    async def clear_conversation(self, conversation) -> None:
        await self._sessions.clear(conversation)
        for handler in self._handlers.handlers:
            clear = getattr(handler, "clear_conversation", None)
            if clear is not None:
                await clear(conversation)

from .callback import CallbackMessageHandler


class PollHandler(CallbackMessageHandler):
    def __init__(self, predicate, callback, priority: int = 60) -> None:
        super().__init__(
            name="poll", priority=priority, predicate=predicate, callback=callback
        )

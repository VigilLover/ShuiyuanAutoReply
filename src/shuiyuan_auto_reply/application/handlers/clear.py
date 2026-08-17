from .callback import CallbackMessageHandler


class ClearHandler(CallbackMessageHandler):
    def __init__(self, predicate, callback, priority: int = 30) -> None:
        super().__init__(
            name="clear", priority=priority, predicate=predicate, callback=callback
        )

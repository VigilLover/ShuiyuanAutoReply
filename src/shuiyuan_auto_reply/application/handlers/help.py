from .callback import CallbackMessageHandler


class HelpHandler(CallbackMessageHandler):
    def __init__(self, predicate, callback, priority: int = 10) -> None:
        super().__init__(
            name="help", priority=priority, predicate=predicate, callback=callback
        )

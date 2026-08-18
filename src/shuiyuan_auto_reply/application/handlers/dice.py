from .callback import CallbackMessageHandler


class DiceHandler(CallbackMessageHandler):
    def __init__(self, predicate, callback, priority: int = 50) -> None:
        super().__init__(
            name="dice", priority=priority, predicate=predicate, callback=callback
        )

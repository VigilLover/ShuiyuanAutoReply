from .callback import CallbackMessageHandler


class PetHandler(CallbackMessageHandler):
    def __init__(self, predicate, callback, priority: int = 20) -> None:
        super().__init__(
            name="rua", priority=priority, predicate=predicate, callback=callback
        )

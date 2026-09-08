from typing import Any, Protocol


class ToolProvider(Protocol):
    def build_tools(self, context: Any) -> list[Any]: ...

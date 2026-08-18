class StaticToolProvider:
    def __init__(self, tools) -> None:
        self._tools = tuple(tools)

    def build_tools(self, context) -> list:
        return list(self._tools)


__all__ = ["StaticToolProvider"]

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class MemoryScope:
    namespace: str


@dataclass(frozen=True, slots=True)
class MemoryCommand:
    payload: dict[str, Any]


class LongTermMemoryPort(Protocol):
    async def search(self, scope: MemoryScope, query: str, limit: int) -> str: ...
    async def manage(self, command: MemoryCommand) -> str: ...

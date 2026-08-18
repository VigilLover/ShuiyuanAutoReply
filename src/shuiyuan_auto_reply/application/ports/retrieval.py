from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StyleExample:
    text: str
    score: float | None = None


class StyleRetriever(Protocol):
    async def search(
        self, persona_id: str, query: str, limit: int
    ) -> list[StyleExample]: ...

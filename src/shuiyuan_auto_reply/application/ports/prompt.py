from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PromptScope(str, Enum):
    FORUM = "forum"
    WEB = "web"


@dataclass(frozen=True, slots=True)
class PromptBundle:
    persona_id: str
    system_prompt: str
    version: str = "1"


class PromptRepository(Protocol):
    def load(
        self,
        persona_id: str,
        capabilities: set[str],
        scope: PromptScope = PromptScope.FORUM,
    ) -> PromptBundle: ...

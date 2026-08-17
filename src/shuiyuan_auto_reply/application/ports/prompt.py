from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PromptBundle:
    persona_id: str
    system_prompt: str
    version: str = "1"


class PromptRepository(Protocol):
    def load(self, persona_id: str, capabilities: set[str]) -> PromptBundle: ...

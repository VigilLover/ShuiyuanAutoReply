"""Provider capabilities without depending on a specific LLM SDK."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    tools: bool = True
    multimodal: bool = False
    streaming: bool = False

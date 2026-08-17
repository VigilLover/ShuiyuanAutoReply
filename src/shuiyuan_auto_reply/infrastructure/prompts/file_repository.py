"""UTF-8 text-resource prompt repository.

Text is read without YAML parsing or whitespace normalization so a prompt can be
verified byte-for-byte against its characterization snapshot.
"""

import json
from importlib import resources

from shuiyuan_auto_reply.application.ports.prompt import PromptBundle


class FilePromptRepository:
    def __init__(self, package: str = "shuiyuan_auto_reply.prompts") -> None:
        root = resources.files(package)
        manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
        self._root = root
        self._version = str(manifest["version"])
        self._default = manifest["default_persona"]
        self._personas: dict[str, str] = manifest["personas"]
        self._system_template: str = manifest["system_template"]
        self._capabilities: dict[str, str] = manifest.get("capabilities", {})

    @staticmethod
    def _read_text(resource) -> str:
        text = resource.read_text(encoding="utf-8")
        return text[:-1] if text.endswith("\n") else text

    def load(self, persona_id: str, capabilities: set[str]) -> PromptBundle:
        selected = persona_id if persona_id in self._personas else self._default
        persona = self._read_text(self._root.joinpath(self._personas[selected]))
        template = self._read_text(self._root.joinpath(self._system_template))
        capability_text = ""
        if "multimodal" in capabilities:
            capability_text = self._read_text(
                self._root.joinpath(self._capabilities["multimodal"])
            ) + "\n\n"
        system_prompt = (
            template.replace("{{persona_prompt}}", persona)
            .replace("{{persona_id}}", persona_id)
            .replace("{{multimodal_rules}}", capability_text)
        )
        return PromptBundle(selected, system_prompt, self._version)

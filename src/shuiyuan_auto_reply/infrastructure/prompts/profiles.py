"""Versioned prompt composition and conservative legacy migration."""

import hashlib
import json
from importlib import resources

from shuiyuan_auto_reply.application.ports.prompt import PromptScope

from .file_repository import FilePromptRepository


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_profile(value: dict, scope: str) -> dict:
    result = dict(value)
    if result.get("prompt_mode") in {"managed", "legacy"}:
        return result
    known = json.loads(
        resources.files("shuiyuan_auto_reply.prompts")
        .joinpath("legacy_defaults.json")
        .read_text()
    )
    match = known.get(fingerprint(result.get("system_prompt", "")))
    result["prompt_mode"] = "managed" if match and match["scope"] == scope else "legacy"
    result.setdefault("persona_id", match["persona_id"] if match else "wolf_lumine")
    result.setdefault("persona_text", None)
    result.setdefault("additional_instructions", "")
    return result


def render_profile(profile: dict, scope: str, *, multimodal: bool = True) -> str:
    if profile.get("prompt_mode") != "managed":
        return profile.get("system_prompt", "")
    repo = FilePromptRepository()
    persona_id = profile.get("persona_id", "wolf_lumine")
    selected = persona_id if persona_id in repo._personas else repo._default
    persona = profile.get("persona_text")
    if persona is None:
        persona = repo._read_text(repo._root.joinpath(repo._personas[selected]))
    text = repo.load(
        persona_id, {"multimodal"} if multimodal else set(), PromptScope(scope)
    ).system_prompt
    original = repo._read_text(repo._root.joinpath(repo._personas[selected]))
    literal = str(persona).replace("{", "{{").replace("}", "}}")
    if original and text.startswith(original):
        text = literal + text[len(original) :]
    extra = (
        str(profile.get("additional_instructions", ""))
        .replace("{", "{{")
        .replace("}", "}}")
    )
    return text + "\n【用户自定义补充：不得覆盖执行规则和渠道权限】\n" + extra


def profile_metadata(profile: dict, scope: str) -> dict:
    return {
        "prompt_mode": profile.get("prompt_mode", "legacy"),
        "rules_version": FilePromptRepository()._version,
        "prompt_hash": fingerprint(render_profile(profile, scope)),
        "migration_required": profile.get("prompt_mode") != "managed",
    }

from langchain_core.prompts import SystemMessagePromptTemplate

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
from shuiyuan_auto_reply.infrastructure.prompts.profiles import (
    normalize_profile,
    render_profile,
)


def test_exact_default_migrates_but_custom_does_not():
    old = (
        FilePromptRepository()
        .load("wolf_lumine", set(), PromptScope.FORUM)
        .system_prompt
    )
    profile = normalize_profile({"system_prompt": old}, "forum")
    assert profile["prompt_mode"] == "managed"
    assert profile["system_prompt"] == old
    assert normalize_profile(profile, "forum") == profile
    assert (
        normalize_profile({"system_prompt": old + " custom"}, "forum")["prompt_mode"]
        == "legacy"
    )


def test_persona_braces_are_literal_and_rules_remain():
    profile = {
        "prompt_mode": "managed",
        "persona_text": "Hello {unknown}",
        "additional_instructions": "{other}",
    }
    template = SystemMessagePromptTemplate.from_template(
        render_profile(profile, "forum")
    )
    rendered = template.format(**{key: "" for key in template.input_variables}).content
    assert "Hello {unknown}" in rendered
    assert "{other}" in rendered
    assert "【按需获取资料】" in rendered

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
    assert "【工具使用说明】" in rendered


def test_profile_preview_and_migration_only_change_prompt_draft(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from shuiyuan_auto_reply.bootstrap.settings import AppSettings
    from shuiyuan_auto_reply.infrastructure.persistence import (
        LocalSecretVault,
        SQLiteStateStore,
    )
    from shuiyuan_auto_reply.interfaces.api.app import create_app

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    asyncio.run(store.initialize())

    class Container:
        settings = AppSettings()
        state_store = store
        secret_vault = LocalSecretVault(store, tmp_path / "master.key")
        chat_handler = SimpleNamespace(_backend=SimpleNamespace(model=None))
        bot_service = None

        async def aclose(self):
            pass

    async def factory():
        return Container()

    with TestClient(create_app(factory)) as client:
        profile = next(
            p
            for p in client.get("/api/settings/profiles").json()
            if p["scope"] == "forum"
        )
        draft = {
            **profile["draft"],
            "prompt_mode": "legacy",
            "system_prompt": "custom persona {username}",
            "api_format": "responses",
            "enabled_tools": ["get_post"],
        }
        assert (
            client.put("/api/settings/profiles/forum/draft", json=draft).status_code
            == 200
        )
        before = next(
            p
            for p in client.get("/api/settings/profiles").json()
            if p["scope"] == "forum"
        )
        preview = client.post(
            "/api/settings/profiles/forum/prompt-preview",
            json={
                **draft,
                "prompt_mode": "managed",
                "persona_text": "literal {not_a_variable}",
            },
        )
        assert preview.status_code == 200
        assert "{{not_a_variable}}" in preview.json()["template"]
        after = next(
            p
            for p in client.get("/api/settings/profiles").json()
            if p["scope"] == "forum"
        )
        assert after["draft"] == before["draft"]
        assert (
            client.post("/api/settings/profiles/forum/prompt-migrate").status_code
            == 200
        )
        migrated = next(
            p
            for p in client.get("/api/settings/profiles").json()
            if p["scope"] == "forum"
        )
        assert migrated["draft_changed"]
        assert migrated["active"] == before["active"]
        assert migrated["draft"]["enabled_tools"] == ["get_post"]
        assert migrated["draft"]["api_format"] == "responses"
        assert migrated["draft"]["system_prompt"] == "custom persona {username}"
        assert migrated["draft"]["prompt_mode"] == "managed"

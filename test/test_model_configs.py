"""Stored chat/image model configurations: CRUD, activation and resolution."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from shuiyuan_auto_reply.application import BotService, HandlerRegistry
from shuiyuan_auto_reply.bootstrap import ApplicationContainer, AppSettings
from shuiyuan_auto_reply.bootstrap.providers import apply_profile_endpoint
from shuiyuan_auto_reply.features.mention.image_generation import (
    resolve_image_endpoint,
)
from shuiyuan_auto_reply.infrastructure.persistence import (
    LocalSecretVault,
    SQLiteSessionRepository,
    SQLiteStateStore,
)
from shuiyuan_auto_reply.infrastructure.persistence.model_configs import (
    active_image_endpoint,
    image_endpoint_resolver,
)
from shuiyuan_auto_reply.interfaces.api.app import create_app


class HistoryChat:
    name = "chat"
    priority = 40

    async def matches(self, _context):
        return True

    async def handle(self, context):
        from shuiyuan_auto_reply.domain import ReplyResult

        return ReplyResult("ok")


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A TestClient whose container is a stub: no runtime is actually built."""
    monkeypatch.setenv("SHUIYUAN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_GEN_API_URL", "https://env.example/v1")
    monkeypatch.setenv("IMAGE_GEN_API_KEY", "env-key-1234")
    monkeypatch.setenv("IMAGE_GEN_MODEL", "env-image-model")
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    asyncio.run(store.initialize())
    vault = LocalSecretVault(store, tmp_path / "master.key")
    store.model_config_resolver = image_endpoint_resolver(store, vault)

    class Container:
        settings = AppSettings()
        chat_handler = SimpleNamespace(_backend=SimpleNamespace(model=None))
        state_store = store
        secret_vault = vault
        bot_service = BotService(
            SQLiteSessionRepository(store), HandlerRegistry([HistoryChat()])
        )

        async def aclose(self):
            return None

    async def factory():
        return Container()

    with TestClient(create_app(factory)) as client:
        yield SimpleNamespace(client=client, store=store, vault=vault)

    store.model_config_resolver = None


@pytest.fixture
def runtime_api(tmp_path, monkeypatch):
    """Use the real runtime builder without making provider or forum requests."""
    monkeypatch.setenv("SHUIYUAN_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-default-0000")
    monkeypatch.setenv("DEEPSEEK_MENTION_API_FORMAT", "chat_completions")

    async def factory():
        settings = replace(
            AppSettings(),
            providers=replace(
                AppSettings().providers, deepseek_api_key="sk-default-0000"
            ),
        )
        return await ApplicationContainer.for_api(settings)

    with TestClient(create_app(factory)) as client:
        yield client


def runtime_model(client):
    return client.app.state.container.chat_handler._backend.model


def sample(kind="chat", name="聚合站", **overrides):
    payload = {
        "kind": kind,
        "name": name,
        "base_url": "https://4router.example/v1",
        "model": "gpt-image-2.5-sunburst" if kind == "image" else "deepseek-v4-vision",
        "api_format": "chat_completions" if kind == "chat" else None,
        "api_key": "sk-config-9999",
    }
    payload.update(overrides)
    return payload


def create(api, payload):
    response = api.client.post("/api/settings/model-configs", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_listing_keeps_the_read_only_default_first(api):
    entries = api.client.get("/api/settings/model-configs").json()
    assert entries["active"] == {
        "web": "default",
        "forum": "default",
        "image": "default",
    }
    for kind in ("chat", "image"):
        assert entries[kind][0]["source"] == "default"
        assert entries[kind][0]["id"] == "default"
    assert entries["image"][0]["base_url"] == "https://env.example/v1"
    assert entries["image"][0]["model"] == "env-image-model"
    assert entries["image"][0]["secret"]["source"] == "environment"

    config_id = create(api, sample("image"))
    listed = api.client.get("/api/settings/model-configs").json()
    stored = next(item for item in listed["image"] if item["id"] == config_id)
    assert stored["source"] == "custom"
    assert stored["secret"] == {
        "configured": True,
        "last_four": "9999",
        "source": "ui",
    }


def test_activate_writes_the_profile_and_can_switch_back(api):
    config_id = create(api, sample("chat"))
    activated = api.client.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "web"}
    )
    assert activated.status_code == 200, activated.text
    assert activated.json()["config_id"] == config_id

    profile = next(
        item
        for item in api.client.get("/api/settings/profiles").json()
        if item["scope"] == "web"
    )
    assert profile["draft"]["model"] == "deepseek-v4-vision"
    assert profile["draft"]["base_url"] == "https://4router.example/v1"
    assert profile["draft"]["api_format"] == "chat_completions"
    assert (
        api.client.get("/api/settings/model-configs").json()["active"]["web"]
        == config_id
    )
    # The forum scope is untouched: activation is per application.
    assert (
        api.client.get("/api/settings/model-configs").json()["active"]["forum"]
        == "default"
    )
    forum = next(
        item
        for item in api.client.get("/api/settings/profiles").json()
        if item["scope"] == "forum"
    )
    assert forum["draft"]["base_url"] == ""

    reverted = api.client.post(
        "/api/settings/model-configs/default/activate", json={"scope": "web"}
    )
    assert reverted.status_code == 200
    profile = next(
        item
        for item in api.client.get("/api/settings/profiles").json()
        if item["scope"] == "web"
    )
    assert profile["draft"]["base_url"] == ""
    assert profile["draft"]["model"] == "deepseek-flash"
    assert (
        api.client.get("/api/settings/model-configs").json()["active"]["web"]
        == "default"
    )


def test_real_runtime_switches_between_custom_endpoints_and_default(runtime_api):
    first = runtime_api.post(
        "/api/settings/model-configs",
        json=sample(
            base_url="https://first.example/v1", model="model-a", api_key="sk-first"
        ),
    ).json()["id"]
    second = runtime_api.post(
        "/api/settings/model-configs",
        json=sample(
            base_url="https://second.example/v1",
            model="model-b",
            api_format="responses",
            api_key="sk-second",
        ),
    ).json()["id"]

    for config_id, url, name, key, api_format in (
        (first, "https://first.example/v1", "model-a", "sk-first", "chat_completions"),
        (second, "https://second.example/v1", "model-b", "sk-second", "responses"),
        (
            "default",
            "https://api.deepseek.com",
            "deepseek-flash",
            "sk-default-0000",
            "chat_completions",
        ),
    ):
        response = runtime_api.post(
            f"/api/settings/model-configs/{config_id}/activate",
            json={"scope": "web"},
        )
        assert response.status_code == 200, response.text
        model = runtime_model(runtime_api)
        assert str(model.llm.openai_api_base).rstrip("/") == url
        assert model.llm.model_name == name
        assert model.llm.openai_api_key.get_secret_value() == key
        assert model.api_format.value == api_format
        assert (
            runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
            == config_id
        )


def test_real_runtime_build_failure_keeps_previous_state(
    runtime_api, monkeypatch, caplog
):
    config_id = runtime_api.post("/api/settings/model-configs", json=sample()).json()[
        "id"
    ]
    container = runtime_api.app.state.container
    before = runtime_model(runtime_api)

    async def failing(*args, **kwargs):
        raise RuntimeError("sk-private-failure")

    monkeypatch.setattr(ApplicationContainer, "prepare_runtime_profile", failing)
    response = runtime_api.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "web"}
    )
    assert response.status_code == 400
    assert "sk-private-failure" not in response.text
    assert "sk-private-failure" not in caplog.text
    assert runtime_model(runtime_api) is before
    assert (
        runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
        == "default"
    )
    assert (
        next(
            item
            for item in runtime_api.get("/api/settings/profiles").json()
            if item["scope"] == "web"
        )["active_revision"]
        == 1
    )


def test_real_runtime_commit_failure_keeps_previous_state(runtime_api, monkeypatch):
    config_id = runtime_api.post("/api/settings/model-configs", json=sample()).json()[
        "id"
    ]
    container = runtime_api.app.state.container
    before = runtime_model(runtime_api)
    monkeypatch.setattr(
        container.state_store,
        "activate_model_config_profile",
        AsyncMock(side_effect=RuntimeError("commit failed")),
    )
    with pytest.raises(RuntimeError, match="commit failed"):
        runtime_api.post(
            f"/api/settings/model-configs/{config_id}/activate",
            json={"scope": "web"},
        )
    assert runtime_model(runtime_api) is before
    assert (
        runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
        == "default"
    )


def test_transaction_rolls_back_all_profile_changes(runtime_api):
    config_id = runtime_api.post("/api/settings/model-configs", json=sample()).json()[
        "id"
    ]
    store = runtime_api.app.state.container.state_store
    before = next(
        item
        for item in runtime_api.get("/api/settings/profiles").json()
        if item["scope"] == "web"
    )

    async def abort_active_insert():
        db = await store._connect()
        try:
            await db.execute(
                """CREATE TRIGGER reject_switch BEFORE INSERT ON model_config_active
                BEGIN SELECT RAISE(ABORT, 'injected failure'); END"""
            )
            await db.commit()
        finally:
            await db.close()

    asyncio.run(abort_active_insert())
    previous_runtime = runtime_model(runtime_api)
    with pytest.raises(Exception, match="injected failure"):
        runtime_api.post(
            f"/api/settings/model-configs/{config_id}/activate",
            json={"scope": "web"},
        )
    after = next(
        item
        for item in runtime_api.get("/api/settings/profiles").json()
        if item["scope"] == "web"
    )
    assert after["draft"] == before["draft"]
    assert after["active"] == before["active"]
    assert after["active_revision"] == before["active_revision"]
    assert (
        runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
        == "default"
    )
    assert runtime_model(runtime_api) is previous_runtime


def test_concurrent_switches_keep_runtime_and_database_aligned(runtime_api):
    config_ids = [
        runtime_api.post(
            "/api/settings/model-configs",
            json=sample(base_url=f"https://{name}.example/v1", model=name),
        ).json()["id"]
        for name in ("alpha", "beta")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda config_id: runtime_api.post(
                    f"/api/settings/model-configs/{config_id}/activate",
                    json={"scope": "web"},
                ),
                config_ids,
            )
        )
    assert all(response.status_code == 200 for response in responses)
    active = runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
    expected = "alpha" if active == config_ids[0] else "beta"
    assert (
        str(runtime_model(runtime_api).llm.openai_api_base).rstrip("/")
        == f"https://{expected}.example/v1"
    )


def test_forum_revision_and_active_endpoint_change_together(runtime_api):
    config_id = runtime_api.post(
        "/api/settings/model-configs",
        json=sample(
            base_url="https://forum.example/v1", model="forum-model", api_key=None
        ),
    ).json()["id"]
    response = runtime_api.post(
        f"/api/settings/model-configs/{config_id}/activate",
        json={"scope": "forum"},
    )
    assert response.status_code == 200, response.text
    store = runtime_api.app.state.container.state_store
    profile = asyncio.run(store.get_profile("forum", {}))
    active = asyncio.run(store.active_model_config("forum"))
    assert profile["active_revision"] == response.json()["active_revision"]
    assert (
        profile["active"]["base_url"]
        == active["base_url"]
        == "https://forum.example/v1"
    )
    assert profile["active"]["model"] == active["model"] == "forum-model"
    from shuiyuan_auto_reply.interfaces.worker.main import _forum_provider_settings

    container = runtime_api.app.state.container
    asyncio.run(container.secret_vault.set("forum:deepseek", "sk-forum-ui"))
    effective = asyncio.run(
        _forum_provider_settings(
            container.settings, store, container.secret_vault, profile["active"]
        )
    )
    assert effective.mention_base_url == "https://forum.example/v1"
    assert effective.deepseek_model == "forum-model"
    assert effective.deepseek_api_key == "sk-default-0000"


def test_web_runtime_recovers_selected_endpoint_after_restart(runtime_api):
    config_id = runtime_api.post(
        "/api/settings/model-configs",
        json=sample(base_url="https://restart.example/v1", model="restart-model"),
    ).json()["id"]
    response = runtime_api.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "web"}
    )
    assert response.status_code == 200, response.text
    with TestClient(create_app(ApplicationContainer.for_api)) as restarted:
        model = runtime_model(restarted)
        assert (
            str(model.llm.openai_api_base).rstrip("/") == "https://restart.example/v1"
        )
        assert model.llm.model_name == "restart-model"
        assert model.llm.openai_api_key.get_secret_value() == "sk-config-9999"


def test_restore_default_draft_switches_only_when_applied(runtime_api):
    config_id = runtime_api.post("/api/settings/model-configs", json=sample()).json()[
        "id"
    ]
    runtime_api.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "web"}
    )
    old_runtime = runtime_model(runtime_api)
    response = runtime_api.post("/api/settings/profiles/web/restore-default")
    assert response.status_code == 200
    assert runtime_model(runtime_api) is old_runtime
    assert (
        runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
        == config_id
    )

    applied = runtime_api.post("/api/settings/profiles/web/apply")
    assert applied.status_code == 200, applied.text
    assert (
        runtime_api.get("/api/settings/model-configs").json()["active"]["web"]
        == "default"
    )
    assert (
        str(runtime_model(runtime_api).llm.openai_api_base).rstrip("/")
        == "https://api.deepseek.com"
    )


def test_custom_without_key_uses_environment_not_default_ui_key(runtime_api):
    container = runtime_api.app.state.container
    asyncio.run(container.secret_vault.set("web:deepseek", "sk-web-ui"))
    config_id = runtime_api.post(
        "/api/settings/model-configs",
        json=sample(api_key=None, base_url="https://no-key.example/v1"),
    ).json()["id"]
    custom = runtime_api.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "web"}
    )
    assert custom.status_code == 200, custom.text
    assert (
        runtime_model(runtime_api).llm.openai_api_key.get_secret_value()
        == "sk-default-0000"
    )

    default = runtime_api.post(
        "/api/settings/model-configs/default/activate", json={"scope": "web"}
    )
    assert default.status_code == 200, default.text
    assert (
        runtime_model(runtime_api).llm.openai_api_key.get_secret_value() == "sk-web-ui"
    )


def test_active_configs_cannot_be_edited_or_deleted(api):
    config_id = create(api, sample())
    api.client.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "forum"}
    )
    assert (
        api.client.put(
            f"/api/settings/model-configs/{config_id}", json=sample(model="changed")
        ).status_code
        == 409
    )
    assert (
        api.client.delete(f"/api/settings/model-configs/{config_id}").status_code == 409
    )
    assert (
        asyncio.run(api.store.model_config(config_id))["model"] == "deepseek-v4-vision"
    )
    image_id = create(api, sample("image"))
    api.client.post(
        f"/api/settings/model-configs/{image_id}/activate", json={"scope": "image"}
    )
    assert (
        api.client.put(
            f"/api/settings/model-configs/{image_id}",
            json=sample("image", model="changed"),
        ).status_code
        == 409
    )
    assert (
        api.client.delete(f"/api/settings/model-configs/{image_id}").status_code == 409
    )


def test_image_activation_resolves_for_the_tool(api):
    config_id = create(api, sample("image"))
    api.client.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "image"}
    )
    assert api.client.get("/api/settings/model-configs").json()["active"]["image"] == (
        config_id
    )
    stored = asyncio.run(active_image_endpoint(api.store, api.vault))
    assert stored["base_url"] == "https://4router.example/v1"
    assert stored["api_key"] == "sk-config-9999"
    assert asyncio.run(resolve_image_endpoint(api.store)) == (
        "https://4router.example/v1",
        "sk-config-9999",
        "gpt-image-2.5-sunburst",
    )

    api.client.post(
        "/api/settings/model-configs/default/activate", json={"scope": "image"}
    )
    assert asyncio.run(active_image_endpoint(api.store, api.vault)) is None
    assert asyncio.run(resolve_image_endpoint(api.store)) == (
        "https://env.example/v1",
        "env-key-1234",
        "env-image-model",
    )


def test_image_resolution_falls_back_per_field(api):
    """A stored entry keeps its endpoint but may reuse the deployment key."""
    config_id = create(api, sample("image", name="无密钥", api_key=None))
    api.client.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "image"}
    )
    stored = asyncio.run(active_image_endpoint(api.store, api.vault))
    assert stored["api_key"] == ""
    # Call time resolution fills the gap from the environment, field by field.
    assert asyncio.run(resolve_image_endpoint(api.store)) == (
        "https://4router.example/v1",
        "env-key-1234",
        "gpt-image-2.5-sunburst",
    )


def test_activate_rejects_unknown_and_wrong_kind(api):
    config_id = create(api, sample("image"))
    response = api.client.post(
        f"/api/settings/model-configs/{config_id}/activate", json={"scope": "forum"}
    )
    assert response.status_code == 404
    assert (
        api.client.post(
            "/api/settings/model-configs/missing/activate", json={"scope": "image"}
        ).status_code
        == 404
    )


def test_update_delete_and_key_retention(api):
    config_id = create(api, sample("chat"))
    updated = api.client.put(
        f"/api/settings/model-configs/{config_id}",
        json=sample("chat", name="改名", model="另一个模型", api_key=None),
    )
    assert updated.status_code == 200
    entry = next(
        item
        for item in api.client.get("/api/settings/model-configs").json()["chat"]
        if item["id"] == config_id
    )
    assert entry["name"] == "改名"
    assert entry["model"] == "另一个模型"
    assert entry["secret"]["last_four"] == "9999"  # unchanged without a new key

    assert (
        api.client.delete(f"/api/settings/model-configs/{config_id}").status_code == 200
    )
    assert all(
        item["id"] != config_id
        for item in api.client.get("/api/settings/model-configs").json()["chat"]
    )


def test_config_validation_rejects_bad_input(api):
    for payload, detail in (
        (sample("chat", base_url="ftp://x/v1"), "http"),
        (sample("chat", name="  "), "名称"),
        (sample("chat", model=" "), "模型"),
        (sample(kind="video"), None),
    ):
        response = api.client.post("/api/settings/model-configs", json=payload)
        assert response.status_code in {400, 422}, response.text
        if detail:
            assert detail in str(response.json()["detail"])


def test_probe_reports_models_and_a_missing_target(api, monkeypatch):
    calls = {}

    async def fake_probe(base_url, api_key):
        calls["base_url"], calls["api_key"] = base_url, api_key
        return ["alpha", "gpt-image-2.5-sunburst"]

    monkeypatch.setattr(
        "shuiyuan_auto_reply.interfaces.api.routes.model_configs.list_provider_models",
        fake_probe,
    )
    found = api.client.post(
        "/api/settings/model-configs/probe",
        json={
            "kind": "image",
            "base_url": "https://4router.example/v1/",
            "api_key": "sk-probe",
            "model": "gpt-image-2.5-sunburst",
        },
    ).json()
    assert found["ok"] and found["model_present"]
    assert calls == {
        "base_url": "https://4router.example/v1/",
        "api_key": "sk-probe",
    }

    missing = api.client.post(
        "/api/settings/model-configs/probe",
        json={
            "kind": "chat",
            "base_url": "https://4router.example/v1",
            "model": "not-listed",
        },
    ).json()
    assert missing["ok"] and not missing["model_present"]
    assert "没有" in missing["message"]


def test_probe_surfaces_endpoint_failures(api, monkeypatch):
    async def failing(base_url, api_key):
        raise ValueError("HTTP 401")

    monkeypatch.setattr(
        "shuiyuan_auto_reply.interfaces.api.routes.model_configs.list_provider_models",
        failing,
    )
    failed = api.client.post(
        "/api/settings/model-configs/probe",
        json={"kind": "chat", "base_url": "https://4router.example/v1"},
    ).json()
    assert not failed["ok"] and "401" in failed["message"]


def test_probe_rejects_a_base_url_without_a_scheme(api):
    invalid = api.client.post(
        "/api/settings/model-configs/probe",
        json={"kind": "chat", "base_url": "4router.example"},
    ).json()
    assert not invalid["ok"] and "http" in invalid["message"]


def test_missing_tables_degrade_to_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUIYUAN_STATE_DIR", str(tmp_path))
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    asyncio.run(store.initialize())

    async def drop():
        db = await store._connect()
        try:
            await db.execute("DROP TABLE model_configs")
            await db.execute("DROP TABLE model_config_active")
            await db.commit()
        finally:
            await db.close()

    asyncio.run(drop())
    assert asyncio.run(store.list_model_configs("chat")) == []
    assert asyncio.run(store.active_model_config("web")) is None


def test_base_url_and_model_reach_the_chat_client():
    from dataclasses import replace

    from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings
    from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
        DEEPSEEK_BASE_URL,
        _mk_deepseek_llm,
    )

    settings = replace(
        ProviderSettings(),
        deepseek_api_key="sk-x",
        deepseek_model="aggregator-vision",
        mention_base_url="https://4router.example/v1/",
    )
    client = _mk_deepseek_llm("sk-x", settings.deepseek_model, settings)
    assert client.openai_api_base == "https://4router.example/v1"
    assert client.model_name == "aggregator-vision"

    official = _mk_deepseek_llm("sk-x", "deepseek-flash", ProviderSettings())
    assert official.openai_api_base == DEEPSEEK_BASE_URL


def test_profile_endpoint_prefers_the_active_configuration(tmp_path):
    from dataclasses import replace

    from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings

    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    asyncio.run(store.initialize())
    vault = LocalSecretVault(store, tmp_path / "master.key")
    base = replace(ProviderSettings(), deepseek_model="fallback-model")
    profile = {
        "model": "profile-model",
        "base_url": "https://profile.example/v1",
        "api_format": "chat_completions",
    }

    async def scenario():
        from_profile = await apply_profile_endpoint(
            base, "web", profile, store=store, vault=vault
        )
        assert from_profile.deepseek_model == "profile-model"
        assert from_profile.mention_base_url == "https://profile.example/v1"

        config_id = await store.save_model_config(
            {
                "kind": "chat",
                "name": "x",
                "base_url": "https://stored.example/v1",
                "model": "stored-model",
                "api_format": "responses",
            }
        )
        await vault.set(f"model-config:{config_id}", "sk-stored")
        await store.set_active_model_config("web", config_id)
        from_stored = await apply_profile_endpoint(
            base, "web", profile, store=store, vault=vault
        )
        assert from_stored.deepseek_model == "stored-model"
        assert from_stored.mention_base_url == "https://stored.example/v1"
        assert from_stored.deepseek_api_format.value == "responses"
        assert from_stored.deepseek_api_key == "sk-stored"
        # Another scope keeps using its own profile.
        assert (
            await apply_profile_endpoint(base, "forum", profile)
        ).deepseek_model == ("profile-model")

    asyncio.run(scenario())

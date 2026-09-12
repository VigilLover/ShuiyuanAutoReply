import tomllib
from pathlib import Path

import pytest

from shuiyuan_auto_reply.bootstrap import deployment


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(deployment, "_current", None)
    monkeypatch.setattr(deployment, "load_dotenv", lambda: None)
    monkeypatch.setattr(deployment.os, "environ", {})


def test_precedence_and_relative_paths(tmp_path):
    deployment.os.environ["EMBEDDING_DIMS"] = "256"
    config = tmp_path / "config.toml"
    config.write_text(
        '[common.embedding]\ndims=512\n[profiles.local.embedding]\ndims=768\n[common.forum]\ncookie_file="secret.json"\n'
    )
    result = deployment.load_deployment(
        str(config), overrides={"embedding": {"dims": 1024}}
    )
    assert result.section("embedding")["dims"] == 1024
    assert result.section("forum")["cookie_file"] == str(tmp_path / "secret.json")


def test_secret_redaction(tmp_path):
    (tmp_path / "key").write_text("sensitive")
    config = tmp_path / "config.toml"
    config.write_text('[common.embedding]\napi_key={file="key"}\n')
    result = deployment.load_deployment(str(config))
    assert result.section("embedding")["api_key"] == "sensitive"
    assert "sensitive" not in str(result.redacted())


def test_remote_requires_file():
    with pytest.raises(ValueError):
        deployment.load_deployment(profile="remote")


def test_default_and_example_media_limits_match():
    example = Path(__file__).resolve().parents[1] / "config/deployment.example.toml"
    with example.open("rb") as source:
        media = tomllib.load(source)["common"]["media"]
    defaults = deployment.load_deployment().section("media")
    assert defaults["max_pixels"] == media["max_pixels"] == 32000000
    assert defaults["max_long_edge"] == media["max_long_edge"] == 2048
    assert defaults["max_image_bytes"] == media["max_image_bytes"] == 20971520

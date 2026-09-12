"""Process-wide deployment configuration, loaded before application construction."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULTS = {
    "forum": {"cookie_file": "cookies", "bot_username": "wolf_lumine"},
    "embedding": {
        "backend": "local",
        "model": "moka-ai/m3e-base",
        "dims": 768,
        "base_url": "",
        "api_key": "",
        "cache_folder": "",
        "batch_size": 10,
        "concurrency": 2,
        "timeout": 30,
        "attempts": 3,
        "max_batch_size": 20,
        "preprocessing": "raw-v1",
    },
    "retrieval": {"backend": "neo4j"},
    "database": {
        "url": "",
        "record_url": "",
        "memory_url": "",
        "migration_url": "",
        "auto_migrate": True,
        "strict": False,
    },
    "neo4j": {"url": "", "auth": ""},
    "mcp": {"enabled": True, "url": ""},
    "runtime": {
        "concurrency": 3,
        "queue_limit": 100,
        "timeout": 900,
        "poll_interval": 5,
        "shutdown_timeout": 60,
        "history_limit": 100,
        "history_ttl": 1800,
    },
    "media": {
        "max_image_bytes": 20971520,
        "max_turn_bytes": 41943040,
        "max_images": 20,
        "max_pixels": 32000000,
        "max_long_edge": 2048,
        "quota_bytes": 3221225472,
        "retention_days": 30,
    },
    "paths": {"state_dir": ""},
    "web": {"host": "127.0.0.1", "port": 11451},
    "providers": {},
}
# Legacy variables are read only at this boundary. Provider settings retain their
# existing compatibility mapping while deployment providers override it explicitly.
ENV = {
    "forum.cookie_file": "SHUIYUAN_COOKIE_FILE",
    "forum.bot_username": "SHUIYUAN_BOT_USERNAME",
    "embedding.model": "EMBEDDING_MODEL_NAME",
    "embedding.dims": "EMBEDDING_DIMS",
    "embedding.cache_folder": "EMBEDDING_CACHE_FOLDER",
    "embedding.backend": "EMBEDDING_BACKEND",
    "embedding.base_url": "EMBEDDING_BASE_URL",
    "embedding.api_key": "EMBEDDING_API_KEY",
    "database.url": "POSTGRES_DB_URL",
    "database.record_url": "POSTGRES_RECORD_DB_URL",
    "database.memory_url": "POSTGRES_MEMORY_DB_URL",
    "database.strict": "POSTGRES_STRICT",
    "neo4j.url": "NEO4J_DB_URL",
    "neo4j.auth": "NEO4J_DB_AUTH",
    "paths.state_dir": "SHUIYUAN_STATE_DIR",
    "mcp.url": "MCP_SERVER_URL",
}
PATHS = {"forum.cookie_file", "embedding.cache_folder", "paths.state_dir"}


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and not ({"env", "file"} & value.keys()):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            _merge(target[key], value)
        else:
            target[key] = value


def _resolve(value, root, prefix=""):
    if isinstance(value, dict):
        if "env" in value or "file" in value:
            if len(value) != 1:
                raise ValueError(
                    f"{prefix}: secret reference must contain only env or file"
                )
            if "env" in value:
                result = os.environ.get(value["env"])
                if not result:
                    raise ValueError(
                        f"{prefix}: referenced environment variable is missing"
                    )
                return result
            path = Path(value["file"]).expanduser()
            return (path if path.is_absolute() else root / path).read_text().strip()
        return {
            k: _resolve(v, root, f"{prefix}.{k}".lstrip(".")) for k, v in value.items()
        }
    if prefix in PATHS and value:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (root / path).resolve())
    return value


@dataclass(frozen=True)
class DeploymentConfig:
    profile: str
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]

    @property
    def fingerprint(self) -> str:
        e = self.section("embedding")
        identity = {
            k: e[k] for k in ("backend", "base_url", "model", "dims", "preprocessing")
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def redacted(self):
        def clean(value, key=""):
            if any(part in key.lower() for part in ("key", "auth", "url")) and value:
                return "[REDACTED]"
            if isinstance(value, dict):
                return {k: clean(v, k) for k, v in value.items()}
            return value

        return {
            "profile": self.profile,
            "values": clean(self.values),
            "vector_space": self.fingerprint,
        }


_current: DeploymentConfig | None = None


def load_deployment(
    path: str | None = None, profile: str = "local", overrides=None
) -> DeploymentConfig:
    global _current
    load_dotenv()
    values = copy.deepcopy(DEFAULTS)
    for dotted, name in ENV.items():
        raw = os.environ.get(name)
        if raw is None:
            continue
        section, key = dotted.split(".")
        default = values[section][key]
        values[section][key] = (
            raw.lower() in {"true", "1", "yes"}
            if isinstance(default, bool)
            else int(raw) if isinstance(default, int) else raw
        )
    if path:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as source:
            document = tomllib.load(source)
        if profile not in document.get("profiles", {}) and profile != "local":
            raise ValueError(f"Unknown deployment profile: {profile}")
        # Resolve paths only for explicitly configured fields, not legacy relative paths.
        _merge(values, _resolve(document.get("common", {}), config_path.parent))
        _merge(
            values,
            _resolve(document.get("profiles", {}).get(profile, {}), config_path.parent),
        )
    elif profile != "local":
        raise ValueError("A configuration file is required for non-local profiles")
    _merge(values, overrides or {})
    e = values["embedding"]
    if e["backend"] not in {"local", "openai"} or values["retrieval"][
        "backend"
    ] not in {"neo4j", "pgvector", "disabled"}:
        raise ValueError("Unsupported embedding or retrieval backend")
    for section, keys in {
        "embedding": ("dims", "batch_size", "concurrency", "timeout", "attempts"),
        "runtime": (
            "concurrency",
            "queue_limit",
            "timeout",
            "poll_interval",
            "history_limit",
            "history_ttl",
        ),
        "media": (
            "max_image_bytes",
            "max_turn_bytes",
            "max_images",
            "max_pixels",
            "max_long_edge",
        ),
    }.items():
        if any(
            not isinstance(values[section][key], (int, float))
            or values[section][key] <= 0
            for key in keys
        ):
            raise ValueError(f"{section}: resource limits must be positive")
    if e["batch_size"] > e["max_batch_size"]:
        raise ValueError("embedding.batch_size exceeds model batch limit")
    if e["backend"] == "openai" and (not e["base_url"] or not e["api_key"]):
        raise ValueError("Remote embedding requires base_url and api_key")
    if profile == "remote" and values["database"]["auto_migrate"]:
        raise ValueError("Remote runtime must not perform schema migrations")
    _current = DeploymentConfig(profile, values)
    # Transitional bridge for legacy providers/state modules, all writes centralized here.
    for dotted, name in ENV.items():
        section, key = dotted.split(".")
        value = values[section][key]
        os.environ[name] = str(value)
    os.environ["MCP_SERVER_URL"] = (
        values["mcp"]["url"] if values["mcp"]["enabled"] else ""
    )
    for name, value in values["providers"].items():
        os.environ[name] = str(value)
    return _current


def get_deployment() -> DeploymentConfig:
    return _current or load_deployment()


def add_config_arguments(parser):
    parser.add_argument("--config", help="Deployment TOML file")
    parser.add_argument(
        "--profile", default="local", help="Deployment profile (default: local)"
    )

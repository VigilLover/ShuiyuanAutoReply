"""Offline complete backups. Stop writers before invoking either operation."""

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from shuiyuan_auto_reply.bootstrap.deployment import get_deployment


def postgres_environment():
    from shuiyuan_auto_reply.infrastructure.retrieval.postgres import database_url

    parsed = urlsplit(database_url().replace("postgresql+psycopg://", "postgresql://"))
    env = dict(os.environ)
    env.update(
        PGHOST=parsed.hostname or "",
        PGPORT=str(parsed.port or 5432),
        PGUSER=unquote(parsed.username or ""),
        PGPASSWORD=unquote(parsed.password or ""),
        PGDATABASE=parsed.path.lstrip("/"),
    )
    return env


def backup(destination):
    from shuiyuan_auto_reply.infrastructure.persistence.state import state_directory

    target = Path(destination).resolve()
    state = state_directory().resolve()
    if target == state or state in target.parents:
        raise ValueError("Backup must be outside the state directory")
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    subprocess.run(
        ["pg_dump", "-Fc", "-f", str(target / "postgres.dump")],
        env=postgres_environment(),
        check=True,
    )
    local = target / "state"
    local.mkdir(mode=0o700)
    for item in state.iterdir():
        if item.name.startswith("state.sqlite3"):
            continue
        if item.is_dir():
            shutil.copytree(item, local / item.name)
        else:
            shutil.copy2(item, local / item.name)
    if (state / "state.sqlite3").exists():
        with (
            sqlite3.connect(state / "state.sqlite3") as source,
            sqlite3.connect(local / "state.sqlite3") as dest,
        ):
            source.backup(dest)
    (target / "manifest.json").write_text(
        json.dumps(dict(vector_space=get_deployment().fingerprint, complete=True))
    )


def restore(source):
    from shuiyuan_auto_reply.infrastructure.persistence.state import state_directory

    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        not manifest.get("complete")
        or manifest["vector_space"] != get_deployment().fingerprint
    ):
        raise ValueError("Backup vector space does not match configuration")
    state = state_directory()
    if any(state.iterdir()):
        raise ValueError(
            "Restore requires an empty state directory and an empty target database"
        )
    subprocess.run(
        [
            "pg_restore",
            "--exit-on-error",
            "--no-owner",
            "--no-privileges",
            "--dbname",
            postgres_environment()["PGDATABASE"],
            str(source / "postgres.dump"),
        ],
        env=postgres_environment(),
        check=True,
    )
    shutil.copytree(source / "state", state, dirs_exist_ok=True)

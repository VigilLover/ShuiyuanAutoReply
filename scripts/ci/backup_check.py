"""Container-only stopped-writer restore rehearsal against an empty second database."""

import asyncio
import json
import os
from importlib.resources import files
from pathlib import Path

import psycopg

from shuiyuan_auto_reply.bootstrap.deployment import load_deployment
from shuiyuan_auto_reply.infrastructure.operations.backup import backup, restore
from shuiyuan_auto_reply.infrastructure.persistence.secrets import LocalSecretVault
from shuiyuan_auto_reply.infrastructure.persistence.state import SQLiteStateStore


def main():
    load_deployment("/validation/config.toml", "remote")
    state = Path("/var/lib/shuiyuan")
    asyncio.run(
        LocalSecretVault(SQLiteStateStore()).set("synthetic", "synthetic-value")
    )
    source_key = (state / "master.key").read_bytes()
    backup("/tmp/rehearsal")
    with psycopg.connect(
        os.environ["VALIDATION_ADMIN_URL"], autocommit=True
    ) as connection:
        connection.execute("CREATE DATABASE restore_check")
    with psycopg.connect(
        os.environ["VALIDATION_ADMIN_URL"].replace("/shuiyuan", "/restore_check"),
        autocommit=True,
    ) as connection:
        connection.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT,INSERT,UPDATE,DELETE ON TABLES TO shuiyuan_app"
        )
        connection.execute(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE,SELECT ON SEQUENCES TO shuiyuan_app"
        )
    config = (
        Path("/validation/config.toml")
        .read_text()
        .replace("/var/lib/shuiyuan", "/tmp/restored-state")
    )
    Path("/tmp/restore.toml").write_text(config)
    os.environ["VALIDATION_DB_URL"] = os.environ["VALIDATION_ADMIN_URL"].replace(
        "/shuiyuan", "/restore_check"
    )
    load_deployment("/tmp/restore.toml", "remote")
    restore("/tmp/rehearsal")
    assert Path("/tmp/restored-state/master.key").read_bytes() == source_key
    assert (
        asyncio.run(LocalSecretVault(SQLiteStateStore()).get("synthetic"))
        == "synthetic-value"
    )
    with psycopg.connect(
        os.environ["VALIDATION_DB_URL"].replace(
            "postgres:synthetic-only", "shuiyuan_app:synthetic-app"
        )
    ) as connection:
        assert (
            connection.execute("SELECT count(*) FROM style_sentences").fetchone()[0]
            == 2
        )
    root = files("shuiyuan_auto_reply")
    for name in [
        "assets/pet_responses.json",
        "assets/tarot_img/1.jpg",
        "assets/fonts/Noto_Sans_SC/static/NotoSansSC-Regular.ttf",
        "interfaces/api/static/index.html",
    ]:
        assert root.joinpath(name).is_file(), name
    print("Backup restore and package resources passed")


if __name__ == "__main__":
    main()

"""Deployment commands. No command publishes forum posts."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from shuiyuan_auto_reply.bootstrap.deployment import (
    add_config_arguments,
    load_deployment,
)


def parser():
    root = argparse.ArgumentParser(description="Shuiyuan deployment operations")
    add_config_arguments(root)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("config").add_argument("action", choices=["check"])
    doctor = commands.add_parser("doctor")
    doctor.add_argument(
        "--probe-embedding", action="store_true", help="Explicit paid embedding request"
    )
    commands.add_parser("db").add_argument("action", choices=["migrate"])
    for kind in ("corpus", "memory"):
        data = commands.add_parser(kind)
        data.add_argument("action", choices=["export", "import"])
        data.add_argument("path")
        data.add_argument("--csv")
        data.add_argument("--persona")
        data.add_argument("--dry-run", action="store_true")
    cookie = commands.add_parser("cookie")
    cookie.add_argument("action", choices=["convert"])
    cookie.add_argument("source")
    cookie.add_argument("destination")
    cookie.add_argument(
        "--trust-pickle",
        action="store_true",
        required=True,
        help="Explicitly allow code execution when loading your own trusted pickle",
    )
    for command in ("backup", "restore"):
        item = commands.add_parser(command)
        item.add_argument("path")
        item.add_argument("--writers-stopped", action="store_true", required=True)
    return root


async def run(args, config):
    from shuiyuan_auto_reply.infrastructure.operations import migration

    if args.command == "config":
        print(json.dumps(config.redacted(), indent=2))
    elif args.command == "db":
        await migration.migrate_database()
    elif args.command == "doctor":
        from sqlalchemy import text

        from shuiyuan_auto_reply.infrastructure.embedding import get_embeddings
        from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
            check_vector_space,
            engine_for,
        )

        cookie = Path(config.section("forum")["cookie_file"])
        if not cookie.is_file():
            raise ValueError("Cookie file missing")
        engine = engine_for()
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
                if config.profile == "remote":
                    await check_vector_space(connection)
        finally:
            await engine.dispose()
        if args.probe_embedding:
            await get_embeddings().aembed_query("连接测试")
        print("Configuration, cookie path and database OK; forum identity not verified")
    elif args.command in ("corpus", "memory"):
        if args.action == "import":
            count = await migration.import_data(
                args.command, args.path, dry_run=args.dry_run
            )
        elif args.command == "corpus":
            count = await migration.export_corpus(
                args.path, csv_path=args.csv, persona=args.persona
            )
        else:
            count = await migration.export_memory(args.path)
        print(json.dumps({"records": count}))
    elif args.command == "cookie":
        import pickle

        with open(args.source, "rb") as source:
            cookies = pickle.load(source)
        values = {
            str(k): str(v.value if hasattr(v, "value") else v)
            for k, v in cookies.items()
        }
        fd = os.open(args.destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as target:
            json.dump(
                {"version": 1, "domain": "shuiyuan.sjtu.edu.cn", "cookies": values},
                target,
            )
    else:
        from shuiyuan_auto_reply.infrastructure.operations.backup import backup, restore

        (backup if args.command == "backup" else restore)(args.path)


def main():
    args = parser().parse_args()
    config = load_deployment(args.config, args.profile)

    async def execute():
        try:
            await run(args, config)
        finally:
            from shuiyuan_auto_reply.infrastructure.embedding import close_embeddings

            await close_embeddings()

    asyncio.run(execute())


if __name__ == "__main__":
    main()

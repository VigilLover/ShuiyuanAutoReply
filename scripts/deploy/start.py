#!/usr/bin/env python3
"""Ordered startup: database healthy, migration completed, then application services."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy/compose.yaml")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pull", action="store_true", help="Pull prebuilt configured image tags"
    )
    parser.add_argument("--no-mcp", action="store_true")
    args = parser.parse_args()
    if args.pull:
        subprocess.run(
            COMPOSE + ["pull", "postgres", "bot"] + ([] if args.no_mcp else ["mcp"]),
            check=True,
        )
    subprocess.run(COMPOSE + ["up", "-d", "--wait", "postgres"], check=True)
    subprocess.run(COMPOSE + ["stop", "bot"], check=True)
    subprocess.run(COMPOSE + ["run", "--rm", "migrate"], check=True)
    subprocess.run(
        COMPOSE + ["up", "-d", "--wait", "bot"] + ([] if args.no_mcp else ["mcp"]),
        check=True,
    )
    print("Started. Use /api/runtime-health to inspect forum and database readiness.")


if __name__ == "__main__":
    main()

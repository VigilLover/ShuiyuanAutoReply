#!/usr/bin/env python3
"""Fetch exact build inputs; never update sibling checkouts or accept moving HEAD."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "simplemcp": (
        "https://github.com/VigilLover/SimpleMCP-for-ShuiyuanAutoReply.git",
        "de42d48eb81644604ebade8524af2748c4cc3e6b",
    ),
    "pgvector": (
        "https://github.com/pgvector/pgvector.git",
        "cab9da72c04353f143bb06b42ab70a403daac64a",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcp-source",
        help="Optional local repository to fetch the exact MCP commit from",
    )
    args = parser.parse_args()
    for name, (url, revision) in SOURCES.items():
        target = ROOT / "deploy" / "vendor" / name
        if not target.exists():
            target.mkdir(parents=True)
            subprocess.run(["git", "init", str(target)], check=True)
        current = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if current.stdout.strip() != revision:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(target),
                    "fetch",
                    "--depth=1",
                    args.mcp_source if name == "simplemcp" and args.mcp_source else url,
                    revision,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(target), "status", "--porcelain"], text=True
        ).strip()
        if actual != revision or dirty:
            raise SystemExit(f"{name}: revision mismatch or modified checkout")
        print(f"{name}: {actual}")


if __name__ == "__main__":
    main()

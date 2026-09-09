"""Fail the release before it builds when the compatibility window is unusable."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/deploy"))
from release import compatibility_sources, default_repository


def official(version):
    row = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{default_repository()}/releases/tags/{version}"],
            text=True,
        )
    )
    return row["tag_name"] == version and not row["draft"] and not row["prerelease"]


def main():
    policy = json.loads((ROOT / "deploy/release-policy.json").read_text())
    if policy["migration"] != "backward-compatible":
        print(f"migration={policy['migration']}: no compatibility window to resolve")
        return
    sources = compatibility_sources(policy)
    if not sources:
        raise ValueError("Backward compatibility requires at least one source release")
    for version in sources:
        if not official(version):
            raise ValueError(f"{version} is not an official release")
    print("compatibility window: " + ", ".join(sources))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Compatibility policy rejected: {error}", file=sys.stderr)
        raise SystemExit(1)

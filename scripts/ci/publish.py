"""Publish already-tested images; create the public release only after all uploads."""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from release import VERSION, bundle


def main():
    version = os.environ["VERSION"]
    if not VERSION.fullmatch(version):
        raise ValueError("Invalid version")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    images = {}
    for name in ("bot", "postgres", "mcp"):
        repository = f"ghcr.io/vigillover/shuiyuan-{name}"
        image = f"{repository}:{version}"
        sha_tag = f"{repository}:sha-{sha}"
        subprocess.run(["docker", "tag", image, sha_tag], check=True)
        for tag in (image, sha_tag):
            subprocess.run(["docker", "push", tag], check=True)
        digests = json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "image",
                    "inspect",
                    image,
                    "--format",
                    "{{json .RepoDigests}}",
                ],
                text=True,
            )
        )
        images[name] = next(x for x in digests if x.startswith(repository + "@"))
    bundle(version, images, "dist-release")
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            version,
            "--verify-tag",
            "--draft",
            "--title",
            version,
            "--notes",
            f"Verified linux/amd64 release from dev commit {sha}.",
        ],
        check=True,
    )
    subprocess.run(
        ["gh", "release", "upload", version, *map(str, Path("dist-release").iterdir())],
        check=True,
    )
    subprocess.run(["gh", "release", "edit", version, "--draft=false"], check=True)


if __name__ == "__main__":
    main()

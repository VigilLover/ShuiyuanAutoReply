"""Publish already-tested images; create the public release only after all uploads."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from release import VERSION, bundle, image_inputs


def version_key(tag):
    return tuple(int(part) for part in VERSION.fullmatch(tag).groups())


def previous_manifest(version):
    """Manifest of the newest published release below `version`, else {}."""
    try:
        releases = json.loads(
            subprocess.check_output(
                ["gh", "api", "repos/{owner}/{repo}/releases?per_page=30"], text=True
            )
        )
    except subprocess.CalledProcessError as error:
        print(f"Skipping image reuse: {error}", file=sys.stderr)
        return {}
    older = [
        release["tag_name"]
        for release in releases
        if not release["draft"]
        and not release["prerelease"]
        and VERSION.fullmatch(release["tag_name"])
        and version_key(release["tag_name"]) < version_key(version)
    ]
    if not older:
        return {}
    tag = max(older, key=version_key)
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(
            [
                "gh",
                "release",
                "download",
                tag,
                "--pattern",
                "release.json",
                "--dir",
                directory,
            ],
            capture_output=True,
            text=True,
        )
        manifest = Path(directory) / "release.json"
        if result.returncode or not manifest.is_file():
            print(f"Skipping image reuse: {result.stderr.strip()}", file=sys.stderr)
            return {}
        return json.loads(manifest.read_text())


def publish(name, version, sha):
    """Tag and push a freshly built image, returning its published digest."""
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
    return next(x for x in digests if x.startswith(repository + "@"))


def main():
    version = os.environ["VERSION"]
    if not VERSION.fullmatch(version):
        raise ValueError("Invalid version")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    inputs = image_inputs()
    previous = previous_manifest(version)
    images = {}
    for name in ("bot", "postgres", "mcp"):
        published = previous.get("images", {}).get(name)
        if published and previous.get("image_inputs", {}).get(name) == inputs[name]:
            # Build inputs unchanged: reuse the published image so a deploy does
            # not download and recreate an identical image under a new digest.
            images[name] = published
            continue
        images[name] = publish(name, version, sha)
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
            f"Verified linux/amd64 release from main commit {sha}.",
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

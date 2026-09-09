"""Deterministic, allowlisted release bundles. No credentials or runtime state."""

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

VERSION = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
ROOT = Path(__file__).resolve().parents[2]
IMAGES = {"bot", "postgres", "mcp"}
# Build inputs each image is built from, per its Dockerfile COPY lines. Base image
# digests are pinned inside the Dockerfile, so hashing the file covers them.
IMAGE_INPUTS = {
    "bot": (
        "deploy/Dockerfile",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "src",
        "web",
    ),
    "postgres": ("deploy/postgres", "deploy/vendor/pgvector"),
    "mcp": ("deploy/mcp", "deploy/vendor/simplemcp"),
}
IGNORED = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    ".pytest_cache",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
}
# `recent:N` keeps the compatibility window at the N most recent official
# releases, so the policy file does not have to be edited on every release.
RECENT = re.compile(r"recent:([1-9]\d*)\Z")


def _version_key(tag):
    return tuple(int(part) for part in tag[1:].split("."))


def default_repository():
    """OWNER/REPO of the origin remote, so gh never has to guess (it picks
    upstream when a fork has one)."""
    url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    match = re.fullmatch(
        r"(?:https://|git@)github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?", url
    )
    if match is None:
        raise ValueError("Cannot resolve the GitHub repository from origin")
    return f"{match.group(1)}/{match.group(2)}"


def recent_releases(limit):
    """The newest official releases, most recent first."""
    rows = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{default_repository()}/releases?per_page=100"],
            text=True,
        )
    )
    official = [
        row["tag_name"]
        for row in rows
        if not row["draft"]
        and not row["prerelease"]
        and VERSION.fullmatch(row["tag_name"])
    ]
    official.sort(key=_version_key, reverse=True)
    if len(official) < limit:
        raise ValueError("Not enough official releases for the compatibility window")
    return official[:limit]


def compatibility_sources(policy):
    """Expand compatible_from into the concrete versions a release is tested against."""
    sources = policy.get("compatible_from", [])
    if isinstance(sources, str):
        window = RECENT.fullmatch(sources)
        if window is None:
            raise ValueError("Invalid compatibility policy")
        sources = recent_releases(int(window.group(1)))
    sources = list(sources)
    for version in sources:
        if not VERSION.fullmatch(version):
            raise ValueError("Invalid compatibility version")
    return sources


def validate(manifest):
    if not VERSION.fullmatch(manifest["version"]):
        raise ValueError("Invalid release version")
    if manifest["format"] != 1 or manifest["config_version"] != 1:
        raise ValueError("Unsupported release/config format")
    if manifest["architecture"] != "linux/amd64":
        raise ValueError("Unsupported architecture")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]):
        raise ValueError("Invalid commit")
    if manifest["migration"] not in {"none", "backward-compatible", "manual"}:
        raise ValueError("Invalid migration policy")
    if set(manifest["images"]) != IMAGES:
        raise ValueError("Missing image")
    for name, value in manifest["images"].items():
        if not re.fullmatch(
            rf"ghcr\.io/vigillover/shuiyuan-{name}@sha256:[0-9a-f]{{64}}", value
        ):
            raise ValueError("Invalid image repository/digest")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["schema_id"]):
        raise ValueError("Invalid schema identity")
    # Optional so manifests published before input identities remain readable.
    inputs = manifest.get("image_inputs")
    if inputs is not None and (
        set(inputs) != IMAGES
        or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in inputs.values())
    ):
        raise ValueError("Invalid image input identity")
    return manifest


def schema_id():
    paths = [ROOT / "uv.lock", ROOT / "pyproject.toml"]
    for relative in (
        "src/shuiyuan_auto_reply/database",
        "src/shuiyuan_auto_reply/infrastructure/persistence",
        "src/shuiyuan_auto_reply/infrastructure/operations",
        "src/shuiyuan_auto_reply/infrastructure/retrieval",
    ):
        paths.extend((ROOT / relative).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def image_inputs(root=ROOT):
    """Content identity of every image's build inputs, independent of build time.

    Rebuilding an image yields a new digest even when nothing changed, so release
    guards compare this identity instead of the published digest.
    """
    identities = {}
    for name, entries in IMAGE_INPUTS.items():
        candidates = set()
        for entry in entries:
            path = root / entry
            if path.is_dir():
                candidates.update(item for item in path.rglob("*") if item.is_file())
            elif path.is_file():
                candidates.add(path)
        digest = hashlib.sha256()
        for path in sorted(
            (
                item
                for item in candidates
                if not IGNORED & set(item.relative_to(root).parts)
            ),
            key=lambda item: str(item.relative_to(root)),
        ):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        identities[name] = digest.hexdigest()
    return identities


def bundle(version, images, output):
    policy = json.loads((ROOT / "deploy/release-policy.json").read_text())
    sources = (
        compatibility_sources(policy)
        if policy["migration"] == "backward-compatible"
        else []
    )
    manifest = validate(
        dict(
            format=1,
            config_version=1,
            version=version,
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            architecture="linux/amd64",
            images=images,
            image_inputs=image_inputs(),
            migration=policy["migration"],
            compatible_from=sources,
            schema_id=schema_id(),
        )
    )
    evidence = json.loads((ROOT / "reports/integration.json").read_text())
    if evidence.get("passed") is not True:
        raise ValueError("Integration evidence missing")
    manifest["compatibility"] = evidence.get("compatibility", {})
    if policy["migration"] == "backward-compatible" and set(
        manifest["compatibility"]
    ) != set(sources):
        raise ValueError("Compatibility evidence missing")
    files = {
        "deploy/compose.yaml",
        "config/deployment.example.toml",
        "docs/deployment.md",
        "docs/cicd.md",
        "docs/first-deployment.md",
        "docs/branch-strategy.md",
    }
    files.update(
        str(p.relative_to(ROOT)) for p in (ROOT / "scripts/deploy").glob("*.py")
    )
    payload = {name: (ROOT / name).read_bytes() for name in sorted(files)}
    payload["test-summary.json"] = (ROOT / "reports/integration.json").read_bytes()
    manifest["files"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in payload.items()
    }
    payload["release.json"] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"shuiyuan-{version}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "SHA256SUMS").write_text(f"{checksum}  {path.name}\n")
    (output / "release.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist-release"))
    args = parser.parse_args()
    bundle(args.version, json.loads(args.images.read_text()), args.output)

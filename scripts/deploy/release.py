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


def bundle(version, images, output):
    policy = json.loads((ROOT / "deploy/release-policy.json").read_text())
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
            migration=policy["migration"],
            compatible_from=policy.get("compatible_from", []),
            schema_id=schema_id(),
        )
    )
    evidence = json.loads((ROOT / "reports/integration.json").read_text())
    if evidence.get("passed") is not True:
        raise ValueError("Integration evidence missing")
    manifest["compatibility"] = evidence.get("compatibility", {})
    if policy["migration"] == "backward-compatible" and set(
        manifest["compatibility"]
    ) != set(policy["compatible_from"]):
        raise ValueError("Compatibility evidence missing")
    files = {
        "deploy/compose.yaml",
        "config/deployment.example.toml",
        "docs/deployment.md",
        "docs/cicd.md",
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

"""Trusted server-side release controller. Install outside uploaded release bundles."""

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from release import VERSION, validate

ROOT = Path("/opt/shuiyuan")
MAX_BUNDLE = 20 * 1024 * 1024


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


@contextlib.contextmanager
def locked(root):
    with (root / "shared/deploy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_release(root, version):
    if not VERSION.fullmatch(version):
        raise ValueError("Invalid release version")
    directory = root / "releases" / version
    manifest = validate(json.loads((directory / "release.json").read_text()))
    if manifest["version"] != version:
        raise ValueError("Release identity mismatch")
    for name, checksum in manifest["files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or directory.resolve() not in (directory / path).resolve().parents
        ):
            raise ValueError("Invalid bundle path")
        if hashlib.sha256((directory / path).read_bytes()).hexdigest() != checksum:
            raise ValueError("Release file integrity failure")
    return manifest


def verify_published(root, version, checksum):
    """Check GitHub's own asset digest, independently of the SSH caller."""
    if not VERSION.fullmatch(version) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid publication identity")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "shuiyuan-release-controller",
    }
    token = root / "shared/github_read_token"
    if token.exists():
        headers["Authorization"] = "Bearer " + token.read_text().strip()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, response_headers, newurl):
            return None

    request = urllib.request.Request(
        "https://api.github.com/repos/VigilLover/ShuiyuanAutoReply/releases/tags/"
        + version,
        headers=headers,
    )
    with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
        body = response.read(1024 * 1024 + 1)
    if len(body) > 1024 * 1024:
        raise ValueError("Release metadata too large")
    metadata = json.loads(body)
    if (
        metadata.get("draft") is not False
        or metadata.get("prerelease") is not False
        or metadata.get("tag_name") != version
    ):
        raise ValueError("Only official published releases are allowed")
    assets = [
        asset
        for asset in metadata.get("assets", [])
        if asset.get("name") == f"shuiyuan-{version}.tar.gz"
    ]
    if (
        len(assets) != 1
        or assets[0].get("digest") != "sha256:" + checksum
        or assets[0].get("state") != "uploaded"
    ):
        raise ValueError("Bundle does not match GitHub's published asset digest")


def receive(root, version, checksum, stream):
    if not VERSION.fullmatch(version) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid receive arguments")
    data = stream.read(MAX_BUNDLE + 1)
    if len(data) > MAX_BUNDLE or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("Bundle size/checksum mismatch")
    with locked(root), tempfile.TemporaryDirectory(dir=root / "releases") as temporary:
        target = Path(temporary)
        total = 0
        names = set()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for entry in archive:
                path = Path(entry.name)
                total += entry.size
                if (
                    not entry.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or len(names) >= 128
                    or len(entry.name) > 200
                    or entry.name in names
                    or total > MAX_BUNDLE
                ):
                    raise ValueError("Unsafe archive member")
                names.add(entry.name)
                output = target / path
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.extractfile(entry).read())
                output.chmod(0o644)
        manifest = validate(json.loads((target / "release.json").read_text()))
        if manifest["version"] != version or names != set(manifest["files"]) | {
            "release.json"
        }:
            raise ValueError("Unexpected bundle contents")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((target / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Bundle integrity failure")
        destination = root / "releases" / version
        if destination.exists():
            if (destination / "release.json").read_bytes() != (
                target / "release.json"
            ).read_bytes():
                raise ValueError("Published versions are immutable")
            read_release(root, version)
        else:
            shutil.copytree(target, destination)
    print(json.dumps({"received": version}))


def transition(previous, target, rollback=False):
    previous_inputs = previous.get("image_inputs", {})
    target_inputs = target.get("image_inputs", {})
    if "postgres" in previous_inputs and "postgres" in target_inputs:
        # Compare build inputs, not digests: rebuilding an unchanged image yields
        # a new digest and must not be treated as a database image change.
        if previous_inputs["postgres"] != target_inputs["postgres"]:
            raise ValueError("Database image changes require maintenance")
    else:
        logging.warning(
            "Release manifests lack image input identity; skipping the database "
            "image guard for %s -> %s",
            previous.get("version"),
            target.get("version"),
        )
    newer = previous if rollback else target
    older = target if rollback else previous
    if newer["migration"] == "manual" or target["migration"] == "manual":
        raise ValueError("Manual migration requires maintenance")
    if newer["schema_id"] != older["schema_id"]:
        evidence = newer.get("compatibility", {})
        if (
            newer["migration"] != "backward-compatible"
            or evidence.get(older["version"]) != older["images"]["bot"]
        ):
            raise ValueError("No tested backward compatibility for this release")


def ready(health, polled_after=0):
    return (
        health.get("process") == "ok"
        and health.get("database") == "ok"
        and health.get("state") == "ok"
        and health.get("forum") == "ok"
        and (
            not polled_after
            or (
                isinstance(health.get("last_poll"), (int, float))
                and health["last_poll"] >= polled_after
            )
        )
    )


class Deployer:
    def __init__(self, root=ROOT):
        self.root = root
        self.record = {}

    def run(self, command, *, env=None):
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=600
        )
        if result.returncode:
            # Container output can contain cookies/provider errors. Never propagate raw logs.
            raise RuntimeError(
                f"Command failed with exit {result.returncode}: {command[0]}"
            )
        return result.stdout

    def compose(self, manifest, arguments):
        env = dict(os.environ)
        env.update(
            SHUIYUAN_CONFIG=str(self.root / "shared/deployment.toml"),
            SHUIYUAN_SECRETS=str(self.root / "shared/secrets"),
            BOT_IMAGE=manifest["images"]["bot"],
            POSTGRES_IMAGE=manifest["images"]["postgres"],
            MCP_IMAGE=manifest["images"]["mcp"],
        )
        return self.run(
            [
                "docker",
                "compose",
                "--project-name",
                "shuiyuan",
                "-f",
                str(
                    self.root / "releases" / manifest["version"] / "deploy/compose.yaml"
                ),
                *arguments,
            ],
            env=env,
        )

    def ops(self, manifest, args, *, extra=(), service="bot"):
        try:
            return self.compose(
                manifest,
                [
                    "run",
                    "--rm",
                    "--no-deps",
                    "--name",
                    "shuiyuan-release-ops",
                    *extra,
                    service,
                    "shuiyuan-ops",
                    "--config",
                    "/etc/shuiyuan/deployment.toml",
                    "--profile",
                    "remote",
                    *args,
                ],
            )
        finally:
            # A timed-out Docker CLI must not leave a migration racing the old app.
            subprocess.run(
                ["docker", "rm", "-f", "shuiyuan-release-ops"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )

    def fingerprint(self, manifest):
        config = json.loads(self.ops(manifest, ["config", "check"]))
        return config["vector_space"]

    def save(self, phase, **fields):
        self.record.update(phase=phase, **fields)
        atomic_json(self.root / "shared/last-deployment.json", self.record)

    def health(self):
        with urllib.request.urlopen(
            "http://127.0.0.1:11451/api/runtime-health", timeout=5
        ) as response:
            return json.load(response)

    def wait_ready(self):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if ready(self.health(), getattr(self, "poll_after", 0)):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(3)
        raise RuntimeError("Business readiness timed out")

    def start(self, manifest, mcp):
        self.poll_after = time.time()
        self.compose(
            manifest,
            ["up", "-d", "--no-deps", "--wait", "bot", *(["mcp"] if mcp else [])],
        )
        if not mcp:
            self.compose(manifest, ["stop", "mcp"])

    def deploy(self, version, *, rollback=False):
        with locked(self.root):
            target = read_release(self.root, version)
            baseline = json.loads((self.root / "shared/current.json").read_text())
            previous = read_release(self.root, baseline["version"])
            transition(previous, target, rollback)
            if platform.machine() not in {"x86_64", "AMD64"}:
                raise ValueError("This release requires amd64")
            if shutil.disk_usage(self.root).free < 8 * 1024**3:
                raise ValueError("Less than 8 GiB free disk space")
            self.record = dict(
                version=version,
                previous=previous["version"],
                rollback=rollback,
                started_at=int(time.time()),
            )
            stopped = False
            try:
                self.save("pull")
                self.compose(target, ["pull", "bot", "postgres", "mcp"])
                fingerprint = self.fingerprint(target)
                if fingerprint != baseline["fingerprint"]:
                    raise ValueError("Vector space changes require maintenance")
                # Require an existing, healthy installation before ordinary releases.
                if not ready(self.health()):
                    raise ValueError("Existing installation is not business-ready")
                import tomllib

                config = tomllib.loads(
                    (self.root / "shared/deployment.toml").read_text()
                )
                common = config.get("common", {}).get("mcp", {})
                mcp = (
                    config.get("profiles", {})
                    .get("remote", {})
                    .get("mcp", {})
                    .get("enabled", common.get("enabled", True))
                )
                self.save("stop")
                stopped = True
                self.compose(previous, ["stop", "bot"])
                backup = (
                    self.root / "backups" / f"{int(time.time())}-{previous['version']}"
                )
                self.save("backup", backup=str(backup))
                self.ops(
                    previous,
                    ["backup", f"/backups/{backup.name}", "--writers-stopped"],
                    extra=["-v", f"{self.root / 'backups'}:/backups"],
                )
                if not rollback and target["migration"] != "none":
                    self.save("migrate")
                    self.ops(target, ["db", "migrate"], service="migrate")
                self.save("start")
                self.start(target, mcp)
                self.save("health")
                self.wait_ready()
                atomic_json(
                    self.root / "shared/current.json",
                    dict(version=version, fingerprint=fingerprint),
                )
                temporary = self.root / "current.next"
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(self.root / "releases" / version)
                temporary.replace(self.root / "current")
                self.save("success")
            except Exception as error:
                failed_phase = self.record.get("phase")
                if stopped:
                    try:
                        self.compose(target, ["stop", "bot"])
                        self.start(previous, mcp)
                        self.wait_ready()
                        self.save(
                            "failed-restored-application",
                            failed_phase=failed_phase,
                            error=type(error).__name__,
                        )
                    except Exception:
                        self.save(
                            "failed-needs-maintenance",
                            failed_phase=failed_phase,
                            error=type(error).__name__,
                        )
                else:
                    self.save(
                        "failed-preflight",
                        failed_phase=failed_phase,
                        error=type(error).__name__,
                    )
                raise
            print(json.dumps(self.record))

    def adopt(self, version):
        """Local administrator only: register an already initialized first release."""
        with locked(self.root):
            if (self.root / "shared/current.json").exists():
                raise ValueError("Already adopted")
            manifest = read_release(self.root, version)
            if not ready(self.health()):
                raise ValueError("Initial installation must be ready before adoption")
            fingerprint = self.fingerprint(manifest)
            atomic_json(
                self.root / "shared/current.json",
                dict(version=version, fingerprint=fingerprint),
            )
            (self.root / "current").symlink_to(self.root / "releases" / version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["deploy", "rollback", "status", "adopt"])
    parser.add_argument("--release")
    args = parser.parse_args()
    controller = Deployer()
    if args.operation == "status":
        print(
            (ROOT / "shared/last-deployment.json").read_text()
            if (ROOT / "shared/last-deployment.json").exists()
            else "{}"
        )
    elif args.operation == "adopt":
        controller.adopt(args.release)
    else:
        controller.deploy(args.release, rollback=args.operation == "rollback")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Release operation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

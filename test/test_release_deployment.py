import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/deploy"
sys.path.insert(0, str(DIRECTORY))
import release
import remote


def identities(postgres="a" * 64):
    return {
        name: postgres if name == "postgres" else "e" * 64
        for name in ("bot", "postgres", "mcp")
    }


def manifest(version="v1.0.0", schema="a", migration="none", inputs=None):
    description = dict(
        format=1,
        config_version=1,
        version=version,
        commit="a" * 40,
        architecture="linux/amd64",
        migration=migration,
        schema_id=schema * 64,
        images={
            name: f"ghcr.io/vigillover/shuiyuan-{name}@sha256:" + "b" * 64
            for name in ("bot", "postgres", "mcp")
        },
        files={},
    )
    if inputs is not None:
        description["image_inputs"] = inputs
    return description


def test_manifest_and_transition_guards():
    old, new = manifest(), manifest("v1.0.1")
    release.validate(new)
    remote.transition(old, new)
    new["schema_id"] = "c" * 64
    with pytest.raises(ValueError, match="compatibility"):
        remote.transition(old, new)
    new["migration"] = "backward-compatible"
    new["compatibility"] = {old["version"]: old["images"]["bot"]}
    remote.transition(old, new)
    remote.transition(new, old, rollback=True)
    for invalid in ("latest", "v1.2.3;id", "../v1.2.3", "v01.2.3"):
        with pytest.raises(ValueError):
            release.validate(manifest(invalid))
    release.validate(manifest(inputs=identities()))


def test_compatibility_window_expands_recent_releases():
    with patch("release.recent_releases", return_value=["v1.0.5", "v1.0.4"]):
        assert release.compatibility_sources({"compatible_from": "recent:2"}) == [
            "v1.0.5",
            "v1.0.4",
        ]
    assert release.compatibility_sources({"compatible_from": ["v1.0.3"]}) == ["v1.0.3"]
    for invalid in ("recent:0", "recent:", "latest", ["latest"], "../v1.0.3"):
        with pytest.raises(ValueError):
            release.compatibility_sources({"compatible_from": invalid})


def test_recent_releases_keeps_only_official_versions_newest_first():
    rows = [
        {"tag_name": "v1.0.5", "draft": False, "prerelease": False},
        {"tag_name": "v1.0.6", "draft": True, "prerelease": False},
        {"tag_name": "v1.0.4", "draft": False, "prerelease": True},
        {"tag_name": "v1.0.3", "draft": False, "prerelease": False},
    ]
    with (
        patch("release.default_repository", return_value="o/r"),
        patch("subprocess.check_output", return_value=json.dumps(rows)),
    ):
        assert release.recent_releases(2) == ["v1.0.5", "v1.0.3"]
        with pytest.raises(ValueError):
            release.recent_releases(3)


def test_default_repository_reads_origin_without_asking_gh():
    for url, expected in (
        (
            "https://github.com/VigilLover/ShuiyuanAutoReply.git",
            "VigilLover/ShuiyuanAutoReply",
        ),
        (
            "git@github.com:VigilLover/ShuiyuanAutoReply.git",
            "VigilLover/ShuiyuanAutoReply",
        ),
        (
            "https://github.com/VigilLover/ShuiyuanAutoReply",
            "VigilLover/ShuiyuanAutoReply",
        ),
    ):
        with patch("subprocess.check_output", return_value=url):
            assert release.default_repository() == expected
    with patch("subprocess.check_output", return_value="https://gitlab.com/x/y.git"):
        with pytest.raises(ValueError):
            release.default_repository()


def test_database_guard_compares_build_inputs_not_digests():
    shared = identities()
    old, new = manifest(inputs=shared), manifest("v1.0.1", inputs=shared)
    # Rebuilding an unchanged image yields a new digest; that must not block.
    new["images"]["postgres"] = new["images"]["postgres"].replace("b" * 64, "d" * 64)
    remote.transition(old, new)
    new["image_inputs"] = identities(postgres="f" * 64)
    with pytest.raises(ValueError, match="Database"):
        remote.transition(old, new)


def test_database_guard_warns_and_allows_legacy_manifests(caplog):
    old, new = manifest(), manifest("v1.0.1")
    new["images"]["postgres"] = new["images"]["postgres"].replace("b" * 64, "d" * 64)
    remote.transition(old, new)
    assert "lack image input identity" in caplog.text


def test_image_inputs_track_content_and_ignore_noise(tmp_path):
    for directory in (
        "deploy/postgres",
        "deploy/vendor/pgvector/src",
        "deploy/vendor/pgvector/.git",
        "src",
        "web",
    ):
        (tmp_path / directory).mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        (tmp_path / name).write_text(name)
    (tmp_path / "deploy/Dockerfile").write_text("from scratch")
    (tmp_path / "deploy/postgres/Dockerfile").write_text("from scratch")
    (tmp_path / "deploy/postgres/init.sh").write_text("select 1")
    (tmp_path / "deploy/vendor/pgvector/src/vector.c").write_text("int main;")
    (tmp_path / "deploy/vendor/pgvector/.git/HEAD").write_text("ref")
    (tmp_path / "src/app.py").write_text("pass")
    (tmp_path / "web/index.html").write_text("<html>")
    original = release.image_inputs(tmp_path)
    assert release.image_inputs(tmp_path) == original
    (tmp_path / "deploy/vendor/pgvector/.git/HEAD").write_text("moved")
    assert release.image_inputs(tmp_path) == original
    (tmp_path / "deploy/postgres/init.sh").write_text("select 2")
    changed = release.image_inputs(tmp_path)
    assert changed["postgres"] != original["postgres"]
    assert changed["bot"] == original["bot"]


def test_receive_rejects_traversal_and_wrong_checksum(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "releases").mkdir()
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../escape")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    raw = data.getvalue()
    with pytest.raises(ValueError, match="Unsafe"):
        remote.receive(
            tmp_path, "v1.0.0", hashlib.sha256(raw).hexdigest(), io.BytesIO(raw)
        )
    with pytest.raises(ValueError, match="checksum"):
        remote.receive(tmp_path, "v1.0.0", "0" * 64, io.BytesIO(raw))
    assert not (tmp_path / "escape").exists()


class FakeDeployer(remote.Deployer):
    def __init__(self, root, failure=None):
        super().__init__(root)
        self.failure = failure
        self.events = []

    def compose(self, manifest, args):
        self.events.append((manifest["version"], args[0]))
        if args[0] == self.failure:
            raise RuntimeError("synthetic")
        return ""

    def fingerprint(self, manifest):
        return "changed" if self.failure == "fingerprint" else "space"

    def health(self):
        return dict(process="ok", database="ok", state="ok", forum="ok")

    def ops(self, manifest, args, **kwargs):
        self.events.append((manifest["version"], args[0]))
        if args[0] == self.failure:
            raise RuntimeError("synthetic")

    def wait_ready(self):
        pass


@pytest.fixture
def installation(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "backups").mkdir()
    (tmp_path / "shared/deployment.toml").write_text(
        "[profiles.remote.mcp]\nenabled=false"
    )
    (tmp_path / "shared/current.json").write_text(
        json.dumps(dict(version="v1.0.0", fingerprint="space"))
    )
    for version in ("v1.0.0", "v1.0.1"):
        directory = tmp_path / "releases" / version
        directory.mkdir(parents=True)
        (directory / "release.json").write_text(json.dumps(manifest(version)))
    with (
        patch.object(remote.platform, "machine", return_value="x86_64"),
        patch.object(
            remote.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)
        ),
    ):
        yield tmp_path


def test_success_and_backup_failure(installation):
    failed = FakeDeployer(installation, "backup")
    with pytest.raises(RuntimeError):
        failed.deploy("v1.0.1")
    assert ("v1.0.0", "up") in failed.events
    assert ("v1.0.1", "up") not in failed.events
    assert failed.record["phase"] == "failed-restored-application"
    controller = FakeDeployer(installation)
    controller.deploy("v1.0.1")
    assert controller.record["phase"] == "success"
    assert (
        json.loads((installation / "shared/current.json").read_text())["version"]
        == "v1.0.1"
    )


@pytest.mark.parametrize("failure", ["pull", "fingerprint"])
def test_preflight_failure_does_not_stop(installation, failure):
    controller = FakeDeployer(installation, failure)
    with pytest.raises((ValueError, RuntimeError)):
        controller.deploy("v1.0.1")
    assert not any(event[1] == "stop" for event in controller.events)


def test_readiness_requires_sqlite_and_forum():
    assert not remote.ready(
        dict(process="ok", database="ok", state="ok", forum="stale")
    )
    assert not remote.ready(dict(process="ok", database="ok", forum="ok"))


def test_migration_failure_never_starts_new_app(installation):
    path = installation / "releases/v1.0.1/release.json"
    candidate = json.loads(path.read_text())
    candidate["migration"] = "backward-compatible"
    path.write_text(json.dumps(candidate))
    controller = FakeDeployer(installation, "db")
    with pytest.raises(RuntimeError):
        controller.deploy("v1.0.1")
    assert ("v1.0.1", "up") not in controller.events
    assert ("v1.0.0", "up") in controller.events
    assert controller.record["failed_phase"] == "migrate"


def test_readiness_failure_restores_only_application(installation):
    controller = FakeDeployer(installation)
    attempts = []

    def health():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("forum stale")

    controller.wait_ready = health
    with pytest.raises(RuntimeError):
        controller.deploy("v1.0.1")
    assert controller.record["phase"] == "failed-restored-application"
    assert ("v1.0.0", "up") in controller.events
    assert not any(action == "restore" for _, action in controller.events)


def test_server_lock_serializes_processes(tmp_path):
    import subprocess
    import time

    (tmp_path / "shared").mkdir()
    code = 'import sys;sys.path.insert(0,sys.argv[1]);from remote import locked;from pathlib import Path\nwith locked(Path(sys.argv[2])): print("entered",flush=True)'
    with remote.locked(tmp_path):
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(DIRECTORY), str(tmp_path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.1)
        assert child.poll() is None
    output, _ = child.communicate(timeout=5)
    assert output.strip() == "entered"


def test_valid_bundle_can_be_received_twice(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "releases").mkdir()
    contents = b"example"
    description = manifest()
    description["files"] = {"config/example.toml": hashlib.sha256(contents).hexdigest()}
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for name, content in [
            ("config/example.toml", contents),
            ("release.json", json.dumps(description).encode()),
        ]:
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    raw = data.getvalue()
    for _ in range(2):
        remote.receive(
            tmp_path, "v1.0.0", hashlib.sha256(raw).hexdigest(), io.BytesIO(raw)
        )
    assert remote.read_release(tmp_path, "v1.0.0")["version"] == "v1.0.0"
    (tmp_path / "releases/v1.0.0/config/example.toml").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        remote.read_release(tmp_path, "v1.0.0")


def test_deploy_health_requires_new_poll_not_persisted_old_success():
    health = dict(process="ok", database="ok", state="ok", forum="ok", last_poll=99)
    assert not remote.ready(health, polled_after=100)
    health["last_poll"] = 101
    assert remote.ready(health, polled_after=100)


def test_server_independently_checks_official_asset_digest(tmp_path):
    from unittest.mock import MagicMock

    metadata = dict(
        draft=False,
        prerelease=False,
        tag_name="v1.0.0",
        assets=[
            dict(
                name="shuiyuan-v1.0.0.tar.gz",
                digest="sha256:" + "a" * 64,
                state="uploaded",
            )
        ],
    )
    opener = MagicMock()
    with patch.object(remote.urllib.request, "build_opener", return_value=opener):
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(
            metadata
        ).encode()
        remote.verify_published(tmp_path, "v1.0.0", "a" * 64)
        with pytest.raises(ValueError, match="digest"):
            remote.verify_published(tmp_path, "v1.0.0", "b" * 64)
        metadata["draft"] = True
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(
            metadata
        ).encode()
        with pytest.raises(ValueError, match="official"):
            remote.verify_published(tmp_path, "v1.0.0", "a" * 64)

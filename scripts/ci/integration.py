"""Launch disposable production-like containers and collect synthetic evidence."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy/compose.test.yaml")]


def run(args, **kwargs):
    return subprocess.run(COMPOSE + args, check=True, **kwargs)


def probe(path):
    result = run(
        [
            "exec",
            "-T",
            "bot",
            "python",
            "-c",
            "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:11451"
            + path
            + "',timeout=3).read().decode())",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        root.chmod(0o755)
        os.environ["VALIDATION_INPUT"] = directory
        (root / "database_app_password").write_text("synthetic-app")
        (root / "cookie.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "domain": "shuiyuan.sjtu.edu.cn",
                    "cookies": {"test": "synthetic"},
                }
            )
        )
        config = """[profiles.remote.embedding]
backend="openai"
model="synthetic-qwen3.7"
dims=1024
base_url="http://fake:8080/v1"
api_key="synthetic"
[profiles.remote.database]
url={env="VALIDATION_DB_URL"}
auto_migrate=false
[profiles.remote.retrieval]
backend="pgvector"
[profiles.remote.paths]
state_dir="/var/lib/shuiyuan"
[profiles.remote.forum]
cookie_file="/validation/cookie.json"
[profiles.remote.providers]
DEEPSEEK_API_KEY="synthetic"
[profiles.remote.mcp]
enabled=false
[profiles.remote.web]
host="0.0.0.0"
port=11451
"""
        (root / "config.toml").write_text(config)
        started = time.monotonic()
        try:
            run(["up", "-d", "--wait", "postgres", "mcp", "fake"])
            run(["run", "--rm", "migrate"])
            run(["run", "--rm", "migrate"])
            run(["run", "--rm", "checks"])
            compatibility = {}
            policy = json.loads((ROOT / "deploy/release-policy.json").read_text())
            if policy["migration"] == "backward-compatible":
                sys.path.insert(0, str(ROOT / "scripts/deploy"))
                from release import compatibility_sources, validate

                sources = compatibility_sources(policy)
                if not sources:
                    raise ValueError(
                        "Backward compatibility requires explicit prior releases"
                    )
                for version in sources:
                    previous_dir = root / version
                    previous_dir.mkdir()
                    metadata = json.loads(
                        subprocess.check_output(
                            [
                                "gh",
                                "release",
                                "view",
                                version,
                                "--json",
                                "isDraft,isPrerelease",
                            ],
                            text=True,
                        )
                    )
                    if metadata["isDraft"] or metadata["isPrerelease"]:
                        raise ValueError(
                            "Compatibility source must be an official release"
                        )
                    subprocess.run(
                        [
                            "gh",
                            "release",
                            "download",
                            version,
                            "--dir",
                            str(previous_dir),
                            "--pattern",
                            "release.json",
                        ],
                        check=True,
                    )
                    previous = validate(
                        json.loads((previous_dir / "release.json").read_text())
                    )
                    image = previous["images"]["bot"]
                    subprocess.run(["docker", "pull", image], check=True)
                    subprocess.run(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "--read-only",
                            "--user",
                            "10001:10001",
                            "--network",
                            "shuiyuan-validation_validation",
                            "--cap-drop",
                            "ALL",
                            "-e",
                            "VALIDATION_DB_URL=postgresql://shuiyuan_app:synthetic-app@postgres:5432/shuiyuan",
                            "-v",
                            str(root) + ":/validation:ro",
                            "-v",
                            "shuiyuan-validation_bot_state:/var/lib/shuiyuan",
                            "-v",
                            str(ROOT / "scripts/ci/compatibility_check.py")
                            + ":/checks/compatibility.py:ro",
                            image,
                            "python",
                            "/checks/compatibility.py",
                        ],
                        check=True,
                    )
                    compatibility[version] = image
            run(["up", "-d", "--wait", "bot"])
            deadline = time.monotonic() + 120
            while True:
                health = json.loads(probe("/api/runtime-health"))
                # Forum login must fail here by design: the validation network is
                # internal-only and the cookie is synthetic, so the worker never
                # polls and forum health stays "stale"/"unknown" forever.
                if health.get("database") == "ok" and health.get("state") == "ok":
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Bot did not become ready")
                time.sleep(3)
            # The management interface must stay online despite the forum login
            # failure (see docs: "login-failure-web").
            assert health.get("forum") != "ok"
            assert "html" in probe("/").lower()
            request_script = "import json,urllib.request; req=urllib.request.Request('http://127.0.0.1:11451/api/conversations',data=b'{\"title\":\"synthetic-ci\"}',headers={'Content-Type':'application/json'}); result=json.load(urllib.request.urlopen(req)); assert result['title']=='synthetic-ci'; print(result['id'])"
            run(["exec", "-T", "bot", "python", "-c", request_script])
            embedding_script = "import asyncio; from shuiyuan_auto_reply.bootstrap.deployment import load_deployment; from shuiyuan_auto_reply.infrastructure.embedding import get_embeddings; load_deployment('/validation/config.toml','remote'); assert len(asyncio.run(get_embeddings().aembed_query('synthetic'))) == 1024"
            run(["exec", "-T", "bot", "python", "-c", embedding_script])
            run(["restart", "bot"])
            time.sleep(5)
            assert json.loads(probe("/api/live"))
            (root / "deny").touch()
            run(["restart", "bot"])
            time.sleep(5)
            assert "html" in probe("/").lower()
            run(["stop", "bot"])
            run(["run", "--rm", "backup-check"])
            (reports / "integration.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "synthetic": True,
                        "compatibility": compatibility,
                        "seconds": round(time.monotonic() - started),
                        "checks": [
                            "role-dml-no-ddl",
                            "migration-idempotence",
                            "corpus-memory",
                            "queue-recovery",
                            "bot-web-entrypoint",
                            "login-failure-web",
                            "backup-restore",
                            "packaged-assets",
                        ],
                    }
                )
            )
        finally:
            with (reports / "containers.log").open("w") as output:
                subprocess.run(
                    COMPOSE + ["logs", "--no-color", "--tail", "200"],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            run(["down", "-v", "--remove-orphans"])


if __name__ == "__main__":
    main()

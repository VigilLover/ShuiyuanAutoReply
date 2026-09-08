"""Trusted workflow-side deployment transport; never execute downloaded code."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from release import VERSION


def main():
    version, operation = os.environ["VERSION"], os.environ["OPERATION"]
    if not VERSION.fullmatch(version) or operation not in {"deploy", "rollback"}:
        raise ValueError("Invalid deployment inputs")
    release = json.loads(
        subprocess.check_output(
            [
                "gh",
                "release",
                "view",
                version,
                "--json",
                "isDraft,isPrerelease,tagName",
            ],
            text=True,
        )
    )
    if release["isDraft"] or release["isPrerelease"] or release["tagName"] != version:
        raise ValueError("Only official releases may be deployed")
    host, user = os.environ["DEPLOY_HOST"], os.environ["DEPLOY_USER"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", host) or not re.fullmatch(
        r"[a-z_][a-z0-9_-]*", user
    ):
        raise ValueError("Invalid SSH destination")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                version,
                "--dir",
                directory,
                "--pattern",
                f"shuiyuan-{version}.tar.gz",
                "--pattern",
                "SHA256SUMS",
            ],
            check=True,
        )
        bundle = root / f"shuiyuan-{version}.tar.gz"
        expected = (root / "SHA256SUMS").read_text().split()
        actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if expected != [actual, bundle.name]:
            raise ValueError("Release checksum mismatch")
        key = root / "key"
        key.write_text(os.environ["DEPLOY_KEY"] + "\n")
        key.chmod(0o600)
        known = root / "known_hosts"
        known.write_text(os.environ["DEPLOY_KNOWN_HOSTS"] + "\n")
        command = [
            "ssh",
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known}",
            "-o",
            "ConnectTimeout=15",
            f"{user}@{host}",
        ]
        with bundle.open("rb") as stream:
            subprocess.run(
                [*command, f"receive {version} {actual}"],
                stdin=stream,
                check=True,
                timeout=300,
            )
        subprocess.run([*command, f"{operation} {version}"], check=True, timeout=1800)


if __name__ == "__main__":
    main()

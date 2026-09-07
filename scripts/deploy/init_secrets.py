#!/usr/bin/env python3
"""Initialize deployment credentials locally without printing secret values."""

import argparse
import getpass
import json
import os
import secrets
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]


def write(path, content):
    # The parent is 0700; container bind mounts must be readable by different UIDs.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, "w") as target:
        target.write(content + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cookie", required=True, help="Trusted JSON from shuiyuan-ops cookie convert"
    )
    args = parser.parse_args()
    document = json.loads(Path(args.cookie).read_text())
    if document.get("domain") != "shuiyuan.sjtu.edu.cn" or document.get("version") != 1:
        raise SystemExit("Invalid cookie JSON")
    root = ROOT / "secrets"
    root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    names = [
        "database_admin_password",
        "database_app_password",
        "database_admin_url",
        "database_url",
        "embedding_key",
        "deepseek_key",
        "forum_cookie.json",
    ]
    if any((root / name).exists() for name in names):
        raise SystemExit("Secret files already exist; refusing to replace them")
    embedding_key = getpass.getpass("Embedding API key: ").strip()
    deepseek_key = getpass.getpass("DeepSeek API key: ").strip()
    if not embedding_key or not deepseek_key:
        raise SystemExit("Both keys are required")
    admin, app = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    write(root / "database_admin_password", admin)
    write(root / "database_app_password", app)
    write(
        root / "database_admin_url",
        f'postgresql://postgres:{quote(admin,safe="")}@postgres:5432/shuiyuan',
    )
    write(
        root / "database_url",
        f'postgresql://shuiyuan_app:{quote(app,safe="")}@postgres:5432/shuiyuan',
    )
    write(root / "embedding_key", embedding_key)
    write(root / "deepseek_key", deepseek_key)
    write(root / "forum_cookie.json", json.dumps(document))
    print("Created secrets in a private directory; no credentials were printed.")


if __name__ == "__main__":
    main()

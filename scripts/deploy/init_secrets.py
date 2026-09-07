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
        "--cookie", help="Trusted JSON from shuiyuan-ops cookie convert"
    )
    parser.add_argument(
        "--image-key-only",
        action="store_true",
        help="Add only the missing image key to an existing deployment",
    )
    args = parser.parse_args()
    root = ROOT / "secrets"
    if args.image_key_only:
        if args.cookie:
            parser.error("--image-key-only cannot be combined with --cookie")
        root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(root, 0o700)
        if (root / "image_key").exists():
            raise SystemExit("Image key already exists; refusing to replace it")
        image_key = getpass.getpass(
            "Image generation API key (empty to disable): "
        ).strip()
        write(root / "image_key", image_key)
        print("Created image key file; existing credentials were preserved.")
        return
    if not args.cookie:
        parser.error("--cookie is required for full initialization")
    document = json.loads(Path(args.cookie).read_text())
    if document.get("domain") != "shuiyuan.sjtu.edu.cn" or document.get("version") != 1:
        raise SystemExit("Invalid cookie JSON")
    root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    names = [
        "database_admin_password",
        "database_app_password",
        "database_admin_url",
        "database_url",
        "embedding_key",
        "deepseek_key",
        "image_key",
        "forum_cookie.json",
    ]
    if any((root / name).exists() for name in names):
        raise SystemExit("Secret files already exist; refusing to replace them")
    embedding_key = getpass.getpass("Embedding API key: ").strip()
    deepseek_key = getpass.getpass("DeepSeek API key: ").strip()
    image_key = getpass.getpass("Image generation API key (empty to disable): ").strip()
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
    write(root / "image_key", image_key)
    write(root / "forum_cookie.json", json.dumps(document))
    print("Created secrets in a private directory; no credentials were printed.")


if __name__ == "__main__":
    main()

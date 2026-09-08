"""Local administrator installs trusted release controller; never called by CI SSH."""

import os
import shutil
from pathlib import Path


def main():
    if os.geteuid() != 0:
        raise SystemExit("Run locally on the server with sudo")
    source = Path(__file__).resolve().parent
    library = Path("/usr/local/lib/shuiyuan")
    library.mkdir(mode=0o755, parents=True, exist_ok=True)
    for name in ("release.py", "remote.py", "ssh_gate.py"):
        shutil.copy2(source / name, library / name)
        os.chown(library / name, 0, 0)
        (library / name).chmod(0o644)
    for name, module in [("shuiyuan-release", "remote"), ("shuiyuan-ssh", "ssh_gate")]:
        script = Path("/usr/local/sbin") / name
        script.write_text(
            '#!/bin/sh\nexec /usr/bin/python3 -I -c \'import sys,runpy; sys.path.insert(0,"/usr/local/lib/shuiyuan"); runpy.run_module("'
            + module
            + '",run_name="__main__")\' "$@"\n'
        )
        os.chown(script, 0, 0)
        script.chmod(0o755)
    root = Path("/opt/shuiyuan")
    for name in ("releases", "shared", "shared/secrets", "backups"):
        (root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chown(root / "backups", 10001, 10001)
    (root / "backups").chmod(0o700)
    print(
        "Controller installed. Configure restricted SSH and initialize the first release locally."
    )


if __name__ == "__main__":
    main()

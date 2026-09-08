#!/usr/bin/python3
"""Forced SSH command: bounded bundle receive and fixed deployment verbs only."""

import os
import shlex
import signal
import sys

from release import VERSION
from remote import ROOT, Deployer, receive, verify_published


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    arguments = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
    if len(arguments) == 3 and arguments[0] == "receive":
        verify_published(ROOT, arguments[1], arguments[2])
        receive(ROOT, arguments[1], arguments[2], sys.stdin.buffer)
    elif (
        len(arguments) == 2
        and arguments[0] in {"deploy", "rollback"}
        and VERSION.fullmatch(arguments[1])
    ):
        Deployer().deploy(arguments[1], rollback=arguments[0] == "rollback")
    elif arguments == ["status"]:
        path = ROOT / "shared/last-deployment.json"
        print(path.read_text() if path.exists() else "{}")
    else:
        raise ValueError("Command not allowed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Deployment rejected: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1)

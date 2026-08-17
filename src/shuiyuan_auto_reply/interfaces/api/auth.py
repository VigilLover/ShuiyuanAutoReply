"""Loopback management authentication."""

import hmac
import os
import secrets
from pathlib import Path

from shuiyuan_auto_reply.infrastructure.persistence import state_directory


class LocalAdminAuth:
    cookie_name = "shuiyuan_admin"

    def __init__(self, token_path: str | Path | None = None) -> None:
        self.token_path = Path(token_path) if token_path else state_directory() / "admin.token"
        self.token = self._load_or_create()
        self.sessions: dict[str, str] = {}

    def _load_or_create(self) -> str:
        if self.token_path.exists():
            return self.token_path.read_text(encoding="utf-8").strip()
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(32)
        self.token_path.write_text(token + "\n", encoding="utf-8")
        try:
            os.chmod(self.token_path, 0o600)
        except OSError:
            pass
        return token

    def login(self, token: str) -> tuple[str, str] | None:
        if not hmac.compare_digest(token, self.token):
            return None
        session = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        self.sessions[session] = csrf
        return session, csrf

    def valid(self, session: str | None) -> bool:
        return bool(session and session in self.sessions)

    def csrf_token(self, session: str | None) -> str | None:
        return self.sessions.get(session) if session else None

    def valid_csrf(self, session: str | None, token: str | None) -> bool:
        expected = self.csrf_token(session)
        return bool(expected and token and hmac.compare_digest(expected, token))

    def logout(self, session: str | None) -> None:
        if session:
            self.sessions.pop(session, None)

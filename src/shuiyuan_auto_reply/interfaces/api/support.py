"""Helpers shared by two or more route modules.

Anything used by exactly one route module stays local to that module instead
of landing here.
"""

from dataclasses import dataclass

from fastapi import HTTPException, Request

DEEPSEEK_VISION_MODEL = "deepseek-flash"


@dataclass(frozen=True, slots=True)
class SessionData:
    token: str


class SessionRegistry:
    """Token-per-session bookkeeping for the legacy ``/api/chat`` contract."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    def authenticate(self, session_id: str, token: str) -> None:
        current = self._sessions.get(session_id)
        if current is None:
            self._sessions[session_id] = SessionData(token)
        elif current.token != token:
            raise PermissionError("invalid token")

    def authorize_removal(self, session_id: str, token: str) -> bool:
        current = self._sessions.get(session_id)
        if current is None:
            return False
        if current.token != token:
            raise PermissionError("invalid token")
        return True

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)


def normalized_base_url(value: str | None) -> str:
    """Accept an OpenAI-compatible base URL; empty means the built-in endpoint."""
    url = (value or "").strip().rstrip("/")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="Base URL 必须以 http:// 或 https:// 开头"
        )
    return url


def state_store(request: Request):
    store = getattr(request.app.state.container, "state_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="本地状态库未启用")
    return store


def profile_defaults(scope: str) -> dict:
    from shuiyuan_auto_reply.application.ports.prompt import PromptScope
    from shuiyuan_auto_reply.bootstrap import AppSettings
    from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository

    settings = AppSettings().providers
    prompt_scope = PromptScope.WEB if scope == "web" else PromptScope.FORUM
    prompt = (
        FilePromptRepository().load("wolf_lumine", set(), prompt_scope).system_prompt
    )
    return {
        "provider": "deepseek",
        "model": DEEPSEEK_VISION_MODEL,
        "base_url": "",
        "api_format": settings.deepseek_api_format.value,
        "fallback_model": None,
        "system_prompt": prompt,
        "enabled_tools": None,
        "disabled_mcp_tools": [],
    }

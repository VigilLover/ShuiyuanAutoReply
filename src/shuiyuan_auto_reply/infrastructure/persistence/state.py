"""SQLite-backed local state for conversations, traces, artifacts, and profiles."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from shuiyuan_auto_reply.domain import (
    AttachmentRef,
    Channel,
    ChatMessage,
    ConversationRef,
    ReplyRequest,
    ReplyResult,
)
from shuiyuan_auto_reply.application.events import current_run_id


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


_DATA_URL_PATTERN = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=]+")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+")


def _safe_event_value(
    value: Any,
    key: str = "",
    *,
    string_limit: int = 2000,
    list_limit: int = 50,
) -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in {"authorization", "cookie", "set_cookie", "api_key", "apikey", "secret"} or normalized.endswith("_api_key"):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(k): _safe_event_value(
                v,
                str(k),
                string_limit=string_limit,
                list_limit=list_limit,
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_event_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
            )
            for item in value[:list_limit]
        ]
    if isinstance(value, str):
        text = _DATA_URL_PATTERN.sub("[DATA_URL_REDACTED]", value)
        text = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
        return text[:string_limit]
    return value


def state_directory() -> Path:
    configured = os.getenv("SHUIYUAN_STATE_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".shuiyuan-auto-reply"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".write-test-{uuid.uuid4().hex}"
        probe.touch(exist_ok=False)
        probe.unlink()
    except OSError:
        if configured:
            raise
        root = Path.cwd() / "var" / "shuiyuan-auto-reply"
        root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: str
    channel: str
    external_id: str
    bot_id: str
    persona_id: str
    title: str
    title_custom: bool
    context_epoch: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class MessageRecord:
    id: str
    conversation_id: str
    epoch: int
    role: str
    content: str
    status: str
    run_id: str | None
    attachments: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class RunEventRecord:
    id: int
    run_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    id: str
    conversation_id: str | None
    run_id: str | None
    local_path: str
    mime_type: str
    byte_count: int
    width: int | None
    height: int | None
    forum_short_path: str | None
    source_kind: str
    source_url: str | None
    filename: str | None
    sha256: str | None
    last_accessed_at: str
    created_at: str

    @property
    def available(self) -> bool:
        return Path(self.local_path).is_file()


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, channel TEXT NOT NULL, external_id TEXT NOT NULL,
  bot_id TEXT NOT NULL, persona_id TEXT NOT NULL, title TEXT NOT NULL,
  title_custom INTEGER NOT NULL DEFAULT 0, context_epoch INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(channel, external_id, bot_id, persona_id)
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  epoch INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'completed', run_id TEXT,
  attachments_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  provider TEXT, model TEXT, status TEXT NOT NULL,
  input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
  error TEXT, started_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS run_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON run_events(run_id, id);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY, conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
  run_id TEXT REFERENCES runs(id) ON DELETE SET NULL, local_path TEXT NOT NULL,
  mime_type TEXT NOT NULL, byte_count INTEGER NOT NULL, width INTEGER, height INTEGER,
  forum_short_path TEXT, source_kind TEXT NOT NULL DEFAULT 'generated',
  source_url TEXT, filename TEXT, sha256 TEXT, last_accessed_at TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_files (
  artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  provider TEXT NOT NULL, credential_fingerprint TEXT NOT NULL,
  file_id TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(artifact_id, provider, credential_fingerprint)
);
CREATE TABLE IF NOT EXISTS runtime_profiles (
  scope TEXT PRIMARY KEY, draft_json TEXT NOT NULL, active_json TEXT NOT NULL,
  active_revision INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, persona_id TEXT NOT NULL,
  version INTEGER NOT NULL, content TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, UNIQUE(scope, persona_id, version)
);
CREATE TABLE IF NOT EXISTS secret_values (
  name TEXT PRIMARY KEY, ciphertext BLOB NOT NULL, last_four TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_tools (
  scope TEXT NOT NULL, name TEXT NOT NULL, source TEXT NOT NULL,
  enabled INTEGER NOT NULL, loaded INTEGER NOT NULL DEFAULT 1,
  error TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(scope, name)
);
"""


class SQLiteStateStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else state_directory() / "state.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        return db

    async def initialize(self) -> None:
        db = await self._connect()
        try:
            await db.executescript(SCHEMA)
            row = await (await db.execute("SELECT version FROM schema_version LIMIT 1")).fetchone()
            if row is None:
                await db.execute("INSERT INTO schema_version(version) VALUES (1)")
            columns = {
                item["name"]
                for item in await (await db.execute("PRAGMA table_info(artifacts)")).fetchall()
            }
            migrations = {
                "source_kind": "ALTER TABLE artifacts ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'generated'",
                "source_url": "ALTER TABLE artifacts ADD COLUMN source_url TEXT",
                "filename": "ALTER TABLE artifacts ADD COLUMN filename TEXT",
                "sha256": "ALTER TABLE artifacts ADD COLUMN sha256 TEXT",
                "last_accessed_at": "ALTER TABLE artifacts ADD COLUMN last_accessed_at TEXT NOT NULL DEFAULT ''",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    await db.execute(statement)
            await db.execute(
                "UPDATE artifacts SET last_accessed_at=created_at WHERE last_accessed_at=''"
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_sha256 ON artifacts(sha256)")
            await db.execute("UPDATE schema_version SET version=2")
            await db.commit()
        finally:
            await db.close()

    @staticmethod
    def _conversation(row: aiosqlite.Row) -> ConversationRecord:
        data = dict(row)
        data["title_custom"] = bool(data["title_custom"])
        return ConversationRecord(**data)

    async def ensure_conversation(self, ref: ConversationRef, *, title: str | None = None) -> ConversationRecord:
        now = utc_now()
        conversation_id = str(uuid.uuid4())
        topic = ref.external_id.removeprefix("topic:")
        default_title = title or ("新对话" if ref.channel in {Channel.WEB, Channel.API} else f"话题 {topic}")
        db = await self._connect()
        try:
            await db.execute(
                """INSERT OR IGNORE INTO conversations
                (id, channel, external_id, bot_id, persona_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (conversation_id, ref.channel.value, ref.external_id, ref.bot_id, ref.persona_id, default_title, now, now),
            )
            row = await (await db.execute(
                """SELECT * FROM conversations WHERE channel=? AND external_id=?
                AND bot_id=? AND persona_id=?""",
                (ref.channel.value, ref.external_id, ref.bot_id, ref.persona_id),
            )).fetchone()
            await db.commit()
            if row is None:
                raise RuntimeError("failed to create conversation")
            return self._conversation(row)
        finally:
            await db.close()

    async def get_conversation(self, conversation_id: str) -> ConversationRecord | None:
        db = await self._connect()
        try:
            row = await (await db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,))).fetchone()
            return self._conversation(row) if row else None
        finally:
            await db.close()

    async def update_title(self, conversation_id: str, title: str, *, custom: bool) -> None:
        db = await self._connect()
        try:
            await db.execute(
                "UPDATE conversations SET title=?, title_custom=?, updated_at=? WHERE id=?",
                (title.strip() or "新对话", int(custom), utc_now(), conversation_id),
            )
            await db.commit()
        finally:
            await db.close()

    async def update_title_for_ref(self, ref: ConversationRef, title: str) -> None:
        record = await self.ensure_conversation(ref, title=title)
        if not record.title_custom:
            await self.update_title(record.id, title, custom=False)

    async def list_conversations(self, *, channel: str | None = None, search: str | None = None, limit: int = 100, offset: int = 0) -> list[ConversationRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if channel:
            clauses.append("channel=?")
            params.append(channel)
        if search:
            clauses.append("title LIKE ?")
            params.append(f"%{search}%")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend((max(1, min(limit, 200)), max(0, offset)))
        db = await self._connect()
        try:
            rows = await (await db.execute(f"SELECT * FROM conversations{where} ORDER BY updated_at DESC LIMIT ? OFFSET ?", params)).fetchall()
            return [self._conversation(row) for row in rows]
        finally:
            await db.close()

    async def append_message(self, conversation_id: str, role: str, content: str, *, run_id: str | None = None, status: str = "completed", attachments: tuple[str, ...] = (), epoch: int | None = None) -> MessageRecord:
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            raise LookupError("conversation not found")
        current_epoch = conversation.context_epoch if epoch is None else epoch
        record = MessageRecord(str(uuid.uuid4()), conversation_id, current_epoch, role, content, status, run_id, attachments, utc_now())
        db = await self._connect()
        try:
            await db.execute(
                """INSERT INTO messages
                (id, conversation_id, epoch, role, content, status, run_id, attachments_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.id, conversation_id, current_epoch, role, content, status, run_id, json.dumps(attachments), record.created_at),
            )
            await db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (record.created_at, conversation_id))
            if role == "user" and not conversation.title_custom and conversation.title == "新对话":
                title = " ".join(content.strip().split())[:24] or "新对话"
                await db.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))
            await db.commit()
            return record
        finally:
            await db.close()

    async def list_messages(self, conversation_id: str, *, current_epoch_only: bool = False) -> list[MessageRecord]:
        conversation = await self.get_conversation(conversation_id)
        if conversation is None:
            return []
        sql = "SELECT * FROM messages WHERE conversation_id=?"
        params: list[Any] = [conversation_id]
        if current_epoch_only:
            sql += " AND epoch=?"
            params.append(conversation.context_epoch)
        sql += " ORDER BY created_at, rowid"
        db = await self._connect()
        try:
            rows = await (await db.execute(sql, params)).fetchall()
            return [MessageRecord(
                id=row["id"], conversation_id=row["conversation_id"], epoch=row["epoch"], role=row["role"],
                content=row["content"], status=row["status"], run_id=row["run_id"],
                attachments=tuple(json.loads(row["attachments_json"])), created_at=row["created_at"],
            ) for row in rows]
        finally:
            await db.close()

    async def clear_context(self, conversation_id: str) -> None:
        db = await self._connect()
        try:
            await db.execute("UPDATE conversations SET context_epoch=context_epoch+1, updated_at=? WHERE id=?", (utc_now(), conversation_id))
            await db.commit()
        finally:
            await db.close()
        await self.append_message(conversation_id, "system", "上下文已清除")

    async def delete_conversation(self, conversation_id: str) -> list[str]:
        db = await self._connect()
        try:
            rows = await (await db.execute("SELECT local_path FROM artifacts WHERE conversation_id=?", (conversation_id,))).fetchall()
            await db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            await db.commit()
            return [row["local_path"] for row in rows]
        finally:
            await db.close()

    async def create_run(self, request_id: str, conversation_id: str, *, provider: str | None = None, model: str | None = None) -> str:
        run_id = str(uuid.uuid4())
        db = await self._connect()
        try:
            await db.execute(
                "INSERT INTO runs(id, request_id, conversation_id, provider, model, status, started_at) VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (run_id, request_id, conversation_id, provider, model, utc_now()),
            )
            await db.commit()
            return run_id
        finally:
            await db.close()

    async def append_event(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        is_model_prompt = event_type == "model.prompt_prepared"
        is_tool_instruction = event_type == "tool.started"
        encoded = json.dumps(
            _safe_event_value(
                payload or {},
                string_limit=(
                    65536 if is_model_prompt else 32768 if is_tool_instruction else 2000
                ),
                list_limit=200 if is_model_prompt or is_tool_instruction else 50,
            ),
            ensure_ascii=False,
            default=str,
        )
        event_limit = (
            262144 if is_model_prompt else 65536 if is_tool_instruction else 4096
        )
        if len(encoded) > event_limit:
            encoded = json.dumps(
                {"summary": encoded[:event_limit], "truncated": True},
                ensure_ascii=False,
            )
        db = await self._connect()
        try:
            await db.execute("INSERT INTO run_events(run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)", (run_id, event_type, encoded, utc_now()))
            await db.commit()
        finally:
            await db.close()

    async def finish_run(self, run_id: str, *, status: str, error: str | None = None, usage: dict[str, int] | None = None) -> None:
        usage = usage or {}
        db = await self._connect()
        try:
            await db.execute(
                "UPDATE runs SET status=?, error=?, input_tokens=?, output_tokens=?, total_tokens=?, finished_at=? WHERE id=?",
                (status, error, usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"), utc_now(), run_id),
            )
            await db.commit()
        finally:
            await db.close()

    async def list_events_for_conversation(self, conversation_id: str) -> list[RunEventRecord]:
        db = await self._connect()
        try:
            rows = await (await db.execute(
                """SELECT e.* FROM run_events e JOIN runs r ON r.id=e.run_id
                WHERE r.conversation_id=? ORDER BY e.id""", (conversation_id,)
            )).fetchall()
            return [RunEventRecord(row["id"], row["run_id"], row["event_type"], json.loads(row["payload_json"]), row["created_at"]) for row in rows]
        finally:
            await db.close()

    async def list_events_for_request(self, request_id: str) -> list[RunEventRecord]:
        db = await self._connect()
        try:
            rows = await (await db.execute(
                """SELECT e.* FROM run_events e JOIN runs r ON r.id=e.run_id
                WHERE r.request_id=? ORDER BY e.id""",
                (request_id,),
            )).fetchall()
            return [
                RunEventRecord(
                    row["id"], row["run_id"], row["event_type"],
                    json.loads(row["payload_json"]), row["created_at"]
                )
                for row in rows
            ]
        finally:
            await db.close()

    async def append_event_for_request(
        self, request_id: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        db = await self._connect()
        try:
            row = await (await db.execute(
                "SELECT id FROM runs WHERE request_id=? ORDER BY started_at DESC LIMIT 1",
                (request_id,),
            )).fetchone()
        finally:
            await db.close()
        if row is not None:
            await self.append_event(row["id"], event_type, payload)

    async def register_artifact(
        self,
        *,
        artifact_id: str,
        local_path: str,
        mime_type: str,
        byte_count: int,
        width: int | None = None,
        height: int | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        source_kind: str = "generated",
        source_url: str | None = None,
        filename: str | None = None,
        sha256: str | None = None,
    ) -> None:
        now = utc_now()
        db = await self._connect()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO artifacts
                (id, conversation_id, run_id, local_path, mime_type, byte_count, width, height,
                 source_kind, source_url, filename, sha256, last_accessed_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id, conversation_id, run_id, local_path, mime_type,
                    byte_count, width, height, source_kind, source_url, filename,
                    sha256, now, now,
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def attach_artifact(self, artifact_id: str, *, conversation_id: str, run_id: str | None = None) -> None:
        db = await self._connect()
        try:
            await db.execute("UPDATE artifacts SET conversation_id=?, run_id=COALESCE(?, run_id) WHERE id=?", (conversation_id, run_id, artifact_id))
            await db.commit()
        finally:
            await db.close()

    async def set_forum_short_path(self, artifact_id: str, short_path: str) -> None:
        db = await self._connect()
        try:
            await db.execute("UPDATE artifacts SET forum_short_path=? WHERE id=?", (short_path, artifact_id))
            await db.commit()
        finally:
            await db.close()

    async def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        db = await self._connect()
        try:
            row = await (await db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,))).fetchone()
            if row:
                await db.execute(
                    "UPDATE artifacts SET last_accessed_at=? WHERE id=?",
                    (utc_now(), artifact_id),
                )
                await db.commit()
            return ArtifactRecord(**dict(row)) if row else None
        finally:
            await db.close()

    async def find_artifact_by_sha256(
        self, conversation_id: str | None, sha256: str, source_kind: str
    ) -> ArtifactRecord | None:
        if conversation_id is None:
            return None
        db = await self._connect()
        try:
            row = await (await db.execute(
                """SELECT * FROM artifacts
                WHERE conversation_id=? AND sha256=? AND source_kind=?
                ORDER BY created_at DESC LIMIT 1""",
                (conversation_id, sha256, source_kind),
            )).fetchone()
            record = ArtifactRecord(**dict(row)) if row else None
            return record if record and record.available else None
        finally:
            await db.close()

    async def get_provider_file(
        self, artifact_id: str, provider: str, credential_fingerprint: str
    ) -> dict[str, str] | None:
        db = await self._connect()
        try:
            row = await (await db.execute(
                """SELECT file_id, expires_at FROM provider_files
                WHERE artifact_id=? AND provider=? AND credential_fingerprint=?""",
                (artifact_id, provider, credential_fingerprint),
            )).fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def upsert_provider_file(
        self,
        *,
        artifact_id: str,
        provider: str,
        credential_fingerprint: str,
        file_id: str,
        expires_at: str,
    ) -> None:
        db = await self._connect()
        try:
            await db.execute(
                """INSERT INTO provider_files
                (artifact_id, provider, credential_fingerprint, file_id, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id, provider, credential_fingerprint) DO UPDATE SET
                  file_id=excluded.file_id, expires_at=excluded.expires_at,
                  created_at=excluded.created_at""",
                (
                    artifact_id, provider, credential_fingerprint, file_id,
                    expires_at, utc_now(),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def list_provider_files_for_conversation(
        self, conversation_id: str
    ) -> list[dict[str, str]]:
        db = await self._connect()
        try:
            rows = await (await db.execute(
                """SELECT p.provider, p.file_id, p.expires_at
                FROM provider_files p JOIN artifacts a ON a.id=p.artifact_id
                WHERE a.conversation_id=?""",
                (conversation_id,),
            )).fetchall()
            return [dict(row) for row in rows]
        finally:
            await db.close()

    async def get_profile(self, scope: str, defaults: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        encoded = json.dumps(defaults, ensure_ascii=False)
        db = await self._connect()
        try:
            await db.execute("INSERT OR IGNORE INTO runtime_profiles(scope, draft_json, active_json, updated_at) VALUES (?, ?, ?, ?)", (scope, encoded, encoded, now))
            row = await (await db.execute("SELECT * FROM runtime_profiles WHERE scope=?", (scope,))).fetchone()
            await db.commit()
            if row is None:
                raise RuntimeError("failed to initialize runtime profile")
            return {"scope": scope, "draft": json.loads(row["draft_json"]), "active": json.loads(row["active_json"]), "active_revision": row["active_revision"], "updated_at": row["updated_at"]}
        finally:
            await db.close()

    async def save_profile_draft(self, scope: str, value: dict[str, Any]) -> None:
        db = await self._connect()
        try:
            await db.execute("UPDATE runtime_profiles SET draft_json=?, updated_at=? WHERE scope=?", (json.dumps(value, ensure_ascii=False), utc_now(), scope))
            await db.commit()
        finally:
            await db.close()

    async def apply_profile(self, scope: str, persona_id: str = "wolf_lumine") -> int:
        db = await self._connect()
        try:
            profile = await (await db.execute(
                "SELECT draft_json FROM runtime_profiles WHERE scope=?", (scope,)
            )).fetchone()
            if profile is None:
                raise LookupError("profile not found")
            prompt = json.loads(profile["draft_json"]).get("system_prompt", "")
            version_row = await (await db.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM prompt_versions WHERE scope=? AND persona_id=?",
                (scope, persona_id),
            )).fetchone()
            version = int(version_row["version"]) + 1
            now = utc_now()
            await db.execute(
                "UPDATE prompt_versions SET active=0 WHERE scope=? AND persona_id=?",
                (scope, persona_id),
            )
            await db.execute(
                """INSERT INTO prompt_versions(scope, persona_id, version, content, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)""",
                (scope, persona_id, version, prompt, now),
            )
            await db.execute("UPDATE runtime_profiles SET active_json=draft_json, active_revision=active_revision+1, updated_at=? WHERE scope=?", (now, scope))
            row = await (await db.execute("SELECT active_revision FROM runtime_profiles WHERE scope=?", (scope,))).fetchone()
            await db.commit()
            return int(row["active_revision"])
        finally:
            await db.close()

    async def save_prompt_version(
        self, scope: str, persona_id: str, content: str, *, active: bool = True
    ) -> int:
        db = await self._connect()
        try:
            row = await (await db.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM prompt_versions WHERE scope=? AND persona_id=?",
                (scope, persona_id),
            )).fetchone()
            version = int(row["version"]) + 1
            if active:
                await db.execute(
                    "UPDATE prompt_versions SET active=0 WHERE scope=? AND persona_id=?",
                    (scope, persona_id),
                )
            await db.execute(
                """INSERT INTO prompt_versions(scope, persona_id, version, content, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (scope, persona_id, version, content, int(active), utc_now()),
            )
            await db.commit()
            return version
        finally:
            await db.close()

    async def replace_tool_catalog(
        self, scope: str, tools: list[dict[str, Any]]
    ) -> None:
        db = await self._connect()
        try:
            await db.execute("DELETE FROM runtime_tools WHERE scope=?", (scope,))
            now = utc_now()
            await db.executemany(
                """INSERT INTO runtime_tools(scope, name, source, enabled, loaded, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        scope,
                        tool["name"],
                        tool.get("source", "runtime"),
                        int(tool.get("enabled", True)),
                        int(tool.get("loaded", True)),
                        tool.get("error"),
                        now,
                    )
                    for tool in tools
                ],
            )
            await db.commit()
        finally:
            await db.close()

    async def list_tool_catalog(self, scope: str) -> list[dict[str, Any]]:
        db = await self._connect()
        try:
            rows = await (await db.execute(
                "SELECT name, source, enabled, loaded, error FROM runtime_tools WHERE scope=? ORDER BY rowid",
                (scope,),
            )).fetchall()
            return [
                {
                    "name": row["name"],
                    "source": row["source"],
                    "enabled": bool(row["enabled"]),
                    "loaded": bool(row["loaded"]),
                    "error": row["error"],
                }
                for row in rows
            ]
        finally:
            await db.close()


class SQLiteSessionRepository:
    def __init__(self, store: SQLiteStateStore) -> None:
        self.store = store

    async def load(self, key: ConversationRef) -> list[ChatMessage]:
        conversation = await self.store.ensure_conversation(key)
        records = await self.store.list_messages(conversation.id, current_epoch_only=True)
        history = []
        for record in records:
            if record.role not in {"user", "assistant"}:
                continue
            attachments = []
            for artifact_id in record.attachments:
                artifact = await self.store.get_artifact(artifact_id)
                if artifact is None or not artifact.available:
                    continue
                attachments.append(
                    AttachmentRef(
                        f"artifact://{artifact.id}",
                        artifact.mime_type,
                        artifact.id,
                        artifact.source_kind,
                        artifact.source_url,
                        artifact.filename,
                        artifact.width,
                        artifact.height,
                    )
                )
            history.append(ChatMessage(record.role, record.content, tuple(attachments)))
        return history[-16:]

    async def append(self, key: ConversationRef, request: ReplyRequest, result: ReplyResult) -> None:
        conversation = await self.store.ensure_conversation(key)
        run_id = current_run_id()
        input_artifact_ids = tuple(
            attachment.url.removeprefix("artifact://")
            for attachment in (*request.attachments, *result.input_attachments)
            if attachment.url.startswith("artifact://")
        )
        await self.store.append_message(
            conversation.id,
            "user",
            request.content,
            run_id=run_id,
            attachments=tuple(dict.fromkeys(input_artifact_ids)),
        )
        artifact_ids = tuple(a.url.removeprefix("artifact://") for a in result.attachments if a.url.startswith("artifact://"))
        await self.store.append_message(
            conversation.id,
            "assistant",
            result.text,
            run_id=run_id,
            attachments=artifact_ids,
        )
        for artifact_id in tuple(dict.fromkeys((*input_artifact_ids, *artifact_ids))):
            await self.store.attach_artifact(
                artifact_id, conversation_id=conversation.id, run_id=run_id
            )

    async def clear(self, key: ConversationRef) -> None:
        conversation = await self.store.ensure_conversation(key)
        await self.store.clear_context(conversation.id)


__all__ = ["ArtifactRecord", "ConversationRecord", "MessageRecord", "RunEventRecord", "SQLiteSessionRepository", "SQLiteStateStore", "state_directory"]

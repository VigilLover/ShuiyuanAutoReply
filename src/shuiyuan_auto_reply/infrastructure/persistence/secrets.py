"""Encrypted local secret persistence for the management UI."""

import os
from pathlib import Path

from cryptography.fernet import Fernet

from .state import SQLiteStateStore, state_directory, utc_now


class LocalSecretVault:
    def __init__(self, store: SQLiteStateStore, key_path: str | Path | None = None) -> None:
        self.store = store
        self.key_path = Path(key_path) if key_path else state_directory() / "master.key"
        self._fernet: Fernet | None = None

    def _cipher(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        if self.key_path.exists():
            key = self.key_path.read_bytes().strip()
        else:
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            key = Fernet.generate_key()
            self.key_path.write_bytes(key + b"\n")
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
        self._fernet = Fernet(key)
        return self._fernet

    async def set(self, name: str, value: str) -> None:
        encrypted = self._cipher().encrypt(value.encode())
        db = await self.store._connect()
        try:
            await db.execute(
                """INSERT INTO secret_values(name, ciphertext, last_four, updated_at)
                VALUES (?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET
                ciphertext=excluded.ciphertext, last_four=excluded.last_four,
                updated_at=excluded.updated_at""",
                (name, encrypted, value[-4:], utc_now()),
            )
            await db.commit()
        finally:
            await db.close()

    async def get(self, name: str) -> str | None:
        db = await self.store._connect()
        try:
            row = await (await db.execute("SELECT ciphertext FROM secret_values WHERE name=?", (name,))).fetchone()
            return self._cipher().decrypt(row["ciphertext"]).decode() if row else None
        finally:
            await db.close()

    async def metadata(self, name: str) -> dict[str, object]:
        db = await self.store._connect()
        try:
            row = await (await db.execute("SELECT last_four, updated_at FROM secret_values WHERE name=?", (name,))).fetchone()
            return {"configured": bool(row), "last_four": row["last_four"] if row else None, "updated_at": row["updated_at"] if row else None}
        finally:
            await db.close()

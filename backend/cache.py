"""Tiny SQLite key/value cache for expensive operations."""
import json
import sqlite3
import threading
from typing import Any, Optional

from .config import DB_PATH

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        _conn.commit()
    return _conn


def get(key: str) -> Optional[Any]:
    with _lock:
        cur = conn().execute("SELECT v FROM kv WHERE k = ?", (key,))
        row = cur.fetchone()
    return json.loads(row[0]) if row else None


def put(key: str, value: Any) -> None:
    with _lock:
        conn().execute(
            "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn().commit()

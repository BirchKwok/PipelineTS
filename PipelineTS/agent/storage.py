"""ApexBase storage layer for PipelineTS agent.

Persists configuration, session history, and uploaded data using ApexBase
(https://github.com/BirchKwok/ApexBase). Falls back gracefully if not installed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_logger = logging.getLogger("PipelineTS.agent.storage")

_APEX_AVAILABLE = False
try:
    from apexbase import ApexClient
    _APEX_AVAILABLE = True
except ImportError:
    _logger.debug("apexbase not installed; agent storage unavailable")

DEFAULT_STORAGE_DIR = Path.home() / ".pipelinets" / "apex"


class AgentStorage:
    """Persistent storage for agent config, sessions, and uploaded data."""

    def __init__(self, storage_dir: str | Path = None):
        self._dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None
        self._tables_ready = False
        if _APEX_AVAILABLE:
            self._init_apex()

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    def _init_apex(self):
        try:
            self._client = ApexClient(str(self._dir))
            # Reuse or create tables
            existing = set(self._client.list_tables())
            if "config" not in existing:
                self._client.create_table("config")
            if "sessions" not in existing:
                self._client.create_table("sessions")
            if "data_files" not in existing:
                self._client.create_table("data_files")
            # Ensure at least one table is selected so subsequent queries work.
            # When tables already exist (restart), no create_table() call is
            # made and the client has no selected table, causing all queries
            # to fail with "No table selected".
            tbls = self._client.list_tables()
            if tbls:
                self._client.use_table(tbls[0])
            self._tables_ready = True
        except Exception as exc:
            _logger.warning("ApexBase init failed: %s", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None and self._tables_ready

    # ------------------------------------------------------------------
    #  Config (key-value)
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, str]:
        """Return all stored config as a dict."""
        if not self.available:
            return {}
        try:
            r = self._client.execute("SELECT k, v FROM config")
            rows = r.to_dict()
            return {row["k"]: row["v"] for row in rows}
        except Exception as exc:
            _logger.warning("get_config failed: %s", exc)
            return {}

    def set_config(self, key: str, value: str):
        """Upsert a single config key."""
        if not self.available:
            return
        try:
            v = value.replace("'", "''")
            # DELETE may fail if the table has no columns yet (brand new);
            # INSERT will auto-create them.  We ignore the DELETE error.
            try:
                self._client.execute(f"DELETE FROM config WHERE k = '{key}'")
            except Exception:
                pass
            self._client.execute(f"INSERT INTO config (k, v) VALUES ('{key}', '{v}')")
        except Exception as exc:
            _logger.warning("set_config(%s) failed: %s", key, exc)

    def set_config_bulk(self, items: dict[str, str]):
        """Upsert multiple config keys at once."""
        if not self.available:
            return
        try:
            for k, v in items.items():
                if v:
                    ev = v.replace("'", "''")
                    try:
                        self._client.execute(f"DELETE FROM config WHERE k = '{k}'")
                    except Exception:
                        pass  # columns may not exist yet
                    self._client.execute(f"INSERT INTO config (k, v) VALUES ('{k}', '{ev}')")
        except Exception as exc:
            _logger.warning("set_config_bulk failed: %s", exc)

    # ------------------------------------------------------------------
    #  Sessions (chat history)
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """Return all sessions ordered by most recent first."""
        if not self.available:
            return []
        try:
            r = self._client.execute(
                "SELECT id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            return r.to_dict()
        except Exception as exc:
            _logger.warning("list_sessions failed: %s", exc)
            return []

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session with its messages."""
        if not self.available:
            return None
        try:
            sid = session_id.replace("'", "''")
            r = self._client.execute(f"SELECT * FROM sessions WHERE id = '{sid}'")
            rows = r.to_dict()
            if not rows:
                return None
            s = rows[0]
            s["messages"] = json.loads(s.get("messages", "[]"))
            return s
        except Exception as exc:
            _logger.warning("get_session(%s) failed: %s", session_id, exc)
            return None

    def save_session(self, session_id: str, name: str, messages: list[dict]):
        """Save or update a session."""
        if not self.available:
            return
        try:
            sid = session_id.replace("'", "''")
            sname = (name or "Untitled").replace("'", "''")
            msgs = json.dumps(messages, ensure_ascii=False).replace("'", "''")
            existing = self._client.execute(f"SELECT COUNT(*) AS cnt FROM sessions WHERE id = '{sid}'")
            cnt = existing.to_dict()[0]["cnt"]
            if cnt > 0:
                self._client.execute(
                    f"UPDATE sessions SET name = '{sname}', messages = '{msgs}', "
                    f"updated_at = {int(pd.Timestamp.now().timestamp() * 1000)} "
                    f"WHERE id = '{sid}'"
                )
            else:
                now = int(pd.Timestamp.now().timestamp() * 1000)
                self._client.execute(
                    f"INSERT INTO sessions (id, name, messages, created_at, updated_at) "
                    f"VALUES ('{sid}', '{sname}', '{msgs}', {now}, {now})"
                )
        except Exception as exc:
            _logger.warning("save_session(%s) failed: %s", session_id, exc)

    def delete_session(self, session_id: str):
        """Delete a session."""
        if not self.available:
            return
        try:
            sid = session_id.replace("'", "''")
            self._client.execute(f"DELETE FROM sessions WHERE id = '{sid}'")
        except Exception as exc:
            _logger.warning("delete_session(%s) failed: %s", session_id, exc)

    # ------------------------------------------------------------------
    #  Uploaded data
    # ------------------------------------------------------------------

    def save_data(self, file_id: str, filename: str, df: pd.DataFrame):
        """Persist an uploaded DataFrame."""
        if not self.available:
            return
        try:
            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            csv_content = buf.getvalue().replace("'", "''")

            fid = file_id.replace("'", "''")
            fname = filename.replace("'", "''")
            now = int(pd.Timestamp.now().timestamp() * 1000)
            try:
                self._client.execute(f"DELETE FROM data_files WHERE id = '{fid}'")
            except Exception:
                pass  # columns may not exist yet
            self._client.execute(
                f"INSERT INTO data_files (id, filename, csv_content, created_at) "
                f"VALUES ('{fid}', '{fname}', '{csv_content}', {now})"
            )
        except Exception as exc:
            _logger.warning("save_data(%s) failed: %s", file_id, exc)

    def get_data(self, file_id: str) -> Optional[pd.DataFrame]:
        """Retrieve a persisted DataFrame."""
        if not self.available:
            return None
        try:
            fid = file_id.replace("'", "''")
            r = self._client.execute(f"SELECT csv_content FROM data_files WHERE id = '{fid}'")
            rows = r.to_dict()
            if not rows:
                return None
            import io
            return pd.read_csv(io.StringIO(rows[0]["csv_content"]))
        except Exception as exc:
            _logger.warning("get_data(%s) failed: %s", file_id, exc)
            return None

    def list_data_files(self) -> list[dict]:
        """List all stored data files."""
        if not self.available:
            return []
        try:
            r = self._client.execute(
                "SELECT id, filename, created_at FROM data_files ORDER BY created_at DESC"
            )
            return r.to_dict()
        except Exception as exc:
            _logger.warning("list_data_files failed: %s", exc)
            return []

    def delete_data(self, file_id: str):
        """Delete a stored data file."""
        if not self.available:
            return
        try:
            fid = file_id.replace("'", "''")
            self._client.execute(f"DELETE FROM data_files WHERE id = '{fid}'")
        except Exception as exc:
            _logger.warning("delete_data(%s) failed: %s", file_id, exc)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception as exc:
                _logger.debug("close failed: %s", exc)

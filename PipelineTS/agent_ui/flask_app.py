"""Flask web backend for the PipelineTS agent.

Launch with:
    pipelinets web
    python -m PipelineTS.agent.flask_app
"""

from __future__ import annotations

# Prevent macOS GUI crashes from matplotlib in background threads
import matplotlib
matplotlib.use("Agg")

import json
import os
import queue
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from PipelineTS.agent.config import Config
from PipelineTS.agent.agent import TSAgent
from PipelineTS.agent.session import Session
from PipelineTS.agent.storage import AgentStorage


# ---------------------------------------------------------------------------
#  Application state
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.config = Config()
        self.storage = AgentStorage()
        self._load_stored_config()
        self.agent: Optional[TSAgent] = None
        self.session: Optional[Session] = None
        self._session_id = str(uuid.uuid4())
        self._session_name = "New Session"
        self.plot_dir = Path(tempfile.gettempdir()) / "pipelinets_plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _load_stored_config(self):
        if not self.storage.available:
            return
        stored = self.storage.get_config()
        if stored:
            env_map = {
                "provider": "PIPELINETS_PROVIDER",
                "api_format": "PIPELINETS_API_FORMAT",
                "openai_api_key": "OPENAI_API_KEY",
                "openai_base_url": "OPENAI_BASE_URL",
                "openai_model": "OPENAI_MODEL",
                "anthropic_api_key": "ANTHROPIC_API_KEY",
                "anthropic_model": "ANTHROPIC_MODEL",
                "lang": "PIPELINETS_LANG",
                "multimodal": "PIPELINETS_MULTIMODAL",
            }
            self.config.update(**{k: v for k, v in stored.items() if v and not os.environ.get(env_map.get(k, ""))})

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self):
        self._session_id = str(uuid.uuid4())
        self._session_name = "New Session"
        if self.agent is not None:
            self.agent.reset()
            self.session = self.agent.session
        else:
            self.session = Session()

    def ensure_session(self) -> Session:
        if self.session is None:
            self.session = Session()
        return self.session

    def get_or_create_agent(self, **overrides):
        self.config.update(**{k: v for k, v in overrides.items() if v})

        provider = self.config.resolve_provider()
        api_key = self.config.resolve_api_key()
        base_url = self.config.resolve_base_url()
        model = self.config.resolve_model()
        lang = self.config.lang
        multimodal = self.config.multimodal

        if not api_key:
            raise ValueError(
                f"No API key configured for {provider}. Go to Settings to configure."
            )

        previous_session = self.session
        agent = TSAgent(
            provider=provider,
            api_key=api_key,
            base_url=base_url or None,
            model=model,
            lang=lang,
            verbose=False,
            plot_dir=str(self.plot_dir),
            multimodal=multimodal,
        )
        if previous_session is not None:
            agent.session = previous_session
            agent.executor.session = previous_session
        self.agent = agent
        self.session = self.agent.session

        summary = f"{provider} | {model}"
        if lang:
            summary += f" | {lang}"
        if agent.multimodal_enabled:
            summary += " | vision"
        return self.agent, summary


_app = AppState()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _mask_secret(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]


_API_PROFILE_FIELDS = (
    "provider",
    "api_format",
    "openai_api_key",
    "openai_base_url",
    "openai_model",
    "anthropic_api_key",
    "anthropic_model",
)


def _config_source() -> dict:
    source = dict(_app.config._data)
    if _app.storage.available:
        source.update({k: v for k, v in _app.storage.get_config().items() if v})
    return source


def _load_api_profiles(source: dict | None = None) -> dict[str, dict]:
    source = source or _config_source()
    raw = source.get("api_profiles", "")
    if isinstance(raw, dict):
        profiles = raw
    else:
        try:
            profiles = json.loads(raw) if raw else {}
        except Exception:
            profiles = {}
    if not isinstance(profiles, dict):
        return {}
    cleaned = {}
    for alias, profile in profiles.items():
        if isinstance(profile, dict):
            cleaned[str(alias)] = {k: str(v) for k, v in profile.items() if v is not None}
    return cleaned


def _legacy_api_profile(source: dict) -> dict:
    profile = {k: str(source.get(k, "") or "") for k in _API_PROFILE_FIELDS}
    if not profile.get("api_format"):
        profile["api_format"] = "anthropic" if profile.get("provider") == "anthropic" else "openai"
    if not profile.get("provider"):
        profile["provider"] = "openai"
    return profile


def _profiles_with_legacy(source: dict) -> dict[str, dict]:
    profiles = _load_api_profiles(source)
    legacy = _legacy_api_profile(source)
    has_legacy = any(legacy.get(k) for k in ("openai_api_key", "anthropic_api_key", "openai_model", "anthropic_model", "openai_base_url"))
    if has_legacy and not profiles:
        profiles["default"] = legacy
    return profiles


def _profile_aliases(profiles: dict[str, dict]) -> list[dict]:
    aliases = []
    for alias, profile in profiles.items():
        api_format = profile.get("api_format") or ("anthropic" if profile.get("provider") == "anthropic" else "openai")
        model = profile.get("anthropic_model") if api_format == "anthropic" else profile.get("openai_model")
        aliases.append({
            "alias": alias,
            "provider": profile.get("provider", ""),
            "api_format": api_format,
            "model": model or "",
        })
    return aliases


def _settings_result(alias: str = "") -> dict:
    c = _app.config
    source = _config_source()
    profiles = _profiles_with_legacy(source)
    active_alias = (alias or source.get("active_api_alias") or c.active_api_alias or "").strip()
    if active_alias not in profiles:
        active_alias = next(iter(profiles), active_alias or "default")
    profile = dict(profiles.get(active_alias, _legacy_api_profile(source)))
    real_openai_key = profile.get("openai_api_key", "")
    real_anthropic_key = profile.get("anthropic_api_key", "")
    provider = profile.get("provider") or c.provider
    api_format = profile.get("api_format") or ("anthropic" if provider == "anthropic" else "openai")
    return {
        "api_alias": active_alias,
        "active_api_alias": active_alias,
        "api_aliases": _profile_aliases(profiles),
        "provider": provider,
        "api_format": api_format,
        "openai_api_key": _mask_secret(real_openai_key),
        "openai_api_key_set": bool(real_openai_key),
        "openai_base_url": profile.get("openai_base_url", c.openai_base_url),
        "openai_model": profile.get("openai_model", c.openai_model),
        "anthropic_api_key": _mask_secret(real_anthropic_key),
        "anthropic_api_key_set": bool(real_anthropic_key),
        "anthropic_model": profile.get("anthropic_model", c.anthropic_model),
        "lang": c.lang,
        "multimodal": c.multimodal,
    }


def _column_values(value) -> list:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def _column_payload(value):
    values = _column_values(value)
    if len(values) > 1:
        return values
    return values[0] if values else ""


def _primary_column(value) -> str:
    values = _column_values(value)
    return values[0] if values else ""


def _format_column_payload(value) -> str:
    return ", ".join(_column_values(value))


_UPLOAD_PREFIX_RE = re.compile(r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F-]{36})_+")


def _display_filename(value) -> str:
    name = Path(str(value or "Dataset")).name
    while True:
        cleaned = _UPLOAD_PREFIX_RE.sub("", name)
        if cleaned == name:
            return cleaned or "Dataset"
        name = cleaned


def _get_session_status() -> dict:
    provider = _app.config.provider
    model = _app.config.resolve_model()
    configured = {"provider": provider, "model": model} if provider != "auto" and model else None
    has_credentials = bool(_app.config.resolve_api_key())
    info = {
        "ready": _app.agent is not None or has_credentials,
        "session_id": _app.session_id,
        "message": "" if has_credentials else "No API key configured. Go to Settings to set up.",
        "configured": configured,
    }
    s = _app.session
    if s is None:
        info["data"] = None
        info["model"] = None
        return info
    if s.has_data():
        info["data"] = {
            "rows": len(s.data),
            "columns": list(s.data.columns),
            "time_col": s.time_col,
            "target_col": s.target_col,
            "datasets": _serialize_datasets(s),
        }
    elif s.has_data_file():
        active_item = _dataset_for_id(s, s.metadata.get("active_dataset_id"))
        info["data"] = {
            "rows": active_item.get("rows") if active_item else None,
            "columns": active_item.get("columns", []) if active_item else [],
            "time_col": active_item.get("time_col") if active_item else s.time_col,
            "target_col": active_item.get("target_col") if active_item else s.target_col,
            "filepath": s.data_filepath,
            "filename": _display_filename(active_item.get("filename")) if active_item else _display_filename(s.data_filepath),
            "pending": True,
            "datasets": _serialize_datasets(s),
        }
    else:
        info["data"] = None
    if s.has_model():
        info["model"] = {
            "type": s.model_type,
            "best_model": s.best_model_name,
        }
    else:
        info["model"] = None
    return info


def _session_datasets(s: Session) -> list:
    datasets = s.metadata.get("datasets")
    if not isinstance(datasets, list):
        datasets = []
        s.metadata["datasets"] = datasets
    return datasets


def _dataset_for_id(s: Session, dataset_id: str) -> Optional[dict]:
    if not dataset_id:
        return None
    for item in _session_datasets(s):
        if item.get("id") == dataset_id:
            return item
    return None


def _serialize_datasets(s: Session) -> list:
    active_id = s.metadata.get("active_dataset_id")
    records = []
    for item in _session_datasets(s):
        records.append({
            "id": item.get("id"),
            "filename": _display_filename(item.get("filename") or item.get("filepath")),
            "rows": item.get("rows"),
            "columns": item.get("columns", []),
            "time_col": item.get("time_col", ""),
            "target_col": item.get("target_col", ""),
            "id_col": item.get("id_col", ""),
            "active": item.get("id") == active_id,
        })
    return records


def _set_active_dataset(s: Session, item: dict) -> None:
    for dataset in _session_datasets(s):
        dataset["active"] = dataset.get("id") == item.get("id")
    s.metadata["active_dataset_id"] = item.get("id")
    s.data = None
    s.data_filepath = item.get("filepath")
    s.time_col = _primary_column(item.get("time_col")) or None
    s.target_col = _column_payload(item.get("target_col")) or None
    s.id_col = item.get("id_col") or None
    s.clear_model()


def _delete_upload_file(path: str, keep_path: Optional[Path] = None) -> None:
    if not path:
        return
    upload_dir = (Path(tempfile.gettempdir()) / "pipelinets_uploads").resolve()
    try:
        p = Path(path).resolve()
        p.relative_to(upload_dir)
        if keep_path is not None and p == keep_path.resolve():
            return
        if p.exists():
            p.unlink()
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------------
#  Create Flask app
# ---------------------------------------------------------------------------

def create_app() -> "Flask":
    try:
        from flask import Flask, request, jsonify, Response, render_template
    except ImportError:
        raise ImportError("Flask is required. Install it with: pip install flask")

    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        return response

    # ── Serve frontend ────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Serve plot images ─────────────────────────
    @app.route("/api/plots/<path:filename>")
    def serve_plot(filename):
        from flask import send_file
        filepath = _app.plot_dir / filename
        if not filepath.exists() or not filepath.is_file():
            return jsonify({"error": "Plot not found"}), 404
        return send_file(str(filepath), mimetype="image/png")

    # ── Session status ────────────────────────────
    @app.route("/api/session/status")
    def session_status():
        return jsonify(_get_session_status())

    # ── Session reset ─────────────────────────────
    @app.route("/api/session/reset", methods=["POST"])
    def session_reset():
        if _app.agent is not None:
            _app.agent.reset()
        return jsonify({"ok": True})

    # ── New session ───────────────────────────────
    @app.route("/api/session/create", methods=["POST"])
    def session_create():
        _app.new_session()
        return jsonify({"ok": True, "session_id": _app.session_id})

    # ── Chat (SSE streaming) ──────────────────────
    @app.route("/api/chat/stream", methods=["POST"])
    def chat_stream():
        data = request.get_json(force=True) if request.is_json else {}
        message = (data or {}).get("message", "")
        selected_context = (data or {}).get("selected_context", "")
        selected_data_context = (data or {}).get("selected_data_context")

        if not message or not message.strip():
            return jsonify({"error": "Empty message"}), 400

        error_msg = None
        if _app.agent is None:
            try:
                _app.get_or_create_agent()
            except Exception as e:
                error_msg = f"Agent not configured: {e}"

        def generate():
            nonlocal error_msg

            if error_msg:
                yield f"data: {json.dumps({'done': True, 'error': error_msg})}\n\n"
                return

            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

            # Buffer for stream events (tool calls, reasoning) emitted during chat()
            event_buffer = queue.Queue()
            previous_on_event = _app.agent._on_stream_event

            def on_event(evt):
                event_buffer.put(evt)

            _app.agent._on_stream_event = on_event

            def flush_events():
                while True:
                    try:
                        evt = event_buffer.get_nowait()
                    except queue.Empty:
                        break
                    yield f"data: {json.dumps(evt)}\n\n"

            try:
                for chunk in _app.agent.chat(message, stream=True, context=selected_context, selected_data_context=selected_data_context):
                    yield from flush_events()
                    if chunk:
                        yield f"data: {json.dumps({'text': chunk})}\n\n"

                yield from flush_events()

                status = _get_session_status()
                status["done"] = True
                yield f"data: {json.dumps(status)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"
            finally:
                _app.agent._on_stream_event = previous_on_event

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Settings: load ────────────────────────────
    @app.route("/api/settings/load")
    def settings_load():
        alias = request.args.get("alias", "").strip()
        return jsonify(_settings_result(alias))

    # ── Settings: save ────────────────────────────
    @app.route("/api/settings/save", methods=["POST"])
    def settings_save():
        data = request.get_json(force=True) if request.is_json else {}
        if not data:
            return jsonify({"error": "No data"}), 400

        source = _config_source()
        profiles = _profiles_with_legacy(source)
        alias = (
            (data.get("api_alias") or data.get("active_api_alias") or data.get("alias") or "").strip()
            or source.get("active_api_alias")
            or "default"
        )
        existing_profile = profiles.get(alias, {})

        def _keep_if_empty(key: str, default: str = "", profile: bool = True) -> str:
            submitted = (data.get(key) or "").strip()
            if not submitted:
                stored = existing_profile.get(key, "") if profile else ""
                if not stored and not profile:
                    stored = source.get(key, "")
                existing = stored or (_app.config._data.get(key, default) if not profile else default)
                if existing:
                    return str(existing)
            return submitted

        def _preserve_secret(key: str) -> str:
            submitted_key = (data.get(key) or "").strip()
            existing_key = existing_profile.get(key, "")
            if not submitted_key:
                return existing_key
            elif existing_key:
                masked = _mask_secret(existing_key)
                if submitted_key == masked or set(submitted_key) <= {"*", "•"}:
                    return existing_key
            return submitted_key

        provider = _keep_if_empty("provider", "openai")
        api_format = _keep_if_empty("api_format", "anthropic" if provider == "anthropic" else "openai")
        profile_resolved = {
            "provider": provider,
            "api_format": api_format,
            "openai_api_key": _preserve_secret("openai_api_key"),
            "openai_base_url": _keep_if_empty("openai_base_url"),
            "openai_model": _keep_if_empty("openai_model"),
            "anthropic_api_key": _preserve_secret("anthropic_api_key"),
            "anthropic_model": _keep_if_empty("anthropic_model"),
        }
        profiles[alias] = {k: str(v) for k, v in profile_resolved.items() if v}
        profiles_json = json.dumps(profiles, ensure_ascii=False)

        resolved = {
            "provider": profile_resolved.get("provider", "openai"),
            "api_format": profile_resolved.get("api_format", "openai"),
            "openai_api_key": profile_resolved.get("openai_api_key", ""),
            "openai_base_url": profile_resolved.get("openai_base_url", ""),
            "openai_model": profile_resolved.get("openai_model", ""),
            "anthropic_api_key": profile_resolved.get("anthropic_api_key", ""),
            "anthropic_model": profile_resolved.get("anthropic_model", ""),
            "lang": _keep_if_empty("lang", "en", profile=False),
            "multimodal": _keep_if_empty("multimodal", "auto", profile=False),
            "active_api_alias": alias,
            "api_profiles": profiles_json,
        }

        if _app.storage.available:
            _app.storage.set_config_bulk({
                k: str(v) for k, v in resolved.items() if v
            })

        _app.config.update(**resolved)

        save_result = ""
        try:
            saved_path = _app.config.save()
            save_result = f"Saved to {saved_path}"
        except Exception as e:
            save_result = f"Could not save: {e}"

        agent_status = ""
        try:
            _, summary = _app.get_or_create_agent()
            agent_status = f"Connected: {summary}"
        except Exception as e:
            agent_status = f"Agent init failed: {e}"

        result = _settings_result(alias)
        result.update({
            "save_status": save_result,
            "agent_status": agent_status,
        })
        return jsonify(result)

    # ── Data: info about uploaded/stored data ──────
    @app.route("/api/data/info")
    def data_info():
        s = _app.session
        if s is None:
            return jsonify({"has_data": False, "datasets": []})

        dataset_id = request.args.get("dataset_id", "")
        dataset_item = _dataset_for_id(s, dataset_id)
        if dataset_item:
            filepath = Path(dataset_item.get("filepath", ""))
            if not filepath.exists():
                return jsonify({"has_data": False, "error": "File not found", "datasets": _serialize_datasets(s)})
            return jsonify({
                "has_data": True,
                "total_rows": dataset_item.get("rows") or 0,
                "columns": dataset_item.get("columns", []),
                "time_col": dataset_item.get("time_col", ""),
                "target_col": dataset_item.get("target_col", ""),
                "id_col": dataset_item.get("id_col", ""),
                "source": "file",
                "filename": _display_filename(dataset_item.get("filename") or filepath.name),
                "dataset_id": dataset_item.get("id"),
                "active_dataset_id": s.metadata.get("active_dataset_id"),
                "datasets": _serialize_datasets(s),
            })

        if s.has_data():
            return jsonify({
                "has_data": True,
                "total_rows": len(s.data),
                "columns": list(s.data.columns),
                "time_col": s.time_col,
                "target_col": s.target_col,
                "id_col": s.id_col,
                "source": "memory",
                "active_dataset_id": s.metadata.get("active_dataset_id"),
                "datasets": _serialize_datasets(s),
            })

        if s.has_data_file():
            active_item = _dataset_for_id(s, s.metadata.get("active_dataset_id"))
            filepath = Path(active_item.get("filepath") if active_item else s.data_filepath)
            if not filepath.exists():
                return jsonify({"has_data": False, "error": "File not found"})
            row_count = 0
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    f.readline()
                    row_count = sum(1 for _ in f)
            except Exception:
                row_count = 0
            cols = []
            try:
                df_head = pd.read_csv(filepath, nrows=0)
                cols = list(df_head.columns)
            except Exception:
                pass
            return jsonify({
                "has_data": True,
                "total_rows": active_item.get("rows") if active_item else row_count,
                "columns": active_item.get("columns", cols) if active_item else cols,
                "time_col": active_item.get("time_col") if active_item else s.time_col,
                "target_col": active_item.get("target_col") if active_item else s.target_col,
                "id_col": active_item.get("id_col") if active_item else s.id_col,
                "source": "file",
                "filename": _display_filename(active_item.get("filename", filepath.name)) if active_item else _display_filename(s.data_filepath),
                "active_dataset_id": s.metadata.get("active_dataset_id"),
                "datasets": _serialize_datasets(s),
            })

        return jsonify({"has_data": False, "datasets": _serialize_datasets(s)})

    # ── Data: fetch rows for lazy grid ─────────────
    @app.route("/api/data/rows")
    def data_rows():
        try:
            start = int(request.args.get("start", 0))
            end = int(request.args.get("end", 100))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid start/end params"}), 400

        if start < 0 or end <= start:
            return jsonify({"error": "Invalid range: start >= 0, end > start"}), 400

        s = _app.session
        if s is None:
            return jsonify({"error": "No session"}), 400

        dataset_id = request.args.get("dataset_id", "")
        dataset_item = _dataset_for_id(s, dataset_id)
        if dataset_item:
            filepath = dataset_item.get("filepath")
            if not filepath or not Path(filepath).exists():
                return jsonify({"error": "Data file not found"}), 404
            total_rows = int(dataset_item.get("rows") or 0)
            end = min(end, total_rows)
            try:
                chunk = pd.read_csv(
                    filepath,
                    skiprows=range(1, start + 1),
                    nrows=end - start,
                )
            except Exception as e:
                return jsonify({"error": f"Error reading file: {e}"}), 400
            return jsonify({
                "columns": list(chunk.columns),
                "rows": chunk.values.tolist(),
                "start": start,
                "end": end,
                "total_rows": total_rows,
                "dataset_id": dataset_item.get("id"),
            })

        if s.has_data():
            df = s.data
            total_rows = len(df)
            end = min(end, total_rows)
            chunk = df.iloc[start:end]
            return jsonify({
                "columns": list(df.columns),
                "rows": chunk.values.tolist(),
                "start": start,
                "end": end,
                "total_rows": total_rows,
            })

        if s.has_data_file():
            filepath = s.data_filepath
            if not Path(filepath).exists():
                return jsonify({"error": "Data file not found"}), 404
            try:
                total_rows = 0
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    f.readline()
                    total_rows = sum(1 for _ in f)
            except Exception:
                total_rows = 0
            end = min(end, total_rows)
            try:
                chunk = pd.read_csv(
                    filepath,
                    skiprows=range(1, start + 1),
                    nrows=end - start,
                )
            except Exception as e:
                return jsonify({"error": f"Error reading file: {e}"}), 400
            return jsonify({
                "columns": list(chunk.columns),
                "rows": chunk.values.tolist(),
                "start": start,
                "end": end,
                "total_rows": total_rows,
            })

        return jsonify({"error": "No data available"}), 400

    # ── Data: upload CSV ──────────────────────────
    @app.route("/api/data/upload", methods=["POST"])
    def data_upload():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        s = _app.ensure_session()
        multi_upload = request.form.get("replace") != "1"

        # Save uploaded file to local disk for later load_csv
        upload_dir = Path(tempfile.gettempdir()) / "pipelinets_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
        local_path = upload_dir / safe_filename

        try:
            file.save(str(local_path))
        except Exception as e:
            return jsonify({"error": f"Error saving file: {e}"}), 500

        # Clean up previous uploaded file
        if not multi_upload:
            for item in list(_session_datasets(s)):
                _delete_upload_file(item.get("filepath"), keep_path=local_path)
            _delete_upload_file(s.data_filepath, keep_path=local_path)
            s.metadata["datasets"] = []

        # Count total row count without loading full file
        try:
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                f.readline()  # skip header
                row_count = sum(1 for _ in f)
        except Exception:
            row_count = None

        # Read only first 20 rows for preview
        try:
            df_preview = pd.read_csv(local_path, nrows=20)
        except Exception as e:
            try:
                os.remove(str(local_path))
            except OSError:
                pass
            return jsonify({"error": f"Error reading file: {e}"}), 400

        # Auto-detect columns from preview
        time_col = ""
        target_col = ""
        for col in df_preview.columns:
            if pd.api.types.is_datetime64_any_dtype(df_preview[col]):
                time_col = col
            elif col.lower() in ("value", "target", "y", "sales", "demand", "count", "price", "volume"):
                if not target_col:
                    target_col = col

        if not time_col:
            try:
                col0 = df_preview.columns[0]
                pd.to_datetime(df_preview[col0])
                time_col = col0
            except Exception:
                pass

        if not target_col:
            for col in reversed(df_preview.columns):
                if pd.api.types.is_numeric_dtype(df_preview[col]) and col != time_col:
                    target_col = col
                    break

        dataset_id = uuid.uuid4().hex
        display_filename = _display_filename(file.filename)
        dataset_item = {
            "id": dataset_id,
            "filename": display_filename,
            "filepath": str(local_path),
            "rows": row_count,
            "columns": list(df_preview.columns),
            "time_col": time_col or "",
            "target_col": target_col or "",
            "id_col": "",
        }
        _session_datasets(s).append(dataset_item)
        _set_active_dataset(s, dataset_item)

        preview_rows = df_preview.head(5)
        preview_html = preview_rows.to_html(classes="preview-table", index=False)
        return jsonify({
            "preview": preview_html,
            "rows": row_count,
            "preview_rows": len(preview_rows),
            "columns": len(df_preview.columns),
            "column_names": list(df_preview.columns),
            "time_col": time_col or "",
            "target_col": target_col or "",
            "filename": display_filename,
            "filepath": str(local_path),
            "dataset_id": dataset_id,
            "datasets": _serialize_datasets(s),
        })

    # ── Data: apply column mapping ────────────────
    @app.route("/api/data/apply-columns", methods=["POST"])
    def data_apply_columns():
        s = _app.ensure_session()

        data = request.get_json(force=True) if request.is_json else {}
        time_col = _column_payload((data or {}).get("time_col", ""))
        target_col = _column_payload((data or {}).get("target_col", ""))
        id_col = (data or {}).get("id_col", "")
        dataset_id = (data or {}).get("dataset_id", "")
        dataset_item = _dataset_for_id(s, dataset_id)

        if s.data is None and not s.has_data_file() and dataset_item is None:
            return jsonify({"error": "No data loaded. Upload a CSV first."}), 400

        if dataset_item is not None:
            dataset_item["time_col"] = time_col or dataset_item.get("time_col", "")
            dataset_item["target_col"] = target_col or dataset_item.get("target_col", "")
            dataset_item["id_col"] = id_col or ""
            _set_active_dataset(s, dataset_item)
            msg = f"Columns set: time='{_format_column_payload(time_col)}', target='{_format_column_payload(target_col)}'"
            if id_col:
                msg += f", id='{id_col}'"
            return jsonify({"ok": True, "message": msg, "datasets": _serialize_datasets(s), "active_dataset_id": dataset_item.get("id")})

        if s.has_data():
            df = s.data
            primary_time_col = _primary_column(time_col)
            if primary_time_col and primary_time_col in df.columns:
                try:
                    df[primary_time_col] = pd.to_datetime(df[primary_time_col])
                except Exception:
                    pass
                df = df.sort_values(primary_time_col).reset_index(drop=True)
                s.data = df
                s.time_col = primary_time_col
        else:
            if time_col:
                s.time_col = _primary_column(time_col)

        if target_col:
            s.target_col = target_col
        s.id_col = id_col or None

        s.clear_model()
        msg = f"Columns set: time='{_format_column_payload(time_col)}', target='{_format_column_payload(target_col)}'"
        if id_col:
            msg += f", id='{id_col}'"

        return jsonify({"ok": True, "message": msg})

    @app.route("/api/data/undo-upload", methods=["POST"])
    def data_undo_upload():
        s = _app.ensure_session()
        data = request.get_json(silent=True) if request.is_json else {}
        target_id = (data or {}).get("dataset_id") or s.metadata.get("active_dataset_id")
        datasets = _session_datasets(s)
        target_item = _dataset_for_id(s, target_id)
        is_web_upload = False

        if target_item is not None:
            _delete_upload_file(target_item.get("filepath"))
            datasets[:] = [item for item in datasets if item.get("id") != target_item.get("id")]
            is_web_upload = True
        elif s.data_filepath:
            before = Path(s.data_filepath)
            _delete_upload_file(s.data_filepath)
            is_web_upload = not before.exists()

        if is_web_upload:
            if datasets:
                _set_active_dataset(s, datasets[-1])
            else:
                s.metadata["active_dataset_id"] = None
                s.data = None
                s.data_filepath = None
                s.time_col = None
                s.target_col = None
                s.id_col = None
                s.feature_cols = []
                s.known_covariates = []
                s.past_covariates = []
                s.clear_model()

        return jsonify({
            "ok": True,
            "undone": is_web_upload,
            "message": "Upload undone." if is_web_upload else "No uploaded data to undo.",
            "datasets": _serialize_datasets(s),
            "active_dataset_id": s.metadata.get("active_dataset_id"),
        })

    # ── History: list ─────────────────────────────
    @app.route("/api/history/list")
    def history_list():
        sessions = []
        if _app.storage.available:
            sessions = _app.storage.list_sessions()
        return jsonify({"sessions": sessions, "current_id": _app.session_id})

    # ── History: get ──────────────────────────────
    @app.route("/api/history/<session_id>")
    def history_get(session_id):
        if not _app.storage.available:
            return jsonify({"error": "Storage not available"}), 404
        s = _app.storage.get_session(session_id)
        if s is None:
            return jsonify({"error": "Session not found"}), 404
        # Restore messages into current agent session
        if _app.agent is not None:
            _app.agent.reset()
            _app.agent.session.messages = s.get("messages", [])
            _app.session = _app.agent.session
        _app._session_id = session_id
        _app._session_name = s.get("name", "Restored")
        return jsonify(s)

    # ── History: save ─────────────────────────────
    @app.route("/api/history/save", methods=["POST"])
    def history_save():
        if not _app.storage.available:
            return jsonify({"error": "Storage not available"}), 503
        if _app.agent is None:
            return jsonify({"error": "No active session"}), 400
        data = request.get_json(force=True) if request.is_json else {}
        name = (data or {}).get("name", _app._session_name)
        _app.storage.save_session(
            _app.session_id,
            name,
            _app.agent.session.messages,
        )
        _app._session_name = name
        return jsonify({"ok": True, "session_id": _app.session_id})

    # ── History: delete ───────────────────────────
    @app.route("/api/history/<session_id>", methods=["DELETE"])
    def history_delete(session_id):
        if not _app.storage.available:
            return jsonify({"error": "Storage not available"}), 503
        _app.storage.delete_session(session_id)
        return jsonify({"ok": True})

    return app


# ---------------------------------------------------------------------------
#  Launch helper
# ---------------------------------------------------------------------------

def launch_web(host: str = "127.0.0.1", port: int = 7860, debug: bool = False, **kwargs):
    app = create_app()
    print(f"\n  PipelineTS Agent starting at http://{host}:{port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(host=host, port=port, debug=debug, **kwargs)


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PipelineTS Agent — Flask Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    launch_web(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

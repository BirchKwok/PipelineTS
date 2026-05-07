"""Configuration management for the PipelineTS agent.

Loads and saves API keys, provider settings, and model preferences
from a TOML config file.  Search order:

1. ``PIPELINETS_CONFIG`` environment variable (explicit path)
2. ``./.pipelints.toml`` (project-local)
3. ``~/.pipelints/config.toml`` (user-global)

Merges with environment variables, where env vars take priority.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

# Python 3.11+ has tomllib in stdlib; fall back to tomli
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore[assignment]

DEFAULT_CONFIG_NAME = ".pipelinets.toml"
DEFAULT_USER_DIR = Path.home() / ".pipelinets"
DEFAULT_USER_CONFIG = DEFAULT_USER_DIR / "config.toml"

# Sensitive keys that should be saved with a warning
_SENSITIVE_KEYS = {"api_key", "openai_api_key", "anthropic_api_key"}


class Config:
    """PipelineTS agent configuration.

    Parameters
    ----------
    config_path : str or Path or None
        Explicit path to a TOML config file.
        If None, auto-discovers using the search order above.
    """

    def __init__(self, config_path: str | Path = None):
        self._data: dict[str, Any] = {}
        self._path: Optional[Path] = None

        # Determine config path
        if config_path:
            self._path = Path(config_path)
        elif os.environ.get("PIPELINETS_CONFIG"):
            self._path = Path(os.environ["PIPELINETS_CONFIG"])
        else:
            local = Path.cwd() / DEFAULT_CONFIG_NAME
            if local.exists():
                self._path = local
            elif DEFAULT_USER_CONFIG.exists():
                self._path = DEFAULT_USER_CONFIG

        if self._path:
            self._load(self._path)

        # Apply environment variable overrides
        self._apply_env_overrides()

    # ------------------------------------------------------------------
    #  Loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Load and merge a TOML config file."""
        if tomllib is None:
            return

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        # Flatten [agent] section into top-level for convenience
        if "agent" in data and isinstance(data["agent"], dict):
            for k, v in data["agent"].items():
                self._data.setdefault(k, v)

        for k, v in data.items():
            if k != "agent":
                self._data.setdefault(k, v)

    def _apply_env_overrides(self) -> None:
        """Environment variables take highest priority."""
        env_map = {
            "provider": "PIPELINETS_PROVIDER",
            "openai_api_key": "OPENAI_API_KEY",
            "openai_base_url": "OPENAI_BASE_URL",
            "openai_model": "OPENAI_MODEL",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "anthropic_model": "ANTHROPIC_MODEL",
            "lang": "PIPELINETS_LANG",
        }
        for key, env_var in env_map.items():
            val = os.environ.get(env_var)
            if val:
                self._data[key] = val

    # ------------------------------------------------------------------
    #  Accessors
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return self._data.get("provider", "auto")

    @provider.setter
    def provider(self, value: str):
        self._data["provider"] = value

    @property
    def openai_api_key(self) -> str:
        return self._data.get("openai_api_key", "")

    @openai_api_key.setter
    def openai_api_key(self, value: str):
        self._data["openai_api_key"] = value

    @property
    def openai_base_url(self) -> str:
        return self._data.get("openai_base_url", "")

    @openai_base_url.setter
    def openai_base_url(self, value: str):
        self._data["openai_base_url"] = value

    @property
    def openai_model(self) -> str:
        return self._data.get("openai_model", "")

    @openai_model.setter
    def openai_model(self, value: str):
        self._data["openai_model"] = value

    @property
    def anthropic_api_key(self) -> str:
        return self._data.get("anthropic_api_key", "")

    @anthropic_api_key.setter
    def anthropic_api_key(self, value: str):
        self._data["anthropic_api_key"] = value

    @property
    def anthropic_model(self) -> str:
        return self._data.get("anthropic_model", "")

    @anthropic_model.setter
    def anthropic_model(self, value: str):
        self._data["anthropic_model"] = value

    @property
    def lang(self) -> str:
        return self._data.get("lang", "en")

    @lang.setter
    def lang(self, value: str):
        self._data["lang"] = value

    # ------------------------------------------------------------------
    #  Resolution — decides which provider + credentials to use
    # ------------------------------------------------------------------

    def resolve_provider(self) -> str:
        """Determine the effective provider: 'openai' or 'anthropic'.

        If provider is 'auto', prefers the one with an API key set.
        """
        if self.provider == "anthropic":
            return "anthropic"
        if self.provider == "openai":
            return "openai"
        # respect explicit non-standard providers (deepseek, ollama, etc.)
        if self.provider != "auto":
            return self.provider
        # auto
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "auto"  # no provider configured

    def resolve_api_key(self) -> str:
        """Return the effective API key for the resolved provider."""
        if self.resolve_provider() == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key

    def resolve_base_url(self) -> str:
        """Return the base URL (OpenAI only)."""
        return self.openai_base_url or ""

    def resolve_model(self) -> str:
        """Return the effective model for the resolved provider."""
        if self.resolve_provider() == "anthropic":
            return self.anthropic_model
        return self.openai_model

    # ------------------------------------------------------------------
    #  Saving
    # ------------------------------------------------------------------

    def save(self, path: str | Path = None) -> Path:
        """Persist configuration to a TOML file.

        Parameters
        ----------
        path : str or Path or None
            Target path. If None, saves to the path used during loading
            or to the user-global default.

        Returns
        -------
        Path
            The file path written to.
        """
        if path:
            target = Path(path)
        else:
            target = self._path or DEFAULT_USER_CONFIG

        target.parent.mkdir(parents=True, exist_ok=True)

        # Build TOML-safe dict
        data: dict[str, Any] = {}
        for k, v in sorted(self._data.items()):
            if v is not None and v != "":
                data[k] = v

        # Warn about saving API keys
        has_sensitive = any(k in _SENSITIVE_KEYS for k in data)
        if has_sensitive:
            import warnings

            warnings.warn(
                f"Config file at {target} contains API keys. "
                f"Ensure the file has restricted permissions (e.g., chmod 600).",
                UserWarning,
            )

        # Write TOML
        with open(target, "w", encoding="utf-8") as f:
            f.write("# PipelineTS Agent Configuration\n")
            f.write("# Sensitive: API keys stored here. chmod 600 recommended.\n\n")
            for k, v in data.items():
                f.write(f"{k} = {_toml_value(v)}\n")

        self._path = target
        return target

    # ------------------------------------------------------------------
    #  Dict interface
    # ------------------------------------------------------------------

    def to_dict(self, hide_secrets: bool = True) -> dict[str, Any]:
        """Return a copy of the config as a dict.

        Parameters
        ----------
        hide_secrets : bool
            If True (default), masks API keys.
        """
        d = dict(self._data)
        if hide_secrets:
            for k in _SENSITIVE_KEYS:
                if k in d and d[k]:
                    d[k] = _mask(d[k])
        return d

    def update(self, **kwargs) -> None:
        """Bulk-update configuration values."""
        for k, v in kwargs.items():
            if hasattr(self, k) or k in _SENSITIVE_KEYS | {
                "provider", "openai_api_key", "openai_base_url", "openai_model",
                "anthropic_api_key", "anthropic_model", "lang",
            }:
                self._data[k] = v

    def __repr__(self) -> str:
        d = self.to_dict(hide_secrets=True)
        return f"Config({d})"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _mask(value: str, keep: int = 4) -> str:
    """Mask a secret value, showing only the first and last few chars."""
    if len(value) <= keep * 2:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]


def _toml_value(value: Any) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Escape backslashes and quotes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return f'"{value}"'

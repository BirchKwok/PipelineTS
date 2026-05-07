"""LLM provider abstraction for the PipelineTS agent.

Supports:
- OpenAI API (and any OpenAI-compatible endpoint, e.g., Ollama, vLLM)
- Anthropic API (optional — requires `pip install anthropic`)

Provider selection is automatic based on environment variables:
    OPENAI_API_KEY     -> OpenAI
    ANTHROPIC_API_KEY  -> Anthropic
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
#  Abstract base
# ---------------------------------------------------------------------------


class LLMProvider:
    """Minimal interface for LLM backends."""

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        *,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat completion request. Returns the API response dict."""
        raise NotImplementedError

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        *,
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[dict]:
        """Stream chat completion chunks. Yields delta dicts."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  OpenAI provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """OpenAI / OpenAI-compatible chat completion API.

    Set OPENAI_API_KEY and optionally OPENAI_BASE_URL, OPENAI_MODEL.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it to use OpenAI, or set ANTHROPIC_API_KEY for Anthropic."
            )

        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("OPENAI_MODEL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages,
        tools=None,
        *,
        model=None,
        temperature=0.2,
        max_tokens=4096,
    ):
        kwargs = dict(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return response.model_dump()

    def chat_stream(
        self,
        messages,
        tools=None,
        *,
        model=None,
        temperature=0.2,
        max_tokens=4096,
    ):
        kwargs = dict(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            yield chunk.model_dump()


# ---------------------------------------------------------------------------
#  Anthropic provider
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API.

    Set ANTHROPIC_API_KEY and optionally ANTHROPIC_MODEL.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        self.model = model or os.environ.get("ANTHROPIC_MODEL")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []
        for t in tools:
            func = t["function"]
            anthropic_tools.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"],
            })
        return anthropic_tools

    def _convert_response(self, response) -> dict:
        """Convert Anthropic response to an OpenAI-compatible dict for uniform handling."""
        content = response.content
        message = {"role": "assistant", "content": None, "tool_calls": []}

        for block in content:
            if block.type == "text":
                message["content"] = block.text
            elif block.type == "tool_use":
                message["tool_calls"].append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        if message["content"] is None:
            message["content"] = ""
        return {
            "choices": [{"message": message}],
            "stop_reason": response.stop_reason,
        }

    def chat(
        self,
        messages,
        tools=None,
        *,
        model=None,
        temperature=0.2,
        max_tokens=4096,
    ):
        system = None
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append(m)

        kwargs = dict(
            model=model or self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self.client.messages.create(**kwargs)
        return self._convert_response(response)

    def chat_stream(
        self,
        messages,
        tools=None,
        *,
        model=None,
        temperature=0.2,
        max_tokens=4096,
    ):
        system = None
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append(m)

        kwargs = dict(
            model=model or self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                yield event


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------


def create_provider(
    provider: str = "auto",
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    *,
    config: Any = None,
) -> LLMProvider:
    """Create an LLM provider instance.

    Parameters
    ----------
    provider : str
        One of 'auto', 'openai', 'anthropic'.
        'auto' detects based on environment variables or config.
    api_key : str or None
        API key override (takes priority over env vars and config).
    base_url : str or None
        Base URL override (OpenAI only).
    model : str or None
        Model name override.
    config : Config or None
        Optional PipelineTS Config object. If provided, reads settings
        from it as fallback when not overridden by params/env.

    Returns
    -------
    LLMProvider
    """
    # Resolve config if provided
    cfg_provider = "auto"
    cfg_api_key = None
    cfg_base_url = None
    cfg_model = None
    if config is not None:
        cfg_provider = config.resolve_provider()
        cfg_api_key = config.resolve_api_key() or None
        cfg_base_url = config.resolve_base_url() or None
        cfg_model = config.resolve_model() or None

    # Determine provider
    if provider != "auto":
        effective_provider = provider
    elif os.environ.get("ANTHROPIC_API_KEY"):
        effective_provider = "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        effective_provider = "openai"
    elif cfg_provider != "auto":
        effective_provider = cfg_provider
    elif cfg_api_key:
        effective_provider = cfg_provider  # 'auto' resolved to openai/anthropic
    else:
        raise ValueError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "environment variable, or provide a PipelineTS config file "
            "(~/.pipelinets/config.toml) with API keys.\n\n"
            "For local models via Ollama:\n"
            "  export OPENAI_API_KEY=ollama\n"
            "  export OPENAI_BASE_URL=http://localhost:11434/v1\n"
            "  export OPENAI_MODEL=qwen2.5"
        )

    # Resolve API key with fallback chain: param > env > config
    if effective_provider == "openai":
        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or cfg_api_key
        )
        resolved_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or cfg_base_url
        )
        resolved_model = (
            model
            or os.environ.get("OPENAI_MODEL")
            or cfg_model
        )
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not set. Provide it via parameter, "
                "OPENAI_API_KEY env var, or config file."
            )
        return OpenAIProvider(
            api_key=resolved_key,
            base_url=resolved_url or None,
            model=resolved_model,
        )

    elif effective_provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic\n"
                "Or switch to OpenAI: set provider='openai'"
            )
        resolved_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or cfg_api_key
        )
        resolved_model = (
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or cfg_model
        )
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not set. Provide it via parameter, "
                "ANTHROPIC_API_KEY env var, or config file."
            )
        return AnthropicProvider(
            api_key=resolved_key,
            model=resolved_model,
        )
    else:
        # OpenAI-compatible third-party providers (deepseek, ollama, etc.)
        resolved_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or cfg_api_key
        )
        resolved_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or cfg_base_url
        )
        resolved_model = (
            model
            or os.environ.get("OPENAI_MODEL")
            or cfg_model
        )
        if not resolved_key:
            raise ValueError(
                f"API key not set for provider '{effective_provider}'. "
                "Provide it via parameter, OPENAI_API_KEY env var, or config file."
            )
        return OpenAIProvider(
            api_key=resolved_key,
            base_url=resolved_url or None,
            model=resolved_model,
        )

"""Core agent loop for the PipelineTS natural language interface.

The TSAgent orchestrates the conversation: receives user messages, sends them
to the LLM with tools, executes tool calls, and returns results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from PipelineTS.agent.session import Session
from PipelineTS.agent.executor import Executor
from PipelineTS.agent.llm import LLMProvider, create_provider
from PipelineTS.agent.tools import ALL_TOOLS
from PipelineTS.agent.prompts import get_system_prompt


class TSAgent:
    """Conversational time series analysis agent.

    Parameters
    ----------
    provider : str or LLMProvider
        One of 'auto', 'openai', 'anthropic', or a pre-built LLMProvider.
    model : str or None
        Override the default model name.
    api_key : str or None
        Override API key.
    base_url : str or None
        Override base URL (OpenAI-compatible APIs).
    lang : str
        Language for system prompts: 'en' or 'zh'.
    verbose : bool
        If True, print tool call details to stdout during chat.

    Examples
    --------
    >>> from PipelineTS.agent import TSAgent
    >>> agent = TSAgent()
    >>> response = agent.chat("Load the electric dataset and inspect it")
    >>> print(response)
    """

    def __init__(
        self,
        provider: str | LLMProvider = "auto",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        lang: str = "en",
        verbose: bool = False,
        plot_dir: str = None,
    ):
        if isinstance(provider, LLMProvider):
            self.llm = provider
        else:
            self.llm = create_provider(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

        self.session = Session()
        self.executor = Executor(self.session, plot_dir=Path(plot_dir) if plot_dir else None)
        self.tools = ALL_TOOLS
        self.lang = lang
        self.verbose = verbose
        self._provider = provider if isinstance(provider, str) else getattr(provider, 'model', 'unknown')
        self._model = model or getattr(self.llm, 'model', 'unknown')
        self._base_url = base_url or getattr(self.llm, 'client', None)
        self._base_prompt = get_system_prompt(lang)
        self._transient_user_context: Optional[str] = None

        # Plot result marker pattern: [PLOT]label saved.{filename}[/PLOT]
        self._plot_pattern = re.compile(r'\[PLOT\](.+?)\.\{(.+?)\}\[/PLOT\]')

        # Callbacks
        self._on_tool_call: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_stream_event: Optional[Callable] = None  # for SSE UI events

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def chat(self, message: str, *, stream: bool = False, context: str = None) -> str:
        """Send a message and return the agent's response.

        Parameters
        ----------
        message : str
            User's natural language message.
        stream : bool
            If True, yield response chunks as they arrive (generator).

        Returns
        -------
        str (or Iterator[str] if stream=True)
        """
        self.session.messages.append({"role": "user", "content": message})

        previous_context = self._transient_user_context
        self._transient_user_context = context.strip() if context and context.strip() else None

        if stream:
            return self._chat_stream_with_context(previous_context)
        else:
            try:
                return self._chat_sync()
            finally:
                self._transient_user_context = previous_context

    def _chat_stream_with_context(self, previous_context: Optional[str]) -> Iterator[str]:
        try:
            yield from self._chat_stream()
        finally:
            self._transient_user_context = previous_context

    def _chat_sync(self) -> str:
        """Run the agent loop synchronously, returning the final text."""
        messages = self._build_messages()

        # ── 1. Send to LLM ─────────────────────────────
        response = self.llm.chat(messages=messages, tools=self.tools)
        choice = response["choices"][0]
        msg = choice["message"]

        # ── 2. Handle tool calls ───────────────────────
        while msg.get("tool_calls"):
            # Record assistant message preserving ALL fields (eg. DeepSeek reasoning_content)
            assistant_msg = dict(msg)
            assistant_msg["role"] = "assistant"
            assistant_msg["content"] = assistant_msg.get("content") or ""
            self.session.messages.append(assistant_msg)

            # Execute each tool call
            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                if self.verbose:
                    print(f"\n[T] {tool_name}({arguments})")

                result = self.executor.dispatch(tool_name, arguments)

                if self.verbose and result:
                    preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"[T] → {preview}")

                if self._on_tool_call:
                    self._on_tool_call(tool_name, arguments, result)

                # Strip plot marker from LLM-visible content
                llm_result = self._plot_pattern.sub(r'\1.', result)

                # Append tool result
                self.session.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": llm_result,
                })

            # ── 3. Send tool results back to LLM ────────
            messages = self._build_messages()
            response = self.llm.chat(messages=messages, tools=self.tools)
            choice = response["choices"][0]
            msg = choice["message"]

        # ── 4. Record and return final response ────────
        content = msg.get("content") or ""
        final_msg = {"role": "assistant", "content": content}
        # Preserve extra fields (eg. reasoning_content) returned by the provider
        for key in msg:
            if key not in ("role", "content") and key in msg:
                final_msg[key] = msg[key]
        self.session.messages.append(final_msg)

        if self._on_response:
            self._on_response(content)

        return content

    def _chat_stream(self) -> Iterator[str]:
        """Run the agent loop with streaming output."""
        messages = self._build_messages()

        # Accumulators for streaming mode
        full_content = ""
        full_reasoning = ""
        full_tool_calls: list[dict] = []
        extra_delta_keys: dict[str, str] = {}

        for chunk in self.llm.chat_stream(messages=messages, tools=self.tools):
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta", {})

            # Text content
            if delta.get("content"):
                full_content += delta["content"]
                yield delta["content"]

            # Reasoning content (DeepSeek thinking mode)
            if delta.get("reasoning_content"):
                rchunk = delta["reasoning_content"]
                full_reasoning += rchunk
                if self._on_stream_event:
                    self._on_stream_event({"type": "reasoning", "text": rchunk})

            # Collect any other provider-specific extra fields
            for key in delta:
                if key not in ("role", "content", "reasoning_content", "tool_calls", "function_call"):
                    if isinstance(delta[key], str):
                        extra_delta_keys.setdefault(key, "")
                        extra_delta_keys[key] += delta[key]

            # Tool calls
            if delta.get("tool_calls"):
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta.get("index", 0)
                    while len(full_tool_calls) <= idx:
                        full_tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })

                    if tc_delta.get("id"):
                        full_tool_calls[idx]["id"] = tc_delta["id"]
                    if tc_delta.get("function", {}).get("name"):
                        full_tool_calls[idx]["function"]["name"] = tc_delta["function"]["name"]
                    if tc_delta.get("function", {}).get("arguments"):
                        full_tool_calls[idx]["function"]["arguments"] += tc_delta["function"]["arguments"]

        # Build assistant message preserving extra fields
        def _build_assistant_msg(content, tool_calls, reasoning, extras):
            msg = {"role": "assistant", "content": content or ""}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if reasoning:
                msg["reasoning_content"] = reasoning
            for k, v in extras.items():
                if v:
                    msg[k] = v
            return msg

        # If there are tool calls, handle them and recurse
        if full_tool_calls:
            self.session.messages.append(
                _build_assistant_msg(full_content, full_tool_calls, full_reasoning, extra_delta_keys)
            )

            for tc in full_tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                # Emit tool_call event for UI
                if self._on_stream_event:
                    self._on_stream_event({
                        "type": "tool_call",
                        "name": tool_name,
                        "arguments": arguments,
                    })

                result = self.executor.dispatch(tool_name, arguments)

                # Check for plot image result
                plot_match = self._plot_pattern.search(result)
                plot_filename = plot_match.group(2) if plot_match else None
                # Strip plot marker from LLM-visible content
                # Strip plot marker from LLM-visible content in _chat_stream
                llm_result = self._plot_pattern.sub(r'\1.', result)

                # Emit tool_result event for UI
                if self._on_stream_event:
                    self._on_stream_event({
                        "type": "tool_result",
                        "name": tool_name,
                        "result": llm_result[:200] + "..." if len(llm_result) > 200 else llm_result,
                    })
                    if plot_filename:
                        self._on_stream_event({
                            "type": "plot_image",
                            "name": tool_name,
                            "filename": plot_filename,
                            "url": f"/api/plots/{plot_filename}",
                        })

                self.session.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": llm_result,
                })

            # Recurse for LLM's final response
            yield from self._chat_stream()
        elif full_content:
            self.session.messages.append(
                _build_assistant_msg(full_content, [], full_reasoning, extra_delta_keys)
            )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """Build the full message list including dynamic system prompt."""
        return [
            {"role": "system", "content": self._build_system_prompt()},
        ] + self.session.messages

    def _build_system_prompt(self) -> str:
        """Build a dynamic system prompt with full harness context."""
        import platform
        import sys
        import pandas as pd

        # ── Environment ──
        env_lines = []
        env_lines.append(f"Python: {sys.version.split()[0]}")
        env_lines.append(f"OS: {platform.system()} {platform.release()}")
        # Key dependencies
        for pkg in ("pandas", "numpy", "scikit-learn", "matplotlib", "torch", "statsmodels"):
            try:
                m = __import__(pkg)
                v = getattr(m, "__version__", "?")
            except Exception:
                v = "not installed"
            env_lines.append(f"{pkg}: {v}")

        # ── Model info ──
        model_lines = []
        model_lines.append(f"Provider: {self._provider}")
        model_lines.append(f"Model: {self._model}")
        if self._base_url:
            bu = str(self._base_url)
            if not bu.startswith("<"):
                model_lines.append(f"Base URL: {bu}")

        # ── Session state ──
        session_lines = []
        if self.session.has_data():
            df = self.session.data
            session_lines.append(f"Data loaded: {len(df)} rows × {len(df.columns)} cols")
            session_lines.append(f"Columns: {list(df.columns)}")
            session_lines.append(f"Time column: '{self.session.time_col}'")
            session_lines.append(f"Target column: '{self.session.target_col}'")
            if self.session.id_col:
                session_lines.append(f"ID column: '{self.session.id_col}'")
            # Stats
            target_cols = self.session.target_col if isinstance(self.session.target_col, list) else ([self.session.target_col] if self.session.target_col else [])
            for tc in target_cols:
                if not tc or tc not in df.columns:
                    continue
                col = df[tc]
                if pd.api.types.is_numeric_dtype(col):
                    session_lines.append(
                        f"Target '{tc}' stats: mean={col.mean():.4f}, std={col.std():.4f}, "
                        f"min={col.min():.4f}, max={col.max():.4f}, missing={col.isna().sum()}"
                    )
            time_col = self.session.time_col[0] if isinstance(self.session.time_col, list) and self.session.time_col else self.session.time_col
            if time_col and time_col in df.columns:
                t = df[time_col]
                try:
                    session_lines.append(f"Time range: {t.min()} → {t.max()}")
                except Exception:
                    pass
            datasets = self.session.metadata.get("datasets", [])
            if isinstance(datasets, list) and len(datasets) > 1:
                session_lines.append("Uploaded datasets available for comparison:")
                for item in datasets:
                    active = " (active)" if item.get("id") == self.session.metadata.get("active_dataset_id") else ""
                    session_lines.append(
                        f"  - {item.get('filename')}{active}: path={item.get('filepath')}, "
                        f"time_col='{item.get('time_col')}', target_col='{item.get('target_col')}'"
                    )
        elif self.session.has_data_file():
            session_lines.append("Data file uploaded via web UI — NOT yet loaded into memory.")
            session_lines.append(f"File path: {self.session.data_filepath}")
            session_lines.append(f"Time column: '{self.session.time_col}'")
            session_lines.append(f"Target column: '{self.session.target_col}'")
            if self.session.id_col:
                session_lines.append(f"ID column: '{self.session.id_col}'")
            datasets = self.session.metadata.get("datasets", [])
            if isinstance(datasets, list) and len(datasets) > 1:
                session_lines.append("Uploaded datasets available for comparison:")
                for item in datasets:
                    active = " (active)" if item.get("id") == self.session.metadata.get("active_dataset_id") else ""
                    session_lines.append(
                        f"  - {item.get('filename')}{active}: path={item.get('filepath')}, "
                        f"time_col='{item.get('time_col')}', target_col='{item.get('target_col')}'"
                    )
            session_lines.append("IMPORTANT: You MUST call load_csv with the filepath above to load the full dataset before performing any analysis or training.")
        else:
            session_lines.append("No data loaded.")

        if self.session.has_model():
            session_lines.append(f"Trained model: {self.session.model_type}")
            if self.session.best_model_name:
                session_lines.append(f"Best model: {self.session.best_model_name}")

        # ── Available tools ──
        tool_names = sorted([t["function"]["name"] for t in self.tools])
        tool_list = ", ".join(tool_names)

        # ── Assemble ──
        parts = []
        parts.append(self._base_prompt)
        parts.append("\n## Environment")
        parts.append("\n".join(f"- {l}" for l in env_lines))
        parts.append("\n## Current Model")
        parts.append("\n".join(f"- {l}" for l in model_lines))
        parts.append("\n## Session State")
        parts.append("\n".join(f"- {l}" for l in session_lines))
        if self._transient_user_context:
            parts.append("\n## User-selected data context")
            parts.append("The user selected the following cells in the web data preview. Use this only when the current user message is asking about the selected cells or refers to them implicitly; otherwise ignore it.")
            parts.append(self._transient_user_context)
        parts.append(f"\n## Available Tools ({len(self.tools)})\n{tool_list}")

        return "\n".join(parts)

    def reset(self) -> None:
        """Reset the conversation and session state."""
        old_plot_dir = self.executor.plot_dir
        self._transient_user_context = None
        self.session = Session()
        self.executor = Executor(self.session, plot_dir=old_plot_dir)

    def on_tool_call(self, callback: Callable) -> Callable:
        """Decorator/register a callback invoked after each tool execution.

        Signature: callback(tool_name: str, arguments: dict, result: str) -> None
        """
        self._on_tool_call = callback
        return callback

    def on_response(self, callback: Callable) -> Callable:
        """Decorator/register a callback invoked on each final agent response.

        Signature: callback(response: str) -> None
        """
        self._on_response = callback
        return callback

    # ------------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Return the current session status."""
        return self.session.status_summary()

    def save_session(self, filepath: str) -> None:
        """Save the conversation history to a JSON file."""
        import json as _json
        with open(filepath, "w", encoding="utf-8") as f:
            _json.dump(self.session.messages, f, indent=2, ensure_ascii=False)

    def load_session(self, filepath: str) -> None:
        """Load conversation history from a JSON file."""
        import json as _json
        with open(filepath, "r", encoding="utf-8") as f:
            self.session.messages = _json.load(f)

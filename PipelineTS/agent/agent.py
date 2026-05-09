"""Core agent loop for the PipelineTS natural language interface.

The TSAgent orchestrates the conversation: receives user messages, sends them
to the LLM with tools, executes tool calls, and returns results.
"""

from __future__ import annotations

import json
import base64
import mimetypes
import queue
import re
import threading
import time
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
        multimodal: bool | str = "auto",
        max_visual_context_images: int = 4,
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
        self._transient_selected_data_context: Optional[dict] = None
        self.multimodal = self._normalize_multimodal(multimodal)
        self.multimodal_enabled = self._resolve_multimodal_enabled()
        self.max_visual_context_images = max(0, int(max_visual_context_images or 0))

        # Plot result marker pattern: [PLOT]label saved.{filename}[/PLOT]
        self._plot_pattern = re.compile(r'\[PLOT\](.+?)\.\{(.+?)\}\[/PLOT\]')

        # Callbacks
        self._on_tool_call: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_stream_event: Optional[Callable] = None  # for SSE UI events
        self._stream_event_seq = 0

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def chat(self, message: str, *, stream: bool = False, context: str = None, selected_data_context: dict = None) -> str:
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
        previous_selected_data_context = self._transient_selected_data_context
        confirmed_context = isinstance(selected_data_context, dict) and selected_data_context.get("confirmed") is True
        self._transient_user_context = context.strip() if confirmed_context and context and context.strip() else None
        self._transient_selected_data_context = selected_data_context if confirmed_context else None
        self.executor.selected_data_context = self._transient_selected_data_context

        if stream:
            return self._chat_stream_with_context(previous_context, previous_selected_data_context)
        else:
            try:
                return self._chat_sync()
            finally:
                self._transient_user_context = previous_context
                self._transient_selected_data_context = previous_selected_data_context
                self.executor.selected_data_context = previous_selected_data_context

    def _chat_stream_with_context(self, previous_context: Optional[str], previous_selected_data_context: Optional[dict]) -> Iterator[str]:
        try:
            yield from self._chat_stream()
        finally:
            self._transient_user_context = previous_context
            self._transient_selected_data_context = previous_selected_data_context
            self.executor.selected_data_context = previous_selected_data_context

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

            plot_context_messages = []

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

                plot_info = self._extract_plot_info(result)
                if plot_info:
                    plot_context = self._build_plot_context_message(tool_name, plot_info)
                    if plot_context:
                        plot_context_messages.append(plot_context)

                llm_result = self._plot_pattern.sub(r'\1.', result)

                self.session.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": llm_result,
                })

            self.session.messages.extend(plot_context_messages)

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

    @staticmethod
    def _legacy_function_call_to_tool_delta(function_call: dict) -> dict:
        if not isinstance(function_call, dict):
            return {}
        function = {}
        if function_call.get("name"):
            function["name"] = function_call.get("name", "")
        if function_call.get("arguments"):
            function["arguments"] = function_call.get("arguments", "")
        if not function:
            return {}
        return {
            "tool_calls": [{
                "index": 0,
                "id": function_call.get("id", ""),
                "type": "function",
                "function": function,
            }]
        }

    @staticmethod
    def _compute_progress_percent(current, total) -> Optional[float]:
        try:
            cur = float(current)
            tot = float(total)
        except (TypeError, ValueError):
            return None
        if tot <= 0:
            return None
        return round(max(0.0, min(100.0, cur / tot * 100.0)), 1)

    def _emit_stream_event(self, event: dict) -> bool:
        if not self._on_stream_event:
            return False
        payload = dict(event or {})
        if payload.get("sequence") is None:
            self._stream_event_seq += 1
            payload["sequence"] = self._stream_event_seq
        payload.setdefault("created_at", round(time.time(), 3))
        self._on_stream_event(payload)
        return True

    def _dispatch_tool_stream(self, tool_name: str, arguments: dict, call_id: str = None) -> Iterator[str]:
        progress_queue = queue.Queue()
        done_marker = object()
        result_box: dict[str, str] = {}
        started = time.perf_counter()
        previous_callback = getattr(self.executor, "progress_callback", None)

        def progress_callback(event):
            payload = dict(event) if isinstance(event, dict) else {"message": str(event)}
            payload["type"] = "tool_progress"
            payload.setdefault("name", tool_name)
            if call_id:
                payload.setdefault("call_id", call_id)
            payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            if payload.get("progress_percent") is None:
                percent = self._compute_progress_percent(payload.get("current"), payload.get("total"))
                if percent is not None:
                    payload["progress_percent"] = percent
            progress_queue.put(payload)

        def run_tool():
            self.executor.progress_callback = progress_callback
            try:
                result_box["result"] = self.executor.dispatch(tool_name, arguments)
            except Exception as exc:
                result_box["result"] = f"Error executing {tool_name}: {type(exc).__name__}: {exc}"
            finally:
                self.executor.progress_callback = previous_callback
                progress_queue.put(done_marker)

        worker = threading.Thread(target=run_tool, name=f"pipelinets-tool-{tool_name}", daemon=True)
        worker.start()
        last_heartbeat = started

        while True:
            try:
                event = progress_queue.get(timeout=0.5)
            except queue.Empty:
                now = time.perf_counter()
                if now - last_heartbeat >= 5:
                    last_heartbeat = now
                    if self._emit_stream_event({
                        "type": "tool_progress",
                        "name": tool_name,
                        "call_id": call_id,
                        "stage": "running",
                        "message": "Still running...",
                        "elapsed_seconds": round(now - started, 2),
                        "heartbeat": True,
                    }):
                        yield ""
                continue

            if event is done_marker:
                break
            if self._emit_stream_event(event):
                yield ""

        worker.join()
        return result_box.get("result", "")

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
            if delta.get("function_call") and not delta.get("tool_calls"):
                delta.update(self._legacy_function_call_to_tool_delta(delta.get("function_call", {})))

            # Text content
            if delta.get("content"):
                full_content += delta["content"]
                yield delta["content"]

            # Reasoning content (DeepSeek thinking mode)
            if delta.get("reasoning_content"):
                rchunk = delta["reasoning_content"]
                full_reasoning += rchunk
                if self._emit_stream_event({"type": "reasoning", "text": rchunk}):
                    yield ""

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
            for tool_index, tc in enumerate(full_tool_calls):
                if not tc.get("id"):
                    tc["id"] = f"tool_call_{len(self.session.messages)}_{tool_index}"

            self.session.messages.append(
                _build_assistant_msg(full_content, full_tool_calls, full_reasoning, extra_delta_keys)
            )

            plot_context_messages = []

            for tool_index, tc in enumerate(full_tool_calls):
                tool_name = tc["function"]["name"]
                call_id = tc.get("id") or f"tool_call_{len(self.session.messages)}_{tool_index}"
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                # Emit tool_call event for UI
                if self._emit_stream_event({
                    "type": "tool_call",
                    "name": tool_name,
                    "call_id": call_id,
                    "index": tool_index,
                    "arguments": arguments,
                }):
                    yield ""

                tool_started = time.perf_counter()
                result = yield from self._dispatch_tool_stream(tool_name, arguments, call_id=call_id)
                elapsed_seconds = round(time.perf_counter() - tool_started, 2)

                plot_match = self._plot_pattern.search(result)
                plot_filename = plot_match.group(2) if plot_match else None
                plot_info = self._extract_plot_info(result)
                if plot_info:
                    plot_context = self._build_plot_context_message(tool_name, plot_info)
                    if plot_context:
                        plot_context_messages.append(plot_context)

                llm_result = self._plot_pattern.sub(r'\1.', result)

                # Emit tool_result event for UI
                is_error = llm_result.startswith("Error")
                is_truncated = len(llm_result) > 1200
                if self._emit_stream_event({
                    "type": "tool_result",
                    "name": tool_name,
                    "call_id": call_id,
                    "status": "error" if is_error else "success",
                    "success": not is_error,
                    "result": llm_result[:1200] + "..." if is_truncated else llm_result,
                    "result_size": len(llm_result),
                    "truncated": is_truncated,
                    "elapsed_seconds": elapsed_seconds,
                    "progress_percent": 100,
                }):
                    if plot_filename:
                        self._emit_stream_event({
                            "type": "plot_image",
                            "name": tool_name,
                            "call_id": call_id,
                            "filename": plot_filename,
                            "url": f"/api/plots/{plot_filename}",
                        })
                    yield ""

                self.session.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": llm_result,
                })

            self.session.messages.extend(plot_context_messages)

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
        visual_indices = [
            i for i, m in enumerate(self.session.messages)
            if m.get("_pipelinets_image_path")
        ]
        active_visual_indices = (
            set(visual_indices[-self.max_visual_context_images:])
            if self.max_visual_context_images
            else set()
        )
        return [
            {"role": "system", "content": self._build_system_prompt()},
        ] + [
            self._materialize_message(m, attach_image=i in active_visual_indices)
            for i, m in enumerate(self.session.messages)
        ]

    def _materialize_message(self, message: dict, attach_image: bool = False) -> dict:
        msg = {
            k: v for k, v in message.items()
            if not str(k).startswith("_pipelinets_")
        }
        image_path = message.get("_pipelinets_image_path")
        if self.multimodal_enabled and attach_image and image_path:
            data_url = self._image_path_to_data_url(image_path)
            if data_url:
                text = msg.get("content") or "Generated plot image."
                msg["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
        return msg

    @staticmethod
    def _normalize_multimodal(value) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        text = str(value or "auto").strip().lower()
        if text in {"1", "true", "yes", "on", "enabled", "enable"}:
            return "on"
        if text in {"0", "false", "no", "off", "disabled", "disable", "none"}:
            return "off"
        return "auto"

    def _resolve_multimodal_enabled(self) -> bool:
        if self.multimodal == "on":
            return True
        if self.multimodal == "off":
            return False
        provider = str(self._provider or "").lower()
        model = str(self._model or "").lower()
        if provider == "anthropic" and (
            "claude-3" in model
            or "claude-4" in model
            or "claude-sonnet-4" in model
            or "claude-opus-4" in model
        ):
            return True
        markers = (
            "gpt-4o", "gpt-4.1", "gpt-4.5", "o3", "o4",
            "vision", "vl", "llava", "internvl", "minicpm-v",
            "glm-4v", "qwen-vl", "qwen2-vl", "qwen2.5-vl",
            "qwen3-vl", "pixtral", "gemma-3", "deepseek-vl",
            "kimi-vl",
        )
        return any(marker in model for marker in markers)

    def _extract_plot_info(self, result: str) -> Optional[dict]:
        match = self._plot_pattern.search(result or "")
        if not match:
            return None
        return {"label": match.group(1), "filename": match.group(2)}

    def _build_plot_context_message(self, tool_name: str, plot_info: dict) -> Optional[dict]:
        if not self.multimodal_enabled:
            return None
        image_path = self._resolve_plot_file(plot_info.get("filename", ""))
        if not image_path:
            return None
        label = plot_info.get("label") or "Generated plot"
        return {
            "role": "user",
            "content": (
                f"Visual context generated by tool `{tool_name}`: {label}. "
                "Inspect this image directly when analyzing the data trend, seasonality, outliers, structural breaks, or forecast behavior."
            ),
            "_pipelinets_image_path": str(image_path),
            "_pipelinets_image_label": label,
            "_pipelinets_image_tool": tool_name,
        }

    def _resolve_plot_file(self, filename: str) -> Optional[Path]:
        if not filename:
            return None
        candidates = []
        if self.executor.plot_dir:
            candidates.append(self.executor.plot_dir / Path(filename).name)
        if self.session.last_plot_path:
            candidates.append(Path(self.session.last_plot_path))
        candidates.append(Path(filename))
        for path in candidates:
            try:
                if path.exists() and path.is_file():
                    return path
            except OSError:
                continue
        return None

    @staticmethod
    def _image_path_to_data_url(image_path: str) -> Optional[str]:
        path = Path(image_path)
        try:
            if not path.exists() or not path.is_file():
                return None
            if path.stat().st_size > 8 * 1024 * 1024:
                return None
            mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
            if not mime_type.startswith("image/"):
                return None
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        except OSError:
            return None

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
        for pkg in ("pandas", "numpy", "scikit-learn", "matplotlib", "torch", "scipy"):
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
        model_lines.append(f"Multimodal image input: {'enabled' if self.multimodal_enabled else 'disabled'} ({self.multimodal})")
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
            parts.append("\n## Confirmed user-selected data context")
            parts.append("The user explicitly confirmed this selection in the web data preview by clicking the small selection button. Treat the confirmed selection as the focal scope for this turn. If the user asks only about this period, selected data, selected dates, anomalies, or refers to the selection implicitly, analyze only the confirmed selected scope. If the user asks how this period changes within or compared with a broader scope such as all-day/全天, same-day, overall, surrounding period, or full dataset, first call get_data_context with the matching comparison scope and use the selected rows plus that broader returned context as evidence. Never replace missing broader context with dataset/domain knowledge. State in your answer which scopes were actually used.")
            parts.append(self._transient_user_context)
        parts.append("\n## Data Harness and Evidence Rules")
        parts.append("Uploaded preview rows, confirmed selection payloads, in-memory session data, uploaded CSV files listed in session state, tool results, and generated plots are the only valid data sources. Most diagnostic and plotting tools are automatically scoped to the confirmed selection when a confirmed selection exists. The get_data_context tool is the explicit escape hatch for broader evidence; use it for all-day/same-day/full-dataset comparisons. If a requested scope cannot be retrieved from these sources, say the comparison is unsupported instead of speculating.")
        parts.append(f"\n## Available Tools ({len(self.tools)})\n{tool_list}")

        return "\n".join(parts)

    def reset(self) -> None:
        """Reset the conversation and session state."""
        old_plot_dir = self.executor.plot_dir
        self._transient_user_context = None
        self._transient_selected_data_context = None
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

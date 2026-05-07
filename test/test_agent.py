"""Tests for the PipelineTS agent module.

Run with:  python -m pytest test/test_agent.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
#  Session tests
# ---------------------------------------------------------------------------


class TestSession:
    def test_empty_session(self):
        from PipelineTS.agent.session import Session

        s = Session()
        assert s.has_data() is False
        assert s.has_model() is False
        assert s.get_model() is None
        assert "not loaded" in s.status_summary()

    def test_session_dict(self):
        from PipelineTS.agent.session import Session

        s = Session()
        d = s.to_dict()
        assert d["data_loaded"] is False
        assert d["rows"] == 0

    def test_data_summary_empty(self):
        from PipelineTS.agent.session import Session

        s = Session()
        assert s.data_summary() == "No data loaded."

    def test_default_plot_path(self):
        from PipelineTS.agent.session import Session

        s = Session()
        assert s.default_plot_path("test") == "test.png"

        s.data_filepath = "/some/path/my_data.csv"
        assert s.default_plot_path("test") == "my_data_test.png"


# ---------------------------------------------------------------------------
#  Tools tests
# ---------------------------------------------------------------------------


class TestTools:
    def test_all_tools_have_required_fields(self):
        from PipelineTS.agent.tools import ALL_TOOLS

        for tool in ALL_TOOLS:
            assert tool["type"] == "function"
            f = tool["function"]
            assert "name" in f
            assert "description" in f
            assert "parameters" in f
            assert f["parameters"]["type"] == "object"

    def test_tool_count(self):
        from PipelineTS.agent.tools import ALL_TOOLS

        tool_names = {t["function"]["name"] for t in ALL_TOOLS}
        assert len(tool_names) == len(ALL_TOOLS), "Duplicate tool names"
        # Core operations covered
        assert "load_csv" in tool_names
        assert "train_pipeline" in tool_names
        assert "train_smart_router" in tool_names
        assert "forecast" in tool_names
        assert "get_session_status" in tool_names


# ---------------------------------------------------------------------------
#  Prompts tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_english_prompt(self):
        from PipelineTS.agent.prompts import get_system_prompt

        prompt = get_system_prompt("en")
        assert "PipelineTS" in prompt
        assert "load_csv" in prompt

    def test_chinese_prompt(self):
        from PipelineTS.agent.prompts import get_system_prompt

        prompt = get_system_prompt("zh")
        assert "PipelineTS" in prompt
        assert "时间序列" in prompt


# ---------------------------------------------------------------------------
#  Executor tests
# ---------------------------------------------------------------------------


class TestExecutor:
    @pytest.fixture
    def executor(self):
        from PipelineTS.agent.session import Session
        from PipelineTS.agent.executor import Executor

        s = Session()
        return Executor(s)

    def test_unknown_tool(self, executor):
        result = executor.dispatch("nonexistent_tool", {})
        assert "unknown" in result.lower()

    def test_inspect_without_data(self, executor):
        result = executor.dispatch("inspect_data", {})
        assert "no data loaded" in result.lower()

    def test_load_builtin_dataset(self, executor):
        result = executor.dispatch("load_builtin_dataset", {"dataset_name": "electric"})
        assert "Loaded" in result
        assert "electric" in result
        assert executor.session.has_data()
        assert executor.session.time_col == "date"
        assert executor.session.target_col == "value"

    def test_load_builtin_dataset_invalid(self, executor):
        result = executor.dispatch("load_builtin_dataset", {"dataset_name": "nonexistent"})
        assert "Error" in result or "unknown" in result.lower()

    def test_inspect_data(self, executor):
        executor.dispatch("load_builtin_dataset", {"dataset_name": "electric"})
        result = executor.dispatch("inspect_data", {"n_rows": 3})
        assert "Shape:" in result
        assert "397" in result or "rows" in result

    def test_check_missing_values(self, executor):
        executor.dispatch("load_builtin_dataset", {"dataset_name": "electric"})
        result = executor.dispatch("check_missing_values", {})
        assert "Missing" in result

    def test_check_stationarity(self, executor):
        executor.dispatch("load_builtin_dataset", {"dataset_name": "electric"})
        result = executor.dispatch("check_stationarity", {})
        assert "Stationarity" in result
        assert "non_stationary" in result.lower()

    def test_list_available_models(self, executor):
        result = executor.dispatch("list_available_models", {})
        assert "Neural Network" in result
        assert "Machine Learning" in result

    def test_get_session_status(self, executor):
        result = executor.dispatch("get_session_status", {})
        assert "not loaded" in result.lower()

        executor.dispatch("load_builtin_dataset", {"dataset_name": "electric"})
        result = executor.dispatch("get_session_status", {})
        assert "electric" in result.lower() or "397" in result


# ---------------------------------------------------------------------------
#  LLM provider tests
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_create_provider_no_key(self):
        from PipelineTS.agent.llm import create_provider

        # Temporarily clear keys
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="No LLM API key"):
                create_provider(provider="auto")
        finally:
            if old_openai:
                os.environ["OPENAI_API_KEY"] = old_openai
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_create_provider_openai(self):
        from PipelineTS.agent.llm import create_provider, OpenAIProvider

        os.environ["OPENAI_API_KEY"] = "sk-test"
        try:
            provider = create_provider(provider="openai")
            assert isinstance(provider, OpenAIProvider)
        finally:
            del os.environ["OPENAI_API_KEY"]


# ---------------------------------------------------------------------------
#  Agent tests
# ---------------------------------------------------------------------------


class TestAgent:
    def test_agent_init_fails_without_key(self):
        from PipelineTS.agent.agent import TSAgent

        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="No LLM API key"):
                TSAgent(provider="auto")
        finally:
            if old_openai:
                os.environ["OPENAI_API_KEY"] = old_openai
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_agent_status(self):
        from PipelineTS.agent.agent import TSAgent
        from PipelineTS.agent.llm import OpenAIProvider

        # Create agent with a mock provider that won't actually call the API
        # We just test that the agent structure is correct
        os.environ["OPENAI_API_KEY"] = "sk-test"
        try:
            agent = TSAgent(provider="openai")
            status = agent.status()
            assert "not loaded" in status.lower()
            agent.reset()
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_agent_build_messages(self):
        from PipelineTS.agent.agent import TSAgent

        os.environ["OPENAI_API_KEY"] = "sk-test"
        try:
            agent = TSAgent(provider="openai")
            msgs = agent._build_messages()
            assert len(msgs) >= 1
            assert msgs[0]["role"] == "system"
        finally:
            del os.environ["OPENAI_API_KEY"]

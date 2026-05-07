# PipelineTS Agent — natural language interface for time series analysis.
#
# Quick start:
#   from PipelineTS.agent import TSAgent
#   agent = TSAgent()                          # uses OPENAI_API_KEY
#   response = agent.chat("Load the electric dataset and inspect it")
#
# CLI:
#   pipeline-ts chat        # interactive terminal chat
#   pipeline-ts web         # Flask web GUI
#   pipeline-ts list-models # list available models
#
# Configuration:
#   Config file (~/.pipelints/config.toml) or env vars (OPENAI_API_KEY, etc.)

from PipelineTS.agent.agent import TSAgent
from PipelineTS.agent.session import Session
from PipelineTS.agent.config import Config
from PipelineTS.agent.llm import create_provider, LLMProvider, OpenAIProvider, AnthropicProvider

__all__ = [
    "TSAgent",
    "Session",
    "Config",
    "create_provider",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
]

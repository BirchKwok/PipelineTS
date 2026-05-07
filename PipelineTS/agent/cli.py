"""Command-line interface for the PipelineTS agent.

Start an interactive chat session:

    python -m PipelineTS.agent.cli
    pipeline-ts chat
"""

from __future__ import annotations

import os
import sys
import argparse

from PipelineTS.agent.agent import TSAgent


# ---------------------------------------------------------------------------
#  Terminal formatting (with optional rich support)
# ---------------------------------------------------------------------------

def _try_rich():
    """Return Rich console and markup helpers, or None if not available."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.live import Live
        from rich.spinner import Spinner
        from rich import print as rprint

        return Console, Markdown, Panel, Live, Spinner, rprint
    except ImportError:
        return None, None, None, None, None, None


def _print_welcome(console=None):
    """Print the welcome banner."""
    banner = r"""
╔══════════════════════════════════════════════════╗
║        PipelineTS Agent — Time Series AI         ║
║                                                  ║
║  自然语言驱动的时序分析助手                          ║
║  Type /help for commands  •  /quit to exit        ║
╚══════════════════════════════════════════════════╝
"""
    if console:
        from rich.panel import Panel

        console.print(Panel(banner.strip(), style="bold cyan"))
    else:
        print(banner)


def _print_help():
    """Print available commands."""
    help_text = """
Available commands:
  /help          Show this help message
  /status        Show current session status (loaded data, trained models)
  /reset         Reset the conversation and session state
  /save <path>   Save conversation history to a JSON file
  /load <path>   Load conversation history from a JSON file
  /quit, /exit   Exit the agent

During chat, you can use natural language to:
  - Load data: CSV files or built-in datasets
  - Explore data: inspect, check quality, visualize
  - Preprocess: handle missing values, outliers, scaling
  - Train models: single models, ModelPipeline, or SmartRouter
  - Evaluate: leaderboard, backtesting, residual analysis
  - Predict: point forecasts, prediction intervals
"""
    print(help_text)


def _print_status(agent: TSAgent):
    """Print current session status."""
    print("\n── Session Status ──")
    print(agent.session.status_summary())
    print()


# ---------------------------------------------------------------------------
#  Main chat loop
# ---------------------------------------------------------------------------

def run_chat(
    provider: str = "auto",
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    lang: str = "en",
    verbose: bool = False,
    rich_mode: bool = True,
):
    """Run the interactive agent chat loop.

    Parameters
    ----------
    provider, model, api_key, base_url, lang, verbose : see TSAgent
    rich_mode : bool
        Whether to use Rich for fancy terminal output. Falls back to plain text.
    """
    Console, Markdown, Panel, Live, Spinner, rprint = (None,) * 6
    console = None

    if rich_mode:
        Console, Markdown, Panel, Live, Spinner, rprint = _try_rich()
        if Console:
            console = Console()

    # ── Create agent ─────────────────────────────────
    try:
        agent = TSAgent(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            lang=lang,
            verbose=verbose,
        )
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("\nMake sure you have set one of:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nOr for local models (Ollama):")
        print("  export OPENAI_API_KEY=ollama")
        print("  export OPENAI_BASE_URL=http://localhost:11434/v1")
        print("  export OPENAI_MODEL=qwen2.5")
        sys.exit(1)

    _print_welcome(console)

    # ── Event loop ───────────────────────────────────
    while True:
        try:
            user_input = input("\n你> " if lang == "zh" else "\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd, *args = user_input.split(maxsplit=1)
            arg = args[0] if args else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/help":
                _print_help()
            elif cmd == "/status":
                _print_status(agent)
            elif cmd == "/reset":
                agent.reset()
                print("Session reset.")
            elif cmd == "/save":
                path = arg or "session.json"
                try:
                    agent.save_session(path)
                    print(f"Conversation saved to: {path}")
                except Exception as e:
                    print(f"Save failed: {e}")
            elif cmd == "/load":
                if not arg:
                    print("Usage: /load <path>")
                    continue
                try:
                    agent.load_session(arg)
                    print(f"Conversation loaded from: {arg}")
                except Exception as e:
                    print(f"Load failed: {e}")
            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")
            continue

        # ── Send to agent ────────────────────────────
        try:
            if console:
                with console.status("[bold green]Thinking...[/bold green]"):
                    response = agent.chat(user_input)

                # Print with Markdown rendering
                console.print()
                try:
                    md = Markdown(response)
                    console.print(md)
                except Exception:
                    console.print(response)
            else:
                print()
                # Use streaming for plain terminal
                for chunk in agent.chat(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print()

        except Exception as e:
            print(f"\nError: {e}")


# ---------------------------------------------------------------------------
#  Entry points
# ---------------------------------------------------------------------------

def main():
    """Main entry point for `pipeline-ts chat`."""
    parser = argparse.ArgumentParser(
        prog="pipeline-ts",
        description="PipelineTS Agent — natural language time series analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # `pipeline-ts chat`
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--provider", default="auto",
        choices=["auto", "openai", "anthropic"],
        help="LLM provider (default: auto-detect)",
    )
    chat_parser.add_argument(
        "--model", default=None,
        help="Model name override",
    )
    chat_parser.add_argument(
        "--api-key", default=None,
        help="API key override",
    )
    chat_parser.add_argument(
        "--base-url", default=None,
        help="Base URL for OpenAI-compatible APIs",
    )
    chat_parser.add_argument(
        "--lang", default="en", choices=["en", "zh"],
        help="Language for system prompts",
    )
    chat_parser.add_argument(
        "--verbose", action="store_true",
        help="Print tool call details",
    )
    chat_parser.add_argument(
        "--no-rich", action="store_true",
        help="Disable Rich terminal formatting",
    )

    # `pipeline-ts list-models`
    list_parser = subparsers.add_parser(
        "list-models", help="List all available time series models"
    )

    # `pipeline-ts web`
    web_parser = subparsers.add_parser(
        "web", help="Launch the Flask web GUI"
    )
    web_parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to listen on (default: 7860)",
    )
    web_parser.add_argument(
        "--debug", action="store_true",
        help="Enable Flask debug mode",
    )

    args = parser.parse_args()

    if args.command == "chat":
        run_chat(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            lang=args.lang,
            verbose=args.verbose,
            rich_mode=not args.no_rich,
        )
    elif args.command == "list-models":
        from PipelineTS.pipeline.pipeline_models import get_all_available_models

        models = get_all_available_models()
        print("Available models:")
        for name in sorted(models):
            print(f"  {name}")
    elif args.command == "web":
        from PipelineTS.agent.web import launch_web

        launch_web(host=args.host, port=args.port, debug=args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

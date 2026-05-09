"""Web GUI launcher for the PipelineTS agent (Flask backend).

Launch with:
    pipelinets web
    python -m PipelineTS.agent.web
"""

from __future__ import annotations


def launch_web(host: str = "127.0.0.1", port: int = 7860, debug: bool = False, **kwargs):
    """Launch the Flask web UI."""
    from PipelineTS.agent.flask_app import launch_web as _launch
    _launch(host=host, port=port, debug=debug, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PipelineTS Agent — Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    launch_web(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

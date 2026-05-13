"""Console script entry point for the dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yuxin-mea-dashboard",
        description="Launch the yuxin_mea read-only browser dashboard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./pipeline_config.json"),
        help="Path to pipeline config JSON (default: ./pipeline_config.json).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8050, help="Bind port (default: 8050).")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args(argv)

    if not args.config.exists():
        sys.stderr.write(
            f"Config file not found: {args.config}\n"
            "Copy `config/pipeline_config.example.json` to that location, "
            "or run `yuxin-mea-config-builder` (Phase 3) once it ships.\n"
        )
        return 2

    from .app import build_app

    app = build_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0

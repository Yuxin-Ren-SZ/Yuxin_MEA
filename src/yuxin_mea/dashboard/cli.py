"""Console script entry point for the dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yuxin-mea-dashboard",
        description="Launch the yuxin_mea browser dashboard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./pipeline_config.json"),
        help=(
            "Path to pipeline config JSON (default: ./pipeline_config.json). "
            "If the file does not exist yet, the dashboard launches in "
            "config-only mode: data pages show a 'no config yet' banner, "
            "and the first Save on the Settings page creates the file."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8050, help="Bind port (default: 8050).")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args(argv)

    if not args.config.exists():
        sys.stderr.write(
            f"⚠  Config file not found: {args.config}\n"
            "   Launching dashboard in config-only mode — fill in the "
            "Settings page and Save to create it.\n\n"
        )

    from .app import build_app

    app = build_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0

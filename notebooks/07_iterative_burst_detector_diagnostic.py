"""Iterative burst detector — diagnostic dashboard runner.

Runs the iterative burst detector on every Kilosort recording under
``REAL_DATA_ROOT``, optionally writes one interactive HTML per section per
recording, then launches a Dash web app that aggregates everything.

Open the dashboard at http://localhost:8050 (or whatever port you specify).

Usage::

    # Run with defaults
    python notebooks/07_iterative_burst_detector_diagnostic.py

    # Skip the per-section HTML files (faster startup)
    python notebooks/07_iterative_burst_detector_diagnostic.py --no-html

    # Different port / data root
    python notebooks/07_iterative_burst_detector_diagnostic.py \\
        --port 8080 --data-root path/to/recordings

All plotting / loading / batch logic lives in
``notebooks/burst_diagnostic/viz.py`` so this script stays thin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until we find a directory containing pipeline_tasks/."""
    candidate = start.resolve()
    for path in (candidate, *candidate.parents):
        if (path / "pipeline_tasks").is_dir():
            return path
    raise RuntimeError(
        f"Could not locate repository root (no `pipeline_tasks/` ancestor of {start})"
    )


def _bootstrap_sys_path(repo_root: Path) -> None:
    """Ensure pipeline_tasks and burst_diagnostic are importable."""
    for entry in (repo_root, repo_root / "notebooks"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = _resolve_repo_root(Path(__file__).parent)
    default_data = repo_root / "data" / "real_sorted_data"
    default_out = repo_root / "output" / "nb07_diagnostic_figures"

    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=default_data,
        help="Root directory holding all Kilosort recordings to diagnose.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Where to write per-section HTML files.",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=["good"],
        help=(
            "Kilosort cluster labels to include. Pass --labels with no values "
            "to include every cluster."
        ),
    )
    p.add_argument(
        "--trace-kind",
        choices=["default", "no_gate"],
        default="default",
        help="Which detector run feeds the per-recording per-section plots.",
    )
    p.add_argument(
        "--plot-all-iters",
        action="store_true",
        help=(
            "In Section C (LDA PCA slider), keep one frame per iteration "
            "(default: start, middle, final only)."
        ),
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Embed Plotly JS inside each HTML (~3 MB / file) instead of CDN.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip writing the per-section HTML files; only launch the dashboard.",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Dash server.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash server.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode (auto-reload on code changes).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = _resolve_repo_root(Path(__file__).parent)
    _bootstrap_sys_path(repo_root)

    # Late imports so --help works without sklearn / plotly / dash being importable.
    from pipeline_tasks.analysis import IterativeBurstConfig
    from burst_diagnostic import viz

    config = IterativeBurstConfig()

    labels: set[str] | None = set(args.labels) if args.labels else None
    if labels is not None and not labels:
        labels = None

    sources = viz.discover_real_spike_sources(args.data_root)
    if not sources:
        print(f"No recordings found under {args.data_root}", file=sys.stderr)
        return 2

    print(f"discovered {len(sources)} recording(s) under {args.data_root}:")
    for s in sources:
        print(f"  {s}")
    print()

    batch = viz.run_batch(
        sources,
        config_default=config,
        labels=labels,
        verbose=True,
    )

    if not args.no_html:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nwriting per-section HTML files to {args.output_dir} ...")
        saved = viz.save_all_section_htmls(
            batch,
            output_dir=args.output_dir,
            trace_kind=args.trace_kind,
            plot_all_iters=args.plot_all_iters,
            offline=args.offline,
        )
        print(f"  wrote {len(saved)} files")

    print(
        f"\nlaunching dashboard on http://{args.host}:{args.port}/  "
        f"(Ctrl-C to stop)"
    )
    app = viz.build_dashboard_app(batch, initial_trace_kind=args.trace_kind)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

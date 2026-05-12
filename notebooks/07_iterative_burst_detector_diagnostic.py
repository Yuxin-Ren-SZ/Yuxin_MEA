"""Iterative burst detector — diagnostic dashboard runner.

Runs the iterative burst detector on every Kilosort recording under
``REAL_DATA_ROOT``, launches a Dash web app that aggregates every
diagnostic plot, and (optionally, in the background) writes one
standalone HTML per section per recording.

Open the dashboard at http://localhost:8050 (or whatever port you specify).

Usage::

    # First run — recommended fast smoke test (~30 s to "SERVER READY"):
    python notebooks/07_iterative_burst_detector_diagnostic.py --quick 1 --no-html

    # Full run on every recording (~3 min before "SERVER READY";
    # per-section HTML files keep writing in the background while the
    # dashboard is live):
    python notebooks/07_iterative_burst_detector_diagnostic.py

    # Different port / data root
    python notebooks/07_iterative_burst_detector_diagnostic.py \\
        --port 8080 --data-root path/to/recordings

⚠ The dashboard URL only responds **after** the script prints
``SERVER READY``. The detector batch takes ~30 s per recording per config
(default + no_gate), so 3 recordings ≈ 3 minutes before the server binds.
Until then, ``http://localhost:8050`` will refuse the connection.

All plotting / loading / batch logic lives in
``notebooks/burst_diagnostic/viz.py`` so this script stays thin.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
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
        "--quick",
        type=int,
        metavar="N",
        default=None,
        help="Run on only the first N discovered recordings (fast smoke test).",
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

    t0 = time.monotonic()

    def stage(msg: str) -> None:
        """Loud, flushed progress print with elapsed wall time."""
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    config = IterativeBurstConfig()

    labels: set[str] | None = set(args.labels) if args.labels else None
    if labels is not None and not labels:
        labels = None

    stage(f"discovering recordings under {args.data_root} ...")
    sources = viz.discover_real_spike_sources(args.data_root)
    if not sources:
        print(f"No recordings found under {args.data_root}", file=sys.stderr)
        return 2
    if args.quick is not None:
        sources = sources[: args.quick]
        stage(f"quick mode: limiting to first {len(sources)} recording(s)")

    stage(f"found {len(sources)} recording(s):")
    for s in sources:
        print(f"          {s}", flush=True)

    stage(
        f"running detector on {len(sources)} recording(s) × 2 configs "
        f"(~30 s per run; total ~{len(sources) * 60} s) ..."
    )
    batch = viz.run_batch(
        sources,
        config_default=config,
        labels=labels,
        verbose=True,
    )
    total_burstlets = sum(len(r.burstlets) for r in batch.results.values())
    total_net = sum(len(r.network_bursts) for r in batch.results.values())
    stage(
        f"batch complete: {total_burstlets} burstlets, "
        f"{total_net} network bursts across {len(batch.recording_names)} recordings"
    )

    stage("building dashboard layout (~20 s — pre-renders cross-recording figures) ...")
    app = viz.build_dashboard_app(batch, initial_trace_kind=args.trace_kind)
    stage("dashboard layout ready")

    # Start the (slow) HTML save in a daemon thread so it doesn't delay
    # `app.run()`. The thread keeps writing while the dashboard is live;
    # it dies with the process on Ctrl-C.
    if not args.no_html:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        def _save_html_worker() -> None:
            try:
                saved = viz.save_all_section_htmls(
                    batch,
                    output_dir=args.output_dir,
                    trace_kind=args.trace_kind,
                    plot_all_iters=args.plot_all_iters,
                    offline=args.offline,
                )
                print(
                    f"[html-save] wrote {len(saved)} files to {args.output_dir}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — log and continue
                print(f"[html-save] failed: {exc}", flush=True)

        stage(
            f"starting background HTML save → {args.output_dir}  "
            f"(continues while dashboard is live)"
        )
        threading.Thread(
            target=_save_html_worker, daemon=True, name="html-save"
        ).start()

    print(flush=True)
    stage(f"SERVER READY  →  open http://{args.host}:{args.port}/   (Ctrl-C to stop)")
    print(flush=True)

    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

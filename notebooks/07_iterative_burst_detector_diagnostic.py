"""Iterative burst detector — HTML report generator.

Runs the iterative burst detector on every Kilosort recording under
``--data-root`` and writes one standalone HTML per diagnostic section per
recording. No web server; intended for offline report sharing.

For interactive exploration, use ``yuxin-mea-dashboard`` and navigate to
``/burst-diagnostic`` — that page reuses the same figure functions plus a
``BatchResults`` disk cache so reloads are fast.

Usage::

    # Smoke test on one recording
    python notebooks/07_iterative_burst_detector_diagnostic.py --quick 1

    # Full run with offline Plotly JS embedded in each HTML
    python notebooks/07_iterative_burst_detector_diagnostic.py --offline
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from yuxin_mea.analysis.burst_diagnostic import (
    discover_real_spike_sources,
    run_batch,
    save_all_section_htmls,
)
from yuxin_mea.analysis.iterative_burst_detector import IterativeBurstConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_data = repo_root / "data" / "real_sorted_data"
    default_out = repo_root / "output" / "nb07_diagnostic_figures"

    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=default_data,
                   help="Root directory holding all Kilosort recordings.")
    p.add_argument("--output-dir", type=Path, default=default_out,
                   help="Where to write per-section HTML files.")
    p.add_argument("--labels", nargs="*", default=["good"],
                   help="Kilosort cluster labels to include. Pass --labels with no "
                        "values to include every cluster.")
    p.add_argument("--quick", type=int, metavar="N", default=None,
                   help="Run on only the first N discovered recordings.")
    p.add_argument("--trace-kind", choices=["default", "no_gate"], default="default",
                   help="Which detector run feeds the per-recording per-section plots.")
    p.add_argument("--plot-all-iters", action="store_true",
                   help="Keep one frame per iteration in section C (default: start/mid/final).")
    p.add_argument("--offline", action="store_true",
                   help="Embed Plotly JS in each HTML (~3 MB/file) instead of CDN.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.monotonic()

    def stage(msg: str) -> None:
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    labels: set[str] | None = set(args.labels) or None

    stage(f"discovering recordings under {args.data_root} ...")
    sources = discover_real_spike_sources(args.data_root)
    if not sources:
        print(f"No recordings found under {args.data_root}", file=sys.stderr)
        return 2
    if args.quick is not None:
        sources = sources[: args.quick]
        stage(f"quick mode: limiting to first {len(sources)} recording(s)")

    stage(f"found {len(sources)} recording(s); running detector (default + no_gate configs) ...")
    batch = run_batch(sources, config_default=IterativeBurstConfig(),
                      labels=labels, verbose=True)
    total_burstlets = sum(len(r.burstlets) for r in batch.results.values())
    total_net = sum(len(r.network_bursts) for r in batch.results.values())
    stage(f"batch complete: {total_burstlets} burstlets, "
          f"{total_net} network bursts across {len(batch.recording_names)} recordings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage(f"writing per-section HTML files to {args.output_dir} ...")
    saved = save_all_section_htmls(
        batch,
        output_dir=args.output_dir,
        trace_kind=args.trace_kind,
        plot_all_iters=args.plot_all_iters,
        offline=args.offline,
    )
    stage(f"wrote {len(saved)} HTML file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

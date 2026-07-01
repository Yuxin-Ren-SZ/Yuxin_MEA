#!/usr/bin/env python3
"""Sweep the UMAP clustering dimensionality (pre-HDBSCAN) for one assay.

For each requested UMAP n_components (default 2, 3, 5) this driver, sequentially:
  1. writes a temp config with ``tasks.ml_burst_detection.umap_n_components`` and a
     per-dim ``output_root`` (so each dim's burst data is kept side by side),
  2. resets the ml_burst_detection task to NOT_RUN for the target recording
     (``PipelineManager.refresh``) — the runner skips COMPLETE tasks and has no
     ``--force`` flag,
  3. runs the detector (``yuxin-mea-run``),
  4. regenerates the inspection HTML (``inspect_ml_bursts.py``) into a per-dim dir,
  5. regenerates the UMAP animations (``animate_ml_umap.py``) into a per-dim dir.

After every dim it builds a comparison report (metrics table + plots).

The tracked ``pipeline_config.json`` and existing output dirs are never modified;
everything lands under ``--sweep-root``.

Example
-------
    python scripts/sweep_umap_dims.py \
        --config pipeline_config.json \
        --sweep-root /mnt/benshalom-nas/analysis/Sadegh/new/CX138/umap_dim_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("sweep_umap_dims")

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDING = "CX138/260325/T003346/Network/000012"

# metrics.json -> (column label, dotted path under "network_bursts")
NB_FIELDS = {
    "n_bursts": "count",
    "rate_hz": "rate",
    "dur_mean_s": "duration.mean",
    "participation_mean": "participation.mean",
    "peak_synchrony_mean": "peak_synchrony.mean",
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def write_temp_config(base_config: Path, dim: int, data_root: Path, cfg_dir: Path) -> Path:
    cfg = json.loads(base_config.read_text())
    ml = cfg.setdefault("tasks", {}).setdefault("ml_burst_detection", {})
    ml["umap_n_components"] = int(dim)
    ml["cluster_embedding_mode"] = "umap"
    ml["output_root"] = str(data_root)
    ml["debug"] = True  # debug_trace.pkl is required by both scripts
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg_dir / f"cfg_umap{dim}d.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def refresh_detector(cfg_path: Path) -> int:
    """Reset ml_burst_detection -> NOT_RUN for RECORDING. Returns #records reset."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from yuxin_mea.cli.run import _setup_pipeline  # noqa: E402

    _, _, pipeline_mgr = _setup_pipeline(cfg_path)
    n = pipeline_mgr.refresh("ml_burst_detection", recording_key=RECORDING)
    logger.info("refresh: reset %d record(s) to NOT_RUN", n)
    return n


# --------------------------------------------------------------------------- #
# subprocess steps
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_detector(cfg_path: Path) -> None:
    exe = shutil.which("yuxin-mea-run") or "yuxin-mea-run"
    _run([exe, "--config", str(cfg_path), "--tasks", "ml_burst_detection",
          "--recordings", RECORDING])


def run_inspect(cfg_path: Path, out_dir: Path) -> None:
    _run([sys.executable, str(REPO_ROOT / "scripts" / "inspect_ml_bursts.py"),
          "--config", str(cfg_path), "--output-dir", str(out_dir)])


def run_animate(cfg_path: Path, out_dir: Path, duration: float, history: float) -> None:
    _run([sys.executable, str(REPO_ROOT / "scripts" / "animate_ml_umap.py"),
          "--config", str(cfg_path), "--output-dir", str(out_dir),
          "--duration", str(duration), "--history", str(history)])


# --------------------------------------------------------------------------- #
# metrics collection
# --------------------------------------------------------------------------- #
def _dig(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def collect_metrics(data_root: Path, dim: int) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(data_root.glob("**/ml_burst_detection/metrics.json")):
        well_dir = metrics_path.parent.parent  # .../rec####/well###
        rec_name, well_id = well_dir.parent.name, well_dir.name
        metrics = json.loads(metrics_path.read_text())
        nb = metrics.get("network_bursts", {}) or {}
        row = {"dim": dim, "rec": rec_name, "well": well_id}
        for col, path in NB_FIELDS.items():
            row[col] = _dig(nb, path)
        diag_path = metrics_path.parent / "diagnostics.json"
        if diag_path.exists():
            diag = json.loads(diag_path.read_text())
            row["n_clusters"] = diag.get("cluster_n_clusters")
            labels = diag.get("cluster_burst_labels") or []
            row["n_burst_clusters"] = len(labels)
            row["cluster_decision"] = diag.get("cluster_decision")
            row["burst_activity_detected"] = diag.get("burst_activity_detected")
        rows.append(row)
    logger.info("collected metrics for %d well(s) at dim=%d", len(rows), dim)
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(all_rows: list[dict], dims: list[int], sweep_root: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("no metrics collected; skipping report")
        return
    df = df.sort_values(["rec", "well", "dim"]).reset_index(drop=True)
    csv_path = sweep_root / "sweep_metrics.csv"
    df.to_csv(csv_path, index=False)

    plots_dir = sweep_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_files = _make_plots(df, dims, plots_dir)

    # aggregate summary per dim
    agg = (df.groupby("dim")
             .agg(wells=("well", "count"),
                  mean_n_bursts=("n_bursts", "mean"),
                  mean_rate_hz=("rate_hz", "mean"),
                  mean_n_clusters=("n_clusters", "mean"),
                  mean_n_burst_clusters=("n_burst_clusters", "mean"),
                  mean_participation=("participation_mean", "mean"),
                  mean_peak_synchrony=("peak_synchrony_mean", "mean"),
                  n_wells_with_bursts=("burst_activity_detected", "sum"))
             .reindex(dims))

    md = sweep_root / "report_umap_dim_sweep.md"
    lines: list[str] = []
    lines.append("# UMAP clustering-dim sweep — assay 000012 (CX138)\n")
    lines.append(f"Recording: `{RECORDING}` — {df['well'].nunique()} wells "
                 f"× dims {dims}.\n")
    lines.append("UMAP n_components feeds HDBSCAN (the pre-cluster embedding). "
                 "All other params held fixed.\n")

    lines.append("\n## Summary (mean across wells, per dim)\n")
    lines.append(_df_to_md(agg.reset_index().rename(columns={"index": "dim"})))

    lines.append("\n## Comparison plots\n")
    for f in plot_files:
        rel = f.relative_to(sweep_root)
        lines.append(f"![{f.stem}]({rel})\n")

    lines.append("\n## Per-well, per-dim metrics\n")
    lines.append(_df_to_md(df))

    lines.append("\n## Output locations\n")
    for d in dims:
        lines.append(f"- **dim {d}**: data `data_umap{d}d/` · "
                     f"inspect `inspect_umap{d}d/` · anim `anim_umap{d}d/`")
    lines.append(f"\nRaw table: `{csv_path.name}`\n")

    md.write_text("\n".join(lines))
    logger.info("report written: %s", md)


def _df_to_md(df) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # tabulate may be absent; fall back to a plain pipe table
        cols = list(df.columns)
        out = ["| " + " | ".join(map(str, cols)) + " |",
               "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(
                ("" if r[c] is None else f"{r[c]:.4g}" if isinstance(r[c], float)
                 else str(r[c])) for c in cols) + " |")
        return "\n".join(out)


def _make_plots(df, dims: list[int], plots_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files: list[Path] = []
    metrics = [
        ("n_bursts", "network burst count"),
        ("n_clusters", "HDBSCAN cluster count"),
        ("n_burst_clusters", "# clusters labelled burst"),
        ("peak_synchrony_mean", "mean peak synchrony"),
        ("participation_mean", "mean participation"),
        ("rate_hz", "burst rate (Hz)"),
    ]
    # one multi-panel figure: per-well faint lines + mean line vs dim
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        for (rec, well), g in df.groupby(["rec", "well"]):
            g = g.sort_values("dim")
            ax.plot(g["dim"], g[col], color="0.8", lw=0.8, zorder=1)
        means = df.groupby("dim")[col].mean().reindex(dims)
        ax.plot(dims, means.values, "o-", color="C3", lw=2, zorder=3, label="mean")
        ax.set_title(title)
        ax.set_xlabel("UMAP n_components")
        ax.set_xticks(dims)
        ax.legend(fontsize=8)
    fig.suptitle("Burst metrics vs UMAP clustering dim — assay 000012", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = plots_dir / "metrics_vs_dim.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)
    return files


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "pipeline_config.json")
    ap.add_argument("--sweep-root", type=Path, required=True,
                    help="Root dir for all per-dim outputs and the report.")
    ap.add_argument("--dims", type=str, default="2,3,5",
                    help="Comma-separated UMAP n_components values (order = run order).")
    ap.add_argument("--duration", type=float, default=60.0, help="Animation length (s).")
    ap.add_argument("--history", type=float, default=5.0, help="Animation trail (s).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a dim whose data + inspect + anim dirs already exist.")
    ap.add_argument("--no-anim", action="store_true", help="Skip the animation step.")
    ap.add_argument("--report-only", action="store_true",
                    help="Only (re)build the report from existing per-dim data dirs.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    sweep_root: Path = args.sweep_root
    cfg_dir = sweep_root / "configs"
    base_config: Path = args.config

    all_rows: list[dict] = []
    for dim in dims:
        data_root = sweep_root / f"data_umap{dim}d"
        inspect_dir = sweep_root / f"inspect_umap{dim}d"
        anim_dir = sweep_root / f"anim_umap{dim}d"

        if not args.report_only:
            done = (data_root.exists() and inspect_dir.exists()
                    and (args.no_anim or anim_dir.exists()))
            if args.skip_existing and done:
                logger.info("dim=%d: outputs exist, skipping (--skip-existing)", dim)
            else:
                logger.info("=== dim=%d ===", dim)
                cfg_path = write_temp_config(base_config, dim, data_root, cfg_dir)
                refresh_detector(cfg_path)
                run_detector(cfg_path)
                run_inspect(cfg_path, inspect_dir)
                if not args.no_anim:
                    run_animate(cfg_path, anim_dir, args.duration, args.history)

        all_rows.extend(collect_metrics(data_root, dim))

    build_report(all_rows, dims, sweep_root)
    logger.info("sweep complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reset the ``ml_burst_detection`` task to NOT_RUN so the pipeline re-runs it.

There is no CLI ``--overwrite`` in ``yuxin-mea-run``: a task whose status is
COMPLETE is skipped, and a changed config alone does not retrigger it. To
regenerate ML burst outputs (e.g. after the posterior_peak gate + min_fit_units
floor replaced the old mean-LLR gate) the task's status must first be reset.

``ml_burst_detection`` is a leaf (its only dependency is ``auto_curation`` and no
task depends on it), so ``PipelineManager.refresh`` cascades to the task itself
only — every upstream task (preprocessing/sorting/analyzer/auto_curation) stays
COMPLETE and is NOT redone.

Usage
-----
    # preview what would reset for one sample
    python scripts/reset_ml_burst.py --config pipeline_config_run.json \
        --sample CX138 --dry-run

    # actually reset that sample, then emit the recording keys to feed --recordings
    python scripts/reset_ml_burst.py --config pipeline_config_run.json --sample CX138

    # all three samples
    python scripts/reset_ml_burst.py --config pipeline_config_run.json \
        --sample CX118,CX138,CX169

The comma-joined recording keys printed at the end go straight into::

    yuxin-mea-run --config <cfg> --tasks ml_burst_detection --recordings <keys> --jobs N
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from yuxin_mea.cli.run import _setup_pipeline

TASK = "ml_burst_detection"
DEFAULT_SAMPLES = ["CX118", "CX138", "CX169"]


def _status_str(record) -> str:
    st = getattr(record, "status", None)
    return str(getattr(st, "value", st))


def _recording_keys_for_sample(dataset_mgr, pipeline_mgr, sample: str) -> list[str]:
    """Cache keys for `sample` that are actually queued in the pipeline."""
    recs = dataset_mgr.get_recording_by([("sample_id", "==", sample)])
    keys = []
    for r in recs:
        key = r.cache_key
        # keep only recordings that have entries in the pipeline cache
        if pipeline_mgr.get_entries_for_recording(key):
            keys.append(key)
    return sorted(keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True,
                    help="pipeline_config_run.json (same file passed to yuxin-mea-run).")
    ap.add_argument("--sample", default=",".join(DEFAULT_SAMPLES),
                    help="Comma-separated sample ids (default CX118,CX138,CX169).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would reset; change nothing.")
    args = ap.parse_args()

    samples = [s.strip() for s in args.sample.split(",") if s.strip()]
    _cm, dataset_mgr, pipeline_mgr = _setup_pipeline(args.config)

    all_keys: list[str] = []
    for sample in samples:
        keys = _recording_keys_for_sample(dataset_mgr, pipeline_mgr, sample)
        all_keys.extend(keys)
        # tally current ml_burst status across all wells of this sample
        status = Counter()
        n_wells = 0
        for key in keys:
            for entry in pipeline_mgr.get_entries_for_recording(key):
                rec = entry.tasks.get(TASK)
                if rec is None:
                    continue
                n_wells += 1
                status[_status_str(rec)] += 1
        print(f"[{sample}] {len(keys)} recording(s), {n_wells} well(s) with "
              f"{TASK}: {dict(status)}")

    if not all_keys:
        print("No matching recordings found in the pipeline cache — nothing to do.")
        return

    if args.dry_run:
        print(f"\nDRY RUN — would reset {TASK} across {len(all_keys)} recording(s) "
              f"to NOT_RUN. Re-run without --dry-run to apply.")
    else:
        counts = pipeline_mgr.bulk_refresh([TASK], recording_keys=all_keys)
        print(f"\nReset {TASK}: {counts.get(TASK, 0)} record(s) -> NOT_RUN "
              f"across {len(all_keys)} recording(s).")

    print("\nRecording keys for --recordings:\n" + ",".join(all_keys))


if __name__ == "__main__":
    main()

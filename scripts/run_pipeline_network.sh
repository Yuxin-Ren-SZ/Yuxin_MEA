#!/usr/bin/env bash
# Run the full pipeline on ALL Network recordings, logging to a timestamped file.
#
#   bash scripts/run_pipeline_network.sh              # run
#   bash scripts/run_pipeline_network.sh --dry-run    # preview queue, no state change
#
# Cache skips finished tasks (complete + config unchanged). --recordings
# auto-queues any new Network wells before draining. Aggregate tasks run after
# the per-well drain (default on).
#
# NOTE: this does NOT refresh the dataset cache. If you added recordings under
# an already-cached Date dir, run scripts/refresh_cache.py first.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-pipeline_config.json}"
ENV="${ENV:-yuxin_mea}"
JOBS="${JOBS:-8}"
CACHE="$(conda run -n "$ENV" python -c "from pathlib import Path; from yuxin_mea.config import ConfigManager; from yuxin_mea.dataset.cache import CACHE_FILENAME; cm=ConfigManager(); cm.load(Path('$CONFIG')); print(Path(cm.get_global('analysis_root'))/CACHE_FILENAME)")"

KEYS="$(conda run -n "$ENV" python -c "import json; print(','.join(k for k in json.load(open('$CACHE')) if '/Network/' in k))")"
N=$(( $(grep -o ',' <<<"$KEYS" | wc -l) + 1 ))
echo "Network recordings queued: $N"

LOG="run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$ENV" yuxin-mea-run \
  --config "$CONFIG" \
  --recordings "$KEYS" \
  --jobs "$JOBS" \
  "$@" 2>&1 | tee "$LOG"

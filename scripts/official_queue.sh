#!/usr/bin/env bash
# Official-budget (250 epoch) runs on NAS-Bench-201, one at a time.
#   usage: SEED=0 DATASET=cifar10 bash scripts/official_queue.sh
# By default runs GDAS and uniform (SPOS) sampling, each without and with
# CASSO, for SEED. JOBS overrides the list with "seed:sampler:method" entries,
# e.g. JOBS="1:gdas:vanilla 1:gdas:casso".
# Safe to re-run (e.g. from cron): a second instance exits at once, finished
# runs are skipped, and unfinished runs resume from their per-epoch .ckpt.
# Paths: see --data_path/--autodl_root/--oracle in official_nb201.py (or set
# CASSO_DATA, AUTODL_ROOT, NB201_CACHE).
cd "$(dirname "$0")/.."
PY=${PY:-python3}; OUT=${OUT:-runs/official}; SEED=${SEED:-0}; DATASET=${DATASET:-cifar10}
JOBS=${JOBS:-"$SEED:gdas:vanilla $SEED:uniform:vanilla $SEED:gdas:casso $SEED:uniform:casso"}
mkdir -p "$OUT"
exec 9>"$OUT/.queue.lock"
flock -n 9 || exit 0
for job in $JOBS; do
  IFS=: read -r seed sampler method <<< "$job"
  tag="${DATASET}_s${seed}_${sampler}_${method}_e250"
  [ -f "$OUT/$tag.json" ] && continue
  echo "[$(date '+%F %T')] start  $tag"
  $PY scripts/official_nb201.py --dataset "$DATASET" --sampler "$sampler" --method "$method" \
      --rand_seed "$seed" --out "$OUT/$tag.json" >> "$OUT/$tag.stdout" 2>&1
  echo "[$(date '+%F %T')] finish $tag: $(grep -a DONE "$OUT/$tag.log")"
done

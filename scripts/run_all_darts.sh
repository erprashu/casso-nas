#!/bin/bash
# Sequentially runs the paper-faithful CASSO DARTS-space search across the
# 5 seeds used for Table 2 (Sec. 4.2.1: "five independent runs use random
# seeds {0,1,2,3,4}"). Waits for the given PID (typically the NAS-Bench-201
# queue orchestrator) before starting, since there is only one GPU.
set -uo pipefail
cd "$(dirname "$0")/.."

wait_for_pid() {
    local pid="$1"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
}

if [ -n "${1:-}" ]; then
    echo "[darts-queue] waiting for PID $1 to finish before starting (single GPU)..."
    wait_for_pid "$1"
fi

for seed in 0 1 2 3 4; do
    out="runs/full_darts_cifar10_seed${seed}.json"
    log="runs/full_darts_cifar10_seed${seed}.log"
    if [ -f "$out" ]; then
        echo "[darts-queue] $out already exists, skipping"
        continue
    fi
    echo "[darts-queue] launching DARTS-space seed $seed (epochs=50, warmup=15)"
    python3 scripts/search_darts.py --epochs 50 --warmup_epochs 15 --seed "$seed" \
        --eval_every_steps 200 --out "$out" > "$log" 2>&1
    echo "[darts-queue] finished seed $seed"
done

echo "[darts-queue] ALL DARTS RUNS COMPLETE"

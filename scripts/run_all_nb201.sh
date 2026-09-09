#!/bin/bash
# Sequentially runs the paper-faithful CASSO NAS-Bench-201 search across all
# 4 seeds (Sec. 4.2.1: "four random seeds {0,1,2,3}") for both CIFAR-10 and
# CIFAR-100 (ImageNet-16-120 deferred: we don't have that dataset locally).
# Runs strictly one job at a time since there is only one GPU.
set -uo pipefail
cd "$(dirname "$0")/.."

wait_for_pid() {
    local pid="$1"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
}

# seed 0 / cifar10 is assumed to already be running externally (PID passed
# as $1); if not given, launch it here too.
if [ -n "${1:-}" ]; then
    echo "[queue] waiting for externally-launched seed 0 / cifar10 (PID $1)..."
    wait_for_pid "$1"
else
    echo "[queue] launching seed 0 / cifar10 (epochs=193, warmup=15)"
    python3 scripts/search_nb201.py --epochs 193 --warmup_epochs 15 --dataset cifar10 --seed 0 \
        --eval_every_steps 500 --out runs/full_cifar10_seed0.json > runs/full_cifar10_seed0.log 2>&1
fi

for dataset in cifar10 cifar100; do
    for seed in 0 1 2 3; do
        if [ "$dataset" == "cifar10" ] && [ "$seed" == "0" ]; then
            continue  # already done above
        fi
        out="runs/full_${dataset}_seed${seed}.json"
        log="runs/full_${dataset}_seed${seed}.log"
        if [ -f "$out" ]; then
            echo "[queue] $out already exists, skipping"
            continue
        fi
        echo "[queue] launching $dataset seed $seed (epochs=193, warmup=15)"
        python3 scripts/search_nb201.py --epochs 193 --warmup_epochs 15 \
            --dataset "$dataset" --seed "$seed" --eval_every_steps 500 \
            --out "$out" > "$log" 2>&1
        echo "[queue] finished $dataset seed $seed"
    done
done

echo "[queue] ALL RUNS COMPLETE"

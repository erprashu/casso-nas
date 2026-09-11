#!/bin/bash
# Sequentially runs the paper-faithful CASSO NAS-Bench-201 search across all
# 4 seeds (Sec. 4.2.1: "four random seeds {0,1,2,3}") for both CIFAR-10 and
# CIFAR-100 (ImageNet-16-120 deferred: we don't have that dataset locally).
# Runs strictly one job at a time since there is only one GPU.
#
# Retries a failed/interrupted run up to 3 times before giving up on that
# seed and moving on -- safe to do now that search_nb201.py checkpoints
# itself, so a retry RESUMES from the last checkpoint rather than
# restarting from step 0. Added after an earlier version of this script
# silently moved on to the next seed whenever a run was killed early
# (no exit-code check at all), leaving several seeds with no valid result.
set -u
cd "$(dirname "$0")/.."

run_one() {
    local dataset="$1" seed="$2"
    local out="runs/full_${dataset}_seed${seed}.json"
    local log="runs/full_${dataset}_seed${seed}.log"

    if [ -f "$out" ]; then
        echo "[queue] $out already exists, skipping"
        return 0
    fi

    for attempt in 1 2 3; do
        echo "[queue] launching $dataset seed $seed (epochs=193, warmup=15, attempt $attempt/3)"
        python3 scripts/search_nb201.py --epochs 193 --warmup_epochs 15 \
            --dataset "$dataset" --seed "$seed" --eval_every_steps 500 \
            --kendall_every_steps 20000 \
            --out "$out" >> "$log" 2>&1
        if [ -f "$out" ]; then
            echo "[queue] finished $dataset seed $seed (attempt $attempt)"
            return 0
        fi
        echo "[queue] $dataset seed $seed attempt $attempt did NOT produce $out " \
             "(crashed or was interrupted) -- checkpoint should allow attempt $((attempt+1)) to resume"
    done
    echo "[queue] GIVING UP on $dataset seed $seed after 3 attempts -- check $log"
    return 1
}

for dataset in cifar10 cifar100; do
    for seed in 0 1 2 3; do
        run_one "$dataset" "$seed"
    done
done

echo "[queue] ALL RUNS COMPLETE"

# CASSO: Coverage-Aware Supernet Training via Sensitivity-Guided Optimization

Reference implementation of CASSO, a one-shot NAS method that models
multi-model forgetting in weight-sharing supernets as non-uniform across
depth, and uses this to drive a sensitivity-guided archive-selection and
regularization scheme.

Core components (paper section in parentheses):

- `casso/sensitivity.py` -- SNIP-based saliency, depth/sharing-aware
  sensitivity weighting, and sensitivity distance (Sec. 3.3).
- `casso/archive.py` -- streaming facility-location archive selection
  (Algorithm 1, Sec. 3.4).
- `casso/losses.py` -- Multi-Model Regularized Loss: replay, EMA-distillation
  stability, and cross-architecture KL consistency (Sec. 3.5).
- `casso/train_search.py`, `casso/train_search_darts.py` -- the full search
  loop (Algorithm 2) for the NAS-Bench-201 and DARTS search spaces
  respectively.
- `casso/nb201/`, `casso/darts/` -- supernet, cell, and genotype-derivation
  code for each search space.

Built with reference to the general structure of
[quark0/darts](https://github.com/quark0/darts) and
[miaozhang0525/NSAS_FOR_CVPR](https://github.com/miaozhang0525/NSAS_FOR_CVPR)
(the official code for the closest prior work); no code is copied from
either, and every module here follows this paper's own equations.

## Layout

```
casso/
  config.py             default hyperparameters
  ops.py                candidate operations (shared by both search spaces)
  genotypes.py          PRIMITIVES per search space
  sensitivity.py        SNIP saliency, sensitivity weighting/distance
  archive.py            streaming facility-location archive (Algorithm 1)
  losses.py             MMLF loss + EMA teacher
  checkpoint.py         training checkpoint save/resume
  utils.py              data loading, seeding, meters
  train_search.py       search loop, NAS-Bench-201 space
  train_search_darts.py search loop, DARTS space
  nb201/                NAS-Bench-201 supernet + benchmark API wrapper
  darts/                DARTS supernet, cell, and eval-network definitions
tests/                  unit tests + an end-to-end smoke test
scripts/                CLI entry points (search + retrain)
```

## Running

```bash
# Unit tests (fast, no GPU needed)
python3 -m pytest tests/ -v

# End-to-end smoke test (synthetic data)
python3 tests/test_smoke_nb201.py

# One-time: extract the NAS-Bench-201 ground truth used by the search and
# evaluation scripts into a small JSON cache (~2 MB)
python3 scripts/build_oracle_cache.py --pkl <path>/nasbench201_v1_0-e61699.pkl \
    --out <path>/nb201_test_acc_cache.json

# Search on NAS-Bench-201 (paper protocol: seeds 0-3)
python3 scripts/search_nb201.py --dataset cifar10 --seed 0 --out runs/run.json

# Inherited-weight ranking fidelity of a saved search checkpoint (Table 5):
# 200 architectures sampled per seed, BatchNorm re-estimated per path,
# Kendall-tau against NAS-Bench-201 ground truth
python3 scripts/eval_checkpoint_ranking.py --ckpt runs/run.ckpt \
    --dataset cifar10 --seed 0 --out runs/eval_seed0.json

# Search on the DARTS space, then retrain the discovered genotype
python3 scripts/search_darts.py --seed 0 --out runs/darts_run.json
python3 scripts/retrain_darts.py --genotype runs/darts_run.json
```

## Data dependencies

- NAS-Bench-201 benchmark file (`nasbench201_v1_0-e61699.pkl`). The search and
  evaluation scripts read the compact cache written by
  `scripts/build_oracle_cache.py`; loading the full file instead peaks at
  about 16 GB of RAM per process (`casso/nb201/api_wrapper.py` accepts either).
- CIFAR-10/CIFAR-100, loaded via `casso/utils.py`; a HuggingFace parquet
  mirror path can be passed with `--hf_parquet_dir`, falling back to
  torchvision's own download otherwise.

Neither dataset is bundled in this repository.

## Requirements

PyTorch with CUDA support, `scipy`, `nas_201_api`. See `casso/config.py`
for the exact hyperparameter defaults used in the paper's experiments.

## Changes

- The single-path GDAS forward pass (`casso/nb201/cell.py`,
  `casso/darts/cell.py`) now adds the straight-through weights of the
  non-selected operations to each edge, as in the reference GDAS
  implementation. This leaves the forward value unchanged and routes gradient
  to every architecture logit. The checkpoints evaluated for Table 5 were
  trained before this change.

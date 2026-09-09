# CASSO: Coverage-Aware Supernet Optimization

Reimplementation of the CASSO one-shot NAS framework (see `../ASOC-D-26-05728 (2).pdf`
and `../Engg_Casso_NAS/IEEEtran/april'26_casso_nas.tex` for the paper), written
from scratch after the original implementation was lost to a hard-drive
failure. Built with reference to the general structure of
[quark0/darts](https://github.com/quark0/darts) and
[miaozhang0525/NSAS_FOR_CVPR](https://github.com/miaozhang0525/NSAS_FOR_CVPR)
(the official code for Zhang et al. 2020, the closest prior work), but no
code was copied from either -- every module here is a fresh implementation
of the paper's equations, cross-checked with unit tests.

## Status (see `git log` for the detailed history)

**Working and tested:**
- Core algorithm modules: `casso/sensitivity.py` (Eq. 7-9), `casso/archive.py`
  (Algorithm 1), `casso/losses.py` (Eq. 10-11) -- 24 unit tests, all passing,
  each cross-validated against an independent brute-force or hand-derived
  reference rather than just "doesn't crash."
- `casso/nb201/` -- NAS-Bench-201 supernet + GDAS-style single-path
  Gumbel-softmax cell. Smoke-tested end-to-end; a *real* validation run on
  actual CIFAR-10 (not synthetic data) is what caught the KL-term bug
  below.
- `casso/darts/` -- standard DARTS cell/supernet + genotype derivation
  (top-2-edges-per-node "Magnitude" selection, matching Table 2). Forward
  pass smoke-tested at both toy and paper scale (16 channels, 8 layers,
  real 32x32 input).
- Real NAS-Bench-201 benchmark data (`data/nasbench201/`) and a fast
  CIFAR-10 mirror (`data/cifar10_hf/`) are downloaded and verified against
  the official `nas_201_api`.
- GPU: RTX 5090 (sm_120/Blackwell), `torch==2.11.0+cu128`. **Do not
  `pip install torch` unversioned** -- pip will silently keep whatever CPU
  build is already installed even when pointed at a CUDA index, since it
  considers the unversioned requirement already satisfied. Use
  `pip install --force-reinstall torch torchvision --index-url
  https://download.pytorch.org/whl/cu128` if this ever regresses.

**Not yet built:** a DARTS-space equivalent of `CASSOSearcher` (currently
NB201-only), the retrain-from-scratch pipeline needed for Table 2's
numbers, ImageNet transfer (explicitly deferred), and any ablation-study
scripts.

## Bugs found so far by actually running the code

Every one of these was caught by a test or a real run, not by inspection.
Recorded here so the pattern (and the value of testing before trusting) is
visible to future-me:

1. **`Zero` op channel mismatch** (`ops.py`) -- the "none" operation
   returned zeros shaped like its *input*, not the edge's target `c_out`,
   breaking node summation whenever a cell's input channel count differs
   from its output channel count (e.g. the first search cell of a stage).
2. **`skip_connect` misrouted through `FactorizedReduce`** (`ops.py`) --
   when a channel projection was needed at `stride=1`, the code used
   `FactorizedReduce`, which hardcodes `stride=2` internally, silently
   downsampling spatial resolution when it shouldn't.
3. **Sensitivity forward-signature mismatch** (`sensitivity.py`) --
   `compute_snip_saliency` assumed `supernet(x)`, but a single-path
   supernet's forward requires `(x, hardwts, indices)`. Fixed by taking a
   `forward_fn` closure instead.
4. **Archive: new-candidate `g` seeded at 0.0** (`archive.py`) -- should be
   the candidate's true best-coverage against the *current* archive
   members (Eq. 5's `max` term), not a placeholder. Seeding at 0
   systematically overstated every new candidate's own marginal gain.
5. **Archive: `least_contributing_member` included the incoming candidate
   itself** (`archive.py`) -- the paper's $S_t$ is explicitly the stream of
   *previously* sampled architectures, excluding the one currently being
   offered. Including it let a near-worthless near-duplicate candidate's
   trivial self-assignment tip which *existing* (possibly valuable) member
   got evicted.
6. **KL term compared predictions on different inputs** (`losses.py`,
   `train_search.py`) -- the active architecture's logits on the *main*
   training batch were compared against archived architectures' logits on
   their own, differently-sized cached replay batches. Eq. 10 requires
   both distributions evaluated on the *same* replayed inputs; fixed by
   also running the active architecture on each archived sample's replay
   batch.
7. **PyTorch was silently CPU-only** despite a physical RTX 5090 being
   present -- see the GPU note above.

## Layout

```
casso/
  config.py       default hyperparameters (paper Sec. 3.5, Table 5/6 defaults)
  ops.py          candidate operation primitives (shared by both search spaces)
  genotypes.py    PRIMITIVES lists per search space
  sensitivity.py  Eq. 7-9: SNIP saliency, depth/sharing weight, distance
  archive.py      Algorithm 1: streaming facility-location archive
  losses.py       Eq. 10-11: MMLF, EMA teacher
  utils.py        data loading, seeding, meters
  train_search.py Algorithm 2 (currently NB201-specialized: CASSOSearcher)
  nb201/          NAS-Bench-201 search space + benchmark API wrapper
  darts/          standard DARTS search space
tests/            pytest unit tests + an end-to-end smoke test
scripts/          CLI entry points (search_nb201.py so far)
runs/             experiment logs/outputs (gitignored contents TBD)
```

## Running things

```bash
# Unit tests (fast, no GPU needed)
python3 -m pytest tests/test_sensitivity.py tests/test_archive.py tests/test_losses.py -v

# End-to-end smoke test (synthetic data, ~1s)
python3 tests/test_smoke_nb201.py

# Real search on NAS-Bench-201 / CIFAR-10 (needs the GPU + downloaded data)
python3 scripts/search_nb201.py --epochs 1 --dataset cifar10 --seed 0 \
    --out runs/my_run.json
```

## Data dependencies (already downloaded on this machine)

- `data/nasbench201/nasbench201_v1_0-e61699.pkl` -- NAS-Bench-201 benchmark
  (MIT-licensed mirror of the official release, via
  `ThunderStruct/NASBench` on HuggingFace; the official Google Drive link
  is not directly fetchable and a separate `v1_1` HF mirror turned out to
  be an empty placeholder repo).
- `data/cifar10_hf/plain_text/{train,test}-00000-of-00001.parquet` -- CIFAR-10
  via the `uoft-cs/cifar10` HuggingFace mirror. Torchvision's own download
  source was observed to be throttled to ~110 KB/s in this environment
  (~25 min for 170MB); this mirror loads in under a second.

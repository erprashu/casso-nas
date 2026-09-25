"""Ground-truth lookups for NAS-Bench-201 (Sec. 4.4, Table 4; Sec. 4.5.2,
Kendall-tau ranking-fidelity evaluation).

Two backends, same interface:
- a .json cache built once by scripts/build_oracle_cache.py (arch string by
  index + test accuracy per dataset; ~2MB). Preferred: the full benchmark
  pickle peaks at ~16GB RSS to load, so several concurrent runs each loading
  it exhausted system RAM and triggered the kernel OOM killer.
- the full benchmark .pkl, via the official nas_201_api. We bypass
  torch.load's weights_only=True restriction (PyTorch 2.6+) by unpickling
  with the plain `pickle` module and handing the dict to NASBench201API.
"""

import json
import pickle
from functools import lru_cache
from typing import Optional


class NB201Oracle:
    def __init__(self, path: str):
        self.api = None
        self._archs = None
        self._test_acc = None
        if path.endswith(".json"):
            with open(path) as f:
                cache = json.load(f)
            self._archs = cache["archs"]
            self._test_acc = cache["test_accuracy"]
            self._index = {a: i for i, a in enumerate(self._archs)}
        else:
            from nas_201_api import NASBench201API
            with open(path, "rb") as f:
                raw = pickle.load(f)
            self.api = NASBench201API(raw, verbose=False)

    def __len__(self):
        return len(self._archs) if self._archs is not None else len(self.api)

    def _arch(self, idx: int) -> str:
        return self._archs[idx] if self._archs is not None else self.api.arch(idx)

    @lru_cache(maxsize=None)
    def test_accuracy(self, arch_str: str, dataset: str, hp: str = "200") -> Optional[float]:
        """Ground-truth test accuracy (%) for an architecture string, matching
        the format produced by NB201Supernet.genotype_string()."""
        if self._archs is not None:
            assert hp == "200", "cache only stores hp='200' results"
            idx = self._index.get(arch_str)
            return None if idx is None else self._test_acc[dataset][idx]
        try:
            idx = self.api.query_index_by_arch(arch_str)
        except Exception:
            return None
        if idx is None or idx < 0:
            return None
        info = self.api.get_more_info(idx, dataset, hp=hp, is_random=False)
        return info.get("test-accuracy")

    def random_arch_strings(self, n: int, seed: int) -> list:
        import random
        rng = random.Random(seed)
        idxs = rng.sample(range(len(self)), n)
        return [self._arch(i) for i in idxs]

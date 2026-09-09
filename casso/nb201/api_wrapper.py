"""Thin wrapper around the official nas_201_api for ground-truth lookups
(Sec. 4.4, Table 4; Sec. 4.5.2, Kendall-tau ranking-fidelity evaluation).

We bypass torch.load's default weights_only=True restriction (introduced in
PyTorch 2.6) by pre-unpickling the benchmark file ourselves with the plain
`pickle` module (verified to load correctly against the downloaded file)
and handing the resulting dict to NASBench201API, which accepts either a
path or an already-loaded dict.
"""

import pickle
from functools import lru_cache
from typing import Optional

from nas_201_api import NASBench201API


class NB201Oracle:
    def __init__(self, pickle_path: str):
        with open(pickle_path, "rb") as f:
            raw = pickle.load(f)
        self.api = NASBench201API(raw, verbose=False)

    def __len__(self):
        return len(self.api)

    @lru_cache(maxsize=None)
    def test_accuracy(self, arch_str: str, dataset: str, hp: str = "200") -> Optional[float]:
        """Ground-truth test accuracy (%) for an architecture string, matching
        the format produced by NB201Supernet.genotype_string()."""
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
        idxs = rng.sample(range(len(self.api)), n)
        return [self.api.arch(i) for i in idxs]

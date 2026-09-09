"""Coverage-aware archive selection (paper Sec. 3.3, Eq. 5, Algorithm 1).

F(M) = sum_{alpha in S_t} max_{beta in M} kappa(alpha) * sim(alpha, beta)
sim(alpha, beta) = exp(-d(alpha, beta) / tau)

We implement the streaming "online greedy with replacement" variant
(Algorithm 1): O(1) scalar caches g(alpha'), beta*(alpha') are kept per
previously-seen architecture -- NOT raw architectures/weights -- matching
the complexity analysis added to the manuscript.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional


def similarity(distance: float, tau: float) -> float:
    return math.exp(-distance / tau)


@dataclass
class ArchiveMember:
    key: Hashable
    kappa: float
    payload: object  # e.g. (architecture genotype, inherited weights, node list)


class StreamingFacilityLocationArchive:
    """Algorithm 1: Streaming Sensitivity-Facility Location (SFL) for M."""

    def __init__(self, budget: int, distance_fn: Callable[[Hashable, Hashable], float],
                 tau: float = 1.0):
        self.budget = budget
        self.distance_fn = distance_fn
        self.tau = tau
        self.members: Dict[Hashable, ArchiveMember] = {}
        # O(1)-per-element scalar caches over the full stream S_t (Sec. 3.3
        # complexity paragraph): g(alpha') and its best-serving archive key.
        self._g: Dict[Hashable, float] = {}
        self._beta_star: Dict[Hashable, Optional[Hashable]] = {}
        self._stream_kappa: Dict[Hashable, float] = {}

    def __len__(self):
        return len(self.members)

    def _sim(self, a: Hashable, b: Hashable) -> float:
        return similarity(self.distance_fn(a, b), self.tau)

    def _recompute_g_against(self, key: Hashable, kappa: float) -> None:
        """Update every cached g(alpha') given a new/changed archive member `key`."""
        for other_key, other_kappa in self._stream_kappa.items():
            candidate = other_kappa * self._sim(other_key, key)
            if candidate > self._g.get(other_key, -math.inf):
                self._g[other_key] = candidate
                self._beta_star[other_key] = key

    def _full_g_recompute(self) -> None:
        """Recompute g(.) and beta*(.) from scratch against the current
        archive (used only after a removal, since removing a member can only
        ever *decrease* some g values -- an O(|M|*|S_t|) operation, still
        far cheaper than retaining full architecture histories)."""
        member_keys = list(self.members.keys())
        for other_key, other_kappa in self._stream_kappa.items():
            best_val, best_key = -math.inf, None
            for mk in member_keys:
                val = other_kappa * self._sim(other_key, mk)
                if val > best_val:
                    best_val, best_key = val, mk
            self._g[other_key] = 0.0 if best_key is None else best_val
            self._beta_star[other_key] = best_key

    def marginal_gain(self, key: Hashable, kappa: float,
                       excluding: Optional[Hashable] = None) -> float:
        """Delta F(alpha | M) or Delta F(alpha | M \\ {excluding}) (Alg. 1, line 3/7)."""
        total = 0.0
        member_keys = [k for k in self.members if k != excluding]
        for other_key, other_kappa in self._stream_kappa.items():
            g_val = self._g.get(other_key, 0.0)
            if excluding is not None and self._beta_star.get(other_key) == excluding:
                # g(.) may be stale w.r.t. removal; recompute against the
                # reduced member set for this one comparison.
                g_val = max(
                    (other_kappa * self._sim(other_key, mk) for mk in member_keys),
                    default=0.0,
                )
            candidate = other_kappa * self._sim(other_key, key)
            total += max(g_val, candidate) - g_val
        return total

    def least_contributing_member(self) -> Hashable:
        """arg min_{beta in M} sum_{alpha' in S_t} 1{beta*(alpha')=beta} g(alpha')
        (Alg. 1, line 6)."""
        contribution: Dict[Hashable, float] = {k: 0.0 for k in self.members}
        for other_key, best_key in self._beta_star.items():
            if best_key in contribution:
                contribution[best_key] += self._g.get(other_key, 0.0)
        return min(contribution, key=contribution.get)

    def offer(self, key: Hashable, kappa: float, payload: object) -> bool:
        """Offer a new architecture to the archive; returns True if accepted."""
        self._stream_kappa[key] = kappa
        self._g.setdefault(key, 0.0)
        self._beta_star.setdefault(key, None)

        if len(self.members) < self.budget:
            self.members[key] = ArchiveMember(key, kappa, payload)
            self._recompute_g_against(key, kappa)
            return True

        least_key = self.least_contributing_member()
        gain = self.marginal_gain(key, kappa, excluding=least_key)
        if gain > 0:
            del self.members[least_key]
            self.members[key] = ArchiveMember(key, kappa, payload)
            self._full_g_recompute()
            return True
        return False

    def payloads(self) -> List[object]:
        return [m.payload for m in self.members.values()]

    def facility_location_value(self) -> float:
        """F(M) for diagnostics / Table 13 comparison."""
        total = 0.0
        for other_key, other_kappa in self._stream_kappa.items():
            best = max((other_kappa * self._sim(other_key, mk) for mk in self.members), default=0.0)
            total += best
        return total


def offline_greedy_archive(candidates: List[Hashable], kappas: Dict[Hashable, float],
                            distance_fn: Callable[[Hashable, Hashable], float],
                            budget: int, tau: float = 1.0) -> List[Hashable]:
    """Offline (1 - 1/e)-approximate greedy maximization of F(M) over a fixed
    ground set (used only as the Table 13 reference point, not during
    training)."""
    selected: List[Hashable] = []

    def f_value(members: List[Hashable]) -> float:
        total = 0.0
        for c in candidates:
            best = max((kappas[c] * similarity(distance_fn(c, m), tau) for m in members), default=0.0)
            total += best
        return total

    remaining = list(candidates)
    current_value = 0.0
    for _ in range(min(budget, len(candidates))):
        best_gain, best_cand = -math.inf, None
        for cand in remaining:
            trial_value = f_value(selected + [cand])
            gain = trial_value - current_value
            if gain > best_gain:
                best_gain, best_cand = gain, cand
        selected.append(best_cand)
        remaining.remove(best_cand)
        current_value = f_value(selected)
    return selected

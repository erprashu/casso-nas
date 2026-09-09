"""Unit tests for casso/archive.py (Eq. 5, Algorithm 1).

Strategy: rather than hand-deriving expected numbers for the exponential
similarity kernel (error-prone to do by hand), we cross-validate the
streaming implementation's O(1)-cache bookkeeping against a naive,
obviously-correct brute-force recomputation of F(M) = sum_alpha max_beta
kappa(alpha) sim(alpha, beta) directly from Eq. 5, after every step.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casso.archive import (  # noqa: E402
    StreamingFacilityLocationArchive,
    offline_greedy_archive,
    similarity,
)


def brute_force_f(stream_kappa: dict, members: set, dist_fn, tau: float) -> float:
    total = 0.0
    for alpha, kappa in stream_kappa.items():
        best = max((kappa * similarity(dist_fn(alpha, beta), tau) for beta in members), default=0.0)
        total += best
    return total


def dist(a, b):
    return abs(a - b)


def test_fills_to_budget_without_replacement_logic():
    arch = StreamingFacilityLocationArchive(budget=3, distance_fn=dist, tau=1.0)
    for k in [0.0, 5.0, 10.0]:
        accepted, evicted = arch.offer(k, kappa=1.0, payload=k)
        assert accepted is True
        assert evicted == [], "stream is tiny; nothing should age out of the default window"
    assert len(arch) == 3
    assert set(arch.members.keys()) == {0.0, 5.0, 10.0}


def test_stream_window_bounds_memory_without_disturbing_members():
    """The sliding window (added after a real 150k-step run was observed to
    leak ~17GB+ RSS from unboundedly growing _stream_kappa/_g/_beta_star)
    must evict old NON-member stream points once the window fills, while
    never evicting a current archive member regardless of arrival order.

    Tests _evict_stale() directly against hand-seeded bookkeeping, rather
    than driving it through offer()'s full accept/replace decision -- doing
    the latter with deliberately-mismatched kappa values was found to
    trigger a real (and separately documented) property of the paper's own
    Algorithm 1 acceptance rule, which would have made this test about the
    wrong thing entirely."""
    window = 5
    arch = StreamingFacilityLocationArchive(budget=2, distance_fn=dist, tau=1.0,
                                             stream_window=window)
    # Seed two members directly (bypassing offer()'s decision logic; we
    # only want to test _evict_stale() here).
    from casso.archive import ArchiveMember
    arch.members[0.0] = ArchiveMember(0.0, 100.0, 0.0)
    arch.members[1000.0] = ArchiveMember(1000.0, 100.0, 1000.0)
    for k in (0.0, 1000.0):
        arch._stream_kappa[k] = 100.0
        arch._arrival_order.append(k)
        arch._g[k] = 100.0
        arch._beta_star[k] = k

    all_evicted = []
    for i in range(2, 30):
        arch._stream_kappa[float(i)] = 1.0
        arch._arrival_order.append(float(i))
        arch._g[float(i)] = 1.0
        arch._beta_star[float(i)] = float(i)
        all_evicted.extend(arch._evict_stale())

    # The two members must never have been evicted, even though they are
    # by far the OLDEST entries in arrival order.
    assert 0.0 not in all_evicted
    assert 1000.0 not in all_evicted
    assert set(arch.members.keys()) == {0.0, 1000.0}

    # Tracked stream size must stay bounded (window + however many members
    # happen to also still be within the window), not grow with total offers.
    assert len(arch._stream_kappa) <= window + len(arch.members)
    assert len(all_evicted) > 0, "with 28 extra offers against a window of 5, evictions must occur"


def test_facility_location_value_matches_brute_force_after_every_offer():
    tau = 2.0
    arch = StreamingFacilityLocationArchive(budget=2, distance_fn=dist, tau=tau)
    stream_kappa = {}

    for point, kappa in [(0.0, 1.0), (10.0, 1.0), (20.0, 1.0), (30.0, 1.0), (5.0, 3.0)]:
        arch.offer(point, kappa=kappa, payload=point)
        stream_kappa[point] = kappa
        expected = brute_force_f(stream_kappa, set(arch.members.keys()), dist, tau)
        actual = arch.facility_location_value()
        assert actual == pytest.approx(expected, rel=1e-6), (
            f"F(M) mismatch after offering {point}: archive={set(arch.members.keys())}"
        )


def test_replacement_only_happens_when_marginal_gain_positive():
    """A near-duplicate of an existing high-value member, itself carrying
    negligible kappa, should never be admitted purely on the strength of
    imitating something already well-covered -- unless it genuinely
    increases F(M)."""
    tau = 1.0
    arch = StreamingFacilityLocationArchive(budget=2, distance_fn=dist, tau=tau)
    arch.offer(0.0, kappa=10.0, payload=0.0)
    arch.offer(100.0, kappa=10.0, payload=100.0)
    f_before = arch.facility_location_value()
    members_before = set(arch.members.keys())

    # An exact duplicate of an existing member contributes ZERO marginal
    # gain (it can't cover anything better than the original already does).
    accepted, _ = arch.offer(0.0 + 1e-9, kappa=0.001, payload="near-dup")
    f_after = arch.facility_location_value()

    if accepted:
        assert f_after >= f_before - 1e-9, "an accepted offer must not decrease F(M)"
    else:
        assert set(arch.members.keys()) == members_before


def test_marginal_gain_matches_brute_force_difference():
    tau = 1.5
    arch = StreamingFacilityLocationArchive(budget=3, distance_fn=dist, tau=tau)
    stream_kappa = {}
    for point, kappa in [(0.0, 2.0), (10.0, 1.0), (20.0, 1.5)]:
        arch.offer(point, kappa=kappa, payload=point)
        stream_kappa[point] = kappa

    members = set(arch.members.keys())
    f_before = brute_force_f(stream_kappa, members, dist, tau)

    candidate, cand_kappa = 15.0, 1.0
    stream_kappa_with_candidate = dict(stream_kappa)
    stream_kappa_with_candidate[candidate] = cand_kappa

    # marginal_gain() assumes the candidate has already been registered
    # into the stream (offer() does this at its very top, before deciding
    # whether to accept); replicate that here so this direct call matches
    # real usage rather than testing an unsupported calling convention.
    arch._stream_kappa[candidate] = cand_kappa
    arch._init_stream_point(candidate, cand_kappa)

    # Try replacing each current member with the candidate; the archive's
    # own marginal_gain(..., excluding=m) should match the brute-force
    # difference for that same hypothetical swap.
    for m in members:
        trial_members = (members - {m}) | {candidate}
        f_after = brute_force_f(stream_kappa_with_candidate, trial_members, dist, tau)
        f_before_reduced = brute_force_f(stream_kappa_with_candidate, members - {m}, dist, tau)
        expected_gain = f_after - f_before_reduced
        actual_gain = arch.marginal_gain(candidate, cand_kappa, excluding=m)
        assert actual_gain == pytest.approx(expected_gain, rel=1e-5, abs=1e-8), (
            f"marginal_gain mismatch when excluding {m}"
        )


def test_offline_greedy_beats_or_matches_random_selection():
    """Sanity check on the (1-1/e)-approximate offline reference (Table 13's
    upper bound): greedy selection should never do worse than an arbitrary
    fixed subset of the same size."""
    tau = 3.0
    candidates = [float(x) for x in range(0, 50, 5)]
    kappas = {c: 1.0 + (c % 7) for c in candidates}  # non-uniform importance

    greedy = offline_greedy_archive(candidates, kappas, dist, budget=3, tau=tau)
    f_greedy = brute_force_f(kappas, set(greedy), dist, tau)

    arbitrary = set(candidates[:3])  # first 3 candidates, un-optimized
    f_arbitrary = brute_force_f(kappas, arbitrary, dist, tau)

    assert f_greedy >= f_arbitrary - 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

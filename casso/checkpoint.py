"""Checkpoint save/resume for CASSOSearcher, added after repeated container-
level restarts killed multi-hour runs at 89%-94% complete (tmux itself did
not survive; only saving/resuming actual training state can protect
against that). Not part of the paper -- purely operational infrastructure.

We do NOT pickle the archive object whole (its distance_fn is a bound
method closing over the live searcher/net, which does not survive a fresh
process). Instead we extract the archive's plain-data state (members,
caches, arrival order) and restore it onto a freshly-constructed archive
(whose distance_fn is correctly re-bound to the new searcher instance).
"""

import os
from typing import Optional

import torch


def save_checkpoint(path: str, searcher, step: int) -> None:
    tmp_path = path + ".tmp"
    archive = searcher.archive
    state = {
        "step": step,
        "net_state": searcher.net.state_dict(),
        "w_optimizer_state": searcher.w_optimizer.state_dict(),
        "w_scheduler_state": searcher.w_scheduler.state_dict(),
        "phi_optimizer_state": searcher.phi_optimizer.state_dict(),
        "ema_shadow": searcher.ema.shadow,
        # Archive plain-data state (NOT the archive object itself -- its
        # distance_fn is a bound method that won't survive pickling/reload).
        "archive_members": {k: (v.kappa, v.payload) for k, v in archive.members.items()},
        "archive_g": dict(archive._g),
        "archive_beta_star": dict(archive._beta_star),
        "archive_stream_kappa": dict(archive._stream_kappa),
        "archive_arrival_order": list(archive._arrival_order),
        "archive_payloads": dict(searcher.archive_payloads),
        "next_archive_id": searcher._next_archive_id,
        "sharing_count": dict(searcher.sharing_count),
        "s_bar": dict(searcher.s_bar),
        "variance": dict(searcher.variance),
        "omega": dict(searcher.omega),
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # atomic on the same filesystem: never leaves a half-written checkpoint


def load_checkpoint(path: str, searcher) -> Optional[int]:
    """Restores `searcher` in place from `path`; returns the step to resume
    from, or None if no checkpoint exists at `path`."""
    if not os.path.exists(path):
        return None

    state = torch.load(path, map_location=searcher.device, weights_only=False)

    searcher.net.load_state_dict(state["net_state"])
    searcher.w_optimizer.load_state_dict(state["w_optimizer_state"])
    searcher.w_scheduler.load_state_dict(state["w_scheduler_state"])
    searcher.phi_optimizer.load_state_dict(state["phi_optimizer_state"])
    searcher.ema.shadow = state["ema_shadow"]

    from .archive import ArchiveMember
    archive = searcher.archive
    archive.members = {
        k: ArchiveMember(k, kappa, payload) for k, (kappa, payload) in state["archive_members"].items()
    }
    archive._g = state["archive_g"]
    archive._beta_star = state["archive_beta_star"]
    archive._stream_kappa = state["archive_stream_kappa"]
    from collections import deque
    archive._arrival_order = deque(state["archive_arrival_order"])

    searcher.archive_payloads = state["archive_payloads"]
    searcher._next_archive_id = state["next_archive_id"]
    searcher.sharing_count = state["sharing_count"]
    searcher.s_bar = state["s_bar"]
    searcher.variance = state["variance"]
    searcher.omega = state["omega"]

    return state["step"]

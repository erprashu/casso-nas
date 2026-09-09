"""Algorithm 2 (CASSO NAS, single-path GDAS-style), specialized to the
standard DARTS search space (Sec. 3.7, Sec. 4.2.1). Mirrors
train_search.py's CASSOSearcher but accounts for the DARTS cell's two
architecture-parameter tensors (normal_logits, reduce_logits) and its
variable in-degree-per-node topology.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .archive import StreamingFacilityLocationArchive
from .config import CASSOConfig
from .darts.supernet import DARTSSupernet
from .losses import EMATeacher, MMLFLoss
from .sensitivity import (
    compute_depth_sharing_weight,
    compute_snip_saliency,
    layerwise_sensitivity_weight,
    sensitivity_distance,
)
from .train_search import cosine_temperature  # shared cosine Gumbel-temperature schedule
from .utils import accuracy


@dataclass
class DARTSArchivedSample:
    n_idx: torch.Tensor
    r_idx: torch.Tensor
    node_keys: list
    kappa: float
    replay_batch: tuple


@dataclass
class DARTSSearchStats:
    step_losses: List[float] = field(default_factory=list)


class DARTSCASSOSearcher:
    def __init__(self, supernet: DARTSSupernet, cfg: CASSOConfig, device: torch.device,
                 total_steps: int, warmup_steps: int):
        self.net = supernet.to(device)
        self.cfg = cfg
        self.device = device
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        # DARTS space has no separate "stage" grouping (Sec. 4.2.1); depth is
        # simply the cell's position in the single stack of `layers` cells.
        self.num_stages = 1
        self.cells_per_stage = supernet.total_positions

        # Paper-exact optimizer settings (Sec. 4.2.1).
        self.w_optimizer = torch.optim.SGD(
            [p for n, p in self.net.named_parameters() if "logits" not in n],
            lr=0.025, momentum=0.9, weight_decay=0.0003,
        )
        self.w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer, T_max=total_steps
        )
        self.phi_optimizer = torch.optim.Adam(
            [self.net.normal_logits, self.net.reduce_logits],
            lr=6e-4, betas=(0.5, 0.999), weight_decay=1e-3,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.mmlf = MMLFLoss(cfg.beta, cfg.gamma, cfg.eta, cfg.weight_decay)
        self.ema = EMATeacher(self.net, cfg.ema_decay)

        self.archive = StreamingFacilityLocationArchive(
            budget=cfg.archive_size, distance_fn=self._archive_distance, tau=cfg.sim_temperature,
            stream_window=cfg.stream_window,
        )
        self.archive_payloads: Dict[int, DARTSArchivedSample] = {}
        self._next_archive_id = 0

        self.s_bar: Dict = {}
        self.variance: Dict = {}
        self.omega: Dict = {}
        self.sharing_count: Dict = {}

        id_to_layer: Dict[int, int] = {}
        for (_, _, cell_position), params in self.net.all_node_param_map().items():
            for p in params:
                id_to_layer[id(p)] = cell_position
        self._param_name_to_layer: Dict[str, int] = {}
        for name, p in self.net.named_parameters():
            layer = id_to_layer.get(id(p))
            if layer is not None:
                self._param_name_to_layer[name] = layer

        self.stats = DARTSSearchStats()

    def _node_param_map(self):
        return self.net.all_node_param_map()

    def refresh_sensitivity(self, minibatches, tau: float = None):
        tau = tau if tau is not None else self.cfg.gumbel_tau_min

        def forward_fn(x):
            n_hw, n_idx, r_hw, r_idx = self.net.sample_architecture(tau)
            return self.net(x, n_hw, n_idx, r_hw, r_idx)

        s_bar, variance = compute_snip_saliency(
            self.net, minibatches, self._node_param_map, self.criterion, forward_fn=forward_fn
        )
        self.s_bar, self.variance = s_bar, variance
        self.omega = compute_depth_sharing_weight(
            self.sharing_count, self.cells_per_stage, self.num_stages, self.cfg.depth_rho
        )

    def _archive_distance(self, key_a: int, key_b: int) -> float:
        nodes_a = self.archive_payloads[key_a].node_keys
        nodes_b = self.archive_payloads[key_b].node_keys
        return sensitivity_distance(nodes_a, nodes_b, self.s_bar, self.variance,
                                     self.omega, self.cfg.sensitivity_eps)

    def _kappa(self, node_keys) -> float:
        return sum(self.omega.get(u, 0.0) * self.s_bar.get(u, 0.0) for u in node_keys)

    def _layer_of_node(self, node_key):
        return node_key[2]

    def step(self, t: int, train_iter, val_iter, sensitivity_batches=None) -> Dict[str, float]:
        tau = cosine_temperature(t, self.total_steps, self.cfg.gumbel_tau_init, self.cfg.gumbel_tau_min)
        n_hw, n_idx, r_hw, r_idx = self.net.sample_architecture(tau)

        node_keys = self.net.active_node_keys(n_idx, r_idx)
        for u in node_keys:
            self.sharing_count[u] = self.sharing_count.get(u, 0) + 1

        if t > self.warmup_steps and t % self.cfg.refresh_interval == 0 and sensitivity_batches:
            self.refresh_sensitivity(sensitivity_batches)

        x, y = next(train_iter)
        x, y = x.to(self.device), y.to(self.device)

        active_logits = self.net(x, n_hw, n_idx, r_hw, r_idx)
        active_params = self.net.active_params(n_idx, r_idx)

        archived_logits, archived_targets, archived_params_list = [], [], []
        active_on_replay_logits = []
        f_bar: Dict[int, float] = {}
        if self.archive.members and self.omega:
            archive_nodes = [self.archive_payloads[k].node_keys for k in self.archive.members]
            f_bar = layerwise_sensitivity_weight(archive_nodes, self.omega, self.s_bar,
                                                  self._layer_of_node)
            for key in list(self.archive.members)[:min(3, len(self.archive.members))]:
                sample = self.archive_payloads[key]
                ax, ay = sample.replay_batch
                ax, ay = ax.to(self.device), ay.to(self.device)

                a_n_hw = torch.zeros_like(n_hw)
                for e in range(a_n_hw.shape[0]):
                    a_n_hw[e, sample.n_idx[e]] = 1.0
                a_r_hw = torch.zeros_like(r_hw)
                for e in range(a_r_hw.shape[0]):
                    a_r_hw[e, sample.r_idx[e]] = 1.0

                logits_i = self.net(ax, a_n_hw, sample.n_idx, a_r_hw, sample.r_idx)
                archived_logits.append(logits_i)
                archived_targets.append(ay)
                archived_params_list.append(self.net.active_params(sample.n_idx, sample.r_idx))
                active_on_replay_logits.append(self.net(ax, n_hw, n_idx, r_hw, r_idx))

        layer_weights: Dict[int, list] = {}
        ema_weights: Dict[int, list] = {}
        if f_bar:
            for name, p in self.net.named_parameters():
                layer = self._param_name_to_layer.get(name)
                if layer is None or layer not in f_bar:
                    continue
                layer_weights.setdefault(layer, []).append(p)
                ema_weights.setdefault(layer, []).append(self.ema.get(name))

        loss_dict = self.mmlf(
            active_logits, y, active_params,
            archived_logits, archived_targets, archived_params_list,
            layer_weights, ema_weights, f_bar,
            active_on_replay_logits=active_on_replay_logits,
        )

        self.w_optimizer.zero_grad(set_to_none=True)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.w_optimizer.step()
        self.w_scheduler.step()
        self.ema.update(self.net)

        vx, vy = next(val_iter)
        vx, vy = vx.to(self.device), vy.to(self.device)
        v_n_hw, v_n_idx, v_r_hw, v_r_idx = self.net.sample_architecture(tau)
        val_logits = self.net(vx, v_n_hw, v_n_idx, v_r_hw, v_r_idx)
        val_loss = self.criterion(val_logits, vy)
        self.phi_optimizer.zero_grad(set_to_none=True)
        val_loss.backward()
        self.phi_optimizer.step()

        arch_id = self._next_archive_id
        self._next_archive_id += 1
        kappa = self._kappa(node_keys) if self.omega else 0.0
        payload = DARTSArchivedSample(n_idx.detach().cpu(), r_idx.detach().cpu(), node_keys, kappa,
                                       (x[:8].detach().cpu(), y[:8].detach().cpu()))
        self.archive_payloads[arch_id] = payload
        _, evicted = self.archive.offer(arch_id, kappa, payload)
        for evicted_key in evicted:
            self.archive_payloads.pop(evicted_key, None)

        return {
            "loss": loss_dict["total"].item(),
            "acc": accuracy(active_logits, y),
            "tau": tau,
        }

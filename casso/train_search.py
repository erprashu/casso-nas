"""Algorithm 2: CASSO NAS (single-path forward, GDAS-style), specialized to
the NAS-Bench-201 search space (Sec. 3.7, Sec. 4.2.2)."""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .archive import StreamingFacilityLocationArchive
from .config import CASSOConfig
from .losses import EMATeacher, MMLFLoss
from .nb201.supernet import NB201Supernet
from .sensitivity import (
    compute_depth_sharing_weight,
    compute_snip_saliency,
    layerwise_sensitivity_weight,
    sensitivity_distance,
)
from .utils import AverageMeter, accuracy, infinite_loader


@dataclass
class ArchivedSample:
    """What we retain per archived architecture: its genotype string, the
    node keys it uses (for sensitivity bookkeeping), and one cached
    mini-batch to replay it on (Sec. 3.5, replay term)."""

    indices: torch.Tensor
    node_keys: list
    kappa: float
    replay_batch: tuple  # (x, y) tensors, kept small/CPU to bound memory


@dataclass
class SearchStats:
    step_losses: List[float] = field(default_factory=list)
    kendall_tau: Optional[float] = None
    forgetting_curve: Dict[str, list] = field(default_factory=dict)


def cosine_temperature(t: int, total_steps: int, tau0: float, tau_min: float) -> float:
    progress = min(t / max(total_steps, 1), 1.0)
    return tau_min + 0.5 * (tau0 - tau_min) * (1 + math.cos(math.pi * progress))


class CASSOSearcher:
    def __init__(self, supernet: NB201Supernet, cfg: CASSOConfig, device: torch.device,
                 total_steps: int, warmup_steps: int, num_stages: int = 3,
                 cells_per_stage: int = 5):
        self.net = supernet.to(device)
        self.cfg = cfg
        self.device = device
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.num_stages = num_stages
        self.cells_per_stage = cells_per_stage

        self.w_optimizer = torch.optim.SGD(
            [p for n, p in self.net.named_parameters() if n != "arch_logits"],
            lr=0.025, momentum=0.9, weight_decay=cfg.weight_decay,
        )
        self.w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer, T_max=total_steps, eta_min=0.001
        )
        self.phi_optimizer = torch.optim.Adam([self.net.arch_logits], lr=3e-4,
                                               betas=(0.5, 0.999), weight_decay=1e-3)

        # Precompute name -> cell_position ("layer" j in Eq. 11) once, since
        # the set of nn.Parameter objects (and their identities) is fixed
        # for the lifetime of the supernet; only which ops are ACTIVE
        # changes per step. This lets ema_stability_term group the EMA
        # penalty by network depth position without recomputing the map
        # every training step.
        id_to_layer: Dict[int, int] = {}
        for (_, _, cell_position), params in self.net.all_node_param_map().items():
            for p in params:
                id_to_layer[id(p)] = cell_position
        self._param_name_to_layer: Dict[str, int] = {}
        for name, p in self.net.named_parameters():
            layer = id_to_layer.get(id(p))
            if layer is not None:
                self._param_name_to_layer[name] = layer

        self.criterion = nn.CrossEntropyLoss()
        self.mmlf = MMLFLoss(cfg.beta, cfg.gamma, cfg.eta, cfg.weight_decay)
        self.ema = EMATeacher(self.net, cfg.ema_decay)

        self.archive = StreamingFacilityLocationArchive(
            budget=cfg.archive_size, distance_fn=self._archive_distance, tau=cfg.sim_temperature
        )
        self.archive_payloads: Dict[int, ArchivedSample] = {}
        self._next_archive_id = 0

        self.s_bar: Dict = {}
        self.variance: Dict = {}
        self.omega: Dict = {}
        self.sharing_count: Dict = {}

        self.stats = SearchStats()

    # -- sensitivity plumbing -------------------------------------------------

    def _node_param_map(self):
        return self.net.all_node_param_map()

    def refresh_sensitivity(self, minibatches, tau: float = None):
        """Eq. 7 at the CURRENT weights theta_t (Sec. 3.4/3.7): each of the K
        mini-batches samples its own single-path architecture (Gumbel-softmax
        at the current temperature) and back-propagates through it, so the
        resulting s_bar/variance reflect whichever nodes were touched across
        those K draws -- consistent with chi(u) tracking only visited nodes."""
        tau = tau if tau is not None else self.cfg.gumbel_tau_min

        def forward_fn(x):
            hardwts, indices = self.net.sample_architecture(tau)
            return self.net(x, hardwts, indices)

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
        return node_key[2]  # cell_position, see sensitivity.py design note

    # -- main step -------------------------------------------------------------

    def step(self, t: int, train_iter, val_iter, sensitivity_batches=None) -> Dict[str, float]:
        tau = cosine_temperature(t, self.total_steps, self.cfg.gumbel_tau_init, self.cfg.gumbel_tau_min)
        hardwts, indices = self.net.sample_architecture(tau)

        node_keys = self.net.active_node_keys(indices)
        for u in node_keys:
            self.sharing_count[u] = self.sharing_count.get(u, 0) + 1

        if t > self.warmup_steps and t % self.cfg.refresh_interval == 0 and sensitivity_batches:
            self.refresh_sensitivity(sensitivity_batches)

        x, y = next(train_iter)
        x, y = x.to(self.device), y.to(self.device)

        active_logits = self.net(x, hardwts, indices)
        active_params = self.net.active_params(indices)

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
                a_hardwts = torch.zeros_like(hardwts)
                for e_idx in range(a_hardwts.shape[0]):
                    a_hardwts[e_idx, sample.indices[e_idx]] = 1.0
                logits_i = self.net(ax, a_hardwts, sample.indices)
                archived_logits.append(logits_i)
                archived_targets.append(ay)
                archived_params_list.append(self.net.active_params(sample.indices))
                # Active architecture's prediction on this SAME replay batch
                # (same ax), needed for a valid same-input KL comparison
                # (Eq. 10, term 4) -- comparing against active_logits (which
                # was computed on the unrelated, differently-sized main
                # training batch x) would be both shape-mismatched and
                # conceptually meaningless.
                active_on_replay_logits.append(self.net(ax, hardwts, indices))

        layer_weights: Dict[int, list] = {}
        ema_weights: Dict[int, list] = {}
        if f_bar:
            for name, p in self.net.named_parameters():
                layer = self._param_name_to_layer.get(name)
                if layer is None or layer not in f_bar:
                    continue  # only regularize layers actually present in the archive
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

        # GDAS bi-level: update architecture logits on a validation batch.
        vx, vy = next(val_iter)
        vx, vy = vx.to(self.device), vy.to(self.device)
        val_hardwts, val_indices = self.net.sample_architecture(tau)
        val_logits = self.net(vx, val_hardwts, val_indices)
        val_loss = self.criterion(val_logits, vy)
        self.phi_optimizer.zero_grad(set_to_none=True)
        val_loss.backward()
        self.phi_optimizer.step()

        arch_id = self._next_archive_id
        self._next_archive_id += 1
        kappa = self._kappa(node_keys) if self.omega else 0.0
        payload = ArchivedSample(indices.detach().cpu(), node_keys, kappa,
                                  (x[:8].detach().cpu(), y[:8].detach().cpu()))
        self.archive_payloads[arch_id] = payload
        self.archive.offer(arch_id, kappa, payload)

        return {
            "loss": loss_dict["total"].item(),
            "acc": accuracy(active_logits, y),
            "tau": tau,
        }

    def best_architecture(self, val_loader) -> torch.Tensor:
        """Sec. 3.7: select the architecture that minimizes L_val using
        inherited weights (approximated here by evaluating the arg-max
        discrete architecture induced by the learned logits, i.e. tau -> 0)."""
        with torch.no_grad():
            hardwts, indices = self.net.sample_architecture(self.cfg.gumbel_tau_min)
        return indices

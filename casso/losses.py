"""Multi-Model Regularized Loss Function (MMLF), paper Eq. 10-11."""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMLFLoss(nn.Module):
    """Eq. 10:

    L_MMLF = (1-beta) [L_c(active) + lambda ||W(active)||^2]
           + (beta/m) sum_i [L_c(archived_i) + lambda ||W(archived_i)||^2]
           + gamma sum_j F_bar_j(M_t) ||W_j - W_j^EMA||^2
           + eta (1/m) sum_i KL(p_active || p_archived_i)
    """

    def __init__(self, beta: float, gamma: float, eta: float, weight_decay: float):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.weight_decay = weight_decay
        self.ce = nn.CrossEntropyLoss()

    def supervised_term(self, logits: torch.Tensor, targets: torch.Tensor,
                         params: List[torch.Tensor]) -> torch.Tensor:
        l2 = sum((p ** 2).sum() for p in params) if params else torch.tensor(0.0, device=logits.device)
        return self.ce(logits, targets) + self.weight_decay * l2

    def ema_stability_term(self, layer_weights: Dict[int, List[torch.Tensor]],
                            ema_weights: Dict[int, List[torch.Tensor]],
                            f_bar: Dict[int, float]) -> torch.Tensor:
        """gamma * sum_j F_bar_j(M_t) * ||W_j - W_j^EMA||_2^2 (Eq. 10, term 3).

        layer_weights / ema_weights: layer index j -> list of parameter
        tensors belonging to that layer (current and EMA-shadow versions,
        respectively, matched pairwise by list position)."""
        if not layer_weights:
            return torch.tensor(0.0)
        any_tensor = next(iter(layer_weights.values()))[0]
        device = any_tensor.device
        total = torch.tensor(0.0, device=device)
        for j, params in layer_weights.items():
            ema_params = ema_weights.get(j)
            if ema_params is None or len(ema_params) != len(params):
                continue
            layer_sq = torch.tensor(0.0, device=device)
            for w, w_ema in zip(params, ema_params):
                layer_sq = layer_sq + ((w - w_ema.to(device)) ** 2).sum()
            total = total + f_bar.get(j, 0.0) * layer_sq
        return self.gamma * total

    def kl_consistency_term(self, logits_active: torch.Tensor,
                             logits_archived: List[torch.Tensor]) -> torch.Tensor:
        """eta * (1/m) * sum_i KL(p_active || p_archived_i) (Eq. 10, term 4)."""
        if not logits_archived:
            return torch.tensor(0.0, device=logits_active.device)
        log_p_active = F.log_softmax(logits_active, dim=-1)
        p_active = log_p_active.exp()
        total = 0.0
        for logits_i in logits_archived:
            log_p_i = F.log_softmax(logits_i, dim=-1)
            # KL(p_active || p_i) = sum p_active * (log p_active - log p_i)
            kl = (p_active * (log_p_active - log_p_i)).sum(dim=-1).mean()
            total = total + kl
        return self.eta * total / len(logits_archived)

    def forward(
        self,
        active_logits: torch.Tensor,
        active_targets: torch.Tensor,
        active_params: List[torch.Tensor],
        archived_logits: List[torch.Tensor],
        archived_targets: List[torch.Tensor],
        archived_params: List[List[torch.Tensor]],
        layer_weights: Dict[int, torch.Tensor],
        ema_weights: Dict[int, torch.Tensor],
        f_bar: Dict[int, float],
    ) -> Dict[str, torch.Tensor]:
        m = max(len(archived_logits), 1)

        active_term = self.supervised_term(active_logits, active_targets, active_params)

        if archived_logits:
            replay_term = sum(
                self.supervised_term(lg, tg, ps)
                for lg, tg, ps in zip(archived_logits, archived_targets, archived_params)
            ) / m
        else:
            replay_term = torch.tensor(0.0, device=active_logits.device)

        stability_term = self.ema_stability_term(layer_weights, ema_weights, f_bar)
        kl_term = self.kl_consistency_term(active_logits, archived_logits)

        total = (
            (1 - self.beta) * active_term
            + self.beta * replay_term
            + stability_term
            + kl_term
        )
        return {
            "total": total,
            "active": active_term.detach(),
            "replay": replay_term.detach() if torch.is_tensor(replay_term) else torch.tensor(replay_term),
            "stability": stability_term.detach(),
            "kl": kl_term.detach(),
        }


class EMATeacher:
    """Exponential-moving-average teacher weights W^EMA (Sec. 3.5)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name] = p.detach().clone()

    def get(self, name: str):
        return self.shadow.get(name)

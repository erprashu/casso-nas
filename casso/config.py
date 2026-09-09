"""Default hyperparameters for CASSO, matching the values stated in the paper
(Sec. 3.5 "In all experiments, we use beta=0.3, gamma=0.1, eta=0.05, and
lambda=5e-4 unless otherwise specified", Sec. 3.7 for the training protocol,
and Table 5/6 defaults m=10, rho=0.5, R=5)."""

from dataclasses import dataclass


@dataclass
class CASSOConfig:
    # MMLF loss weights (Eq. 10)
    beta: float = 0.3       # replay weight
    gamma: float = 0.1      # EMA stability weight
    eta: float = 0.05       # KL consistency weight
    weight_decay: float = 5e-4  # lambda in Eq. 10

    # Archive (Sec. 3.3, Algorithm 1)
    archive_size: int = 10  # m
    sim_temperature: float = 1.0  # tau in sim(alpha, beta) = exp(-d/tau)
    # stream_window: NOT a paper-specified value -- the paper's S_t is
    # literally all previously-sampled architectures, which is intractable
    # for long runs (see archive.py's docstring: an early full-scale run
    # leaked memory and its per-step cost grew without bound). Tuned here,
    # rather than left at the class default of 2000, specifically to bring
    # measured per-step cost back in line with the paper's own reported
    # ~34560s (0.4 GPU-day) NAS-Bench-201 search-cost budget on this GPU:
    # window=2000 plateaued at ~0.97s/step (~41h total, ~4x over budget);
    # window=200 is calibrated to land close to the stated budget instead.
    stream_window: int = 200

    # Sensitivity (Eqs. 7-9)
    num_minibatches: int = 5     # K: number of mini-batches for SNIP saliency
    refresh_interval: int = 5    # R: steps between sensitivity refreshes (distinct from K)
    depth_rho: float = 0.5       # rho: depth amplification factor in omega(u)
    sensitivity_eps: float = 1e-8  # epsilon in Eq. 9 for numerical stability

    # EMA teacher
    ema_decay: float = 0.999  # mu_EMA

    # Warmup / schedule
    warmup_epochs: int = 15
    gumbel_tau_init: float = 10.0
    gumbel_tau_min: float = 0.1

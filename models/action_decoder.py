
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Decoders (det / gaussian / mdn) — all share the same API:
#   - predict(z) -> a_hat
#   - loss(z,a)  -> scalar training loss
#   - (optional) nll(z,a) for probabilistic models
# =============================================================================

class DeterministicActionDecoder5Head(nn.Module):
    """
    Shared MLP trunk + N primitive heads, each predicting a 2D primitive action.
    predict(z) -> (B, 2*N) action vector.
    loss(z,a)  -> regression loss (Huber by default).
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_primitives: int = 5,
        loss_type: str = "huber",  # "huber" or "mse"
        huber_delta: float = 1.0,
    ):
        super().__init__()
        assert n_layers >= 1
        assert n_primitives >= 1
        self.n_primitives = n_primitives
        self.out_dim = 2 * n_primitives
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)

        if n_layers == 1:
            self.trunk = nn.Linear(in_dim, hidden_dim)
        else:
            layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU(inplace=True)]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True)]
            self.trunk = nn.Sequential(*layers)

        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_primitives)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        parts = [head(h) for head in self.heads]
        return torch.cat(parts, dim=-1)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)

    def loss(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pred = self.forward(z)
        if self.loss_type == "mse":
            return F.mse_loss(pred, a, reduction="mean")
        if self.loss_type == "huber":
            return F.smooth_l1_loss(pred, a, reduction="mean", beta=self.huber_delta)
        raise ValueError(f"Unknown loss_type={self.loss_type}")


class GaussianActionDecoder5Head(nn.Module):
    """
    Diagonal Gaussian decoder with N primitive heads.
    forward(z) -> (mu, log_std) each (B, 2*N)
    predict(z) -> mu
    loss(z,a)  -> NLL
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_primitives: int = 5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        min_std: float = 1e-4,
    ):
        super().__init__()
        assert n_layers >= 1
        assert n_primitives >= 1
        self.n_primitives = n_primitives
        self.out_dim = 2 * n_primitives
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.min_std = float(min_std)

        if n_layers == 1:
            self.trunk = nn.Linear(in_dim, hidden_dim)
        else:
            layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU(inplace=True)]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True)]
            self.trunk = nn.Sequential(*layers)

        self.mu_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_primitives)])
        self.logstd_heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_primitives)])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        mu = torch.cat([head(h) for head in self.mu_heads], dim=-1)
        log_std = torch.cat([head(h) for head in self.logstd_heads], dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(z)
        return mu

    def nll(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu, log_std = self.forward(z)
        std = torch.exp(log_std).clamp_min(self.min_std)
        nll = 0.5 * ((a - mu) / std).pow(2) + torch.log(std) + 0.5 * math.log(2.0 * math.pi)
        return nll.sum(dim=-1).mean()

    def loss(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.nll(z, a)


class MDNActionDecoder5Head(nn.Module):
    """
    K-component diagonal Gaussian mixture over the full 2*N action vector.
    Mixture weights are shared across primitives (one pi per sample).

    forward(z) -> (logits_pi, mu, log_std)
        logits_pi: (B, K)
        mu:        (B, K, 2N)
        log_std:   (B, K, 2N)

    predict(z): mixture mean = sum_k softmax(pi)_k * mu_k  -> (B, 2N)
    loss(z,a):  mixture NLL
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_primitives: int = 5,
        n_components: int = 5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        min_std: float = 1e-4,
    ):
        super().__init__()
        assert n_layers >= 1
        assert n_primitives >= 1
        assert n_components >= 2
        self.n_primitives = n_primitives
        self.out_dim = 2 * n_primitives
        self.K = int(n_components)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.min_std = float(min_std)

        if n_layers == 1:
            self.trunk = nn.Linear(in_dim, hidden_dim)
        else:
            layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU(inplace=True)]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True)]
            self.trunk = nn.Sequential(*layers)

        self.pi_head = nn.Linear(hidden_dim, self.K)
        self.param_heads = nn.ModuleList([nn.Linear(hidden_dim, self.K * 4) for _ in range(n_primitives)])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        h = self.trunk(z)

        logits_pi = self.pi_head(h)

        mus: List[torch.Tensor] = []
        log_stds: List[torch.Tensor] = []
        for head in self.param_heads:
            params = head(h).view(B, self.K, 4)
            mu_xy = params[..., 0:2]
            log_std_xy = params[..., 2:4]
            log_std_xy = torch.clamp(log_std_xy, self.log_std_min, self.log_std_max)
            mus.append(mu_xy)
            log_stds.append(log_std_xy)

        mu = torch.cat(mus, dim=-1)           # (B,K,2N)
        log_std = torch.cat(log_stds, dim=-1) # (B,K,2N)
        return logits_pi, mu, log_std

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        logits_pi, mu, _ = self.forward(z)
        pi = F.softmax(logits_pi, dim=-1)
        return (pi.unsqueeze(-1) * mu).sum(dim=1)

    @staticmethod
    def _log_normal_diag(a: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        a = a.unsqueeze(1)  # (B,1,D)
        std = torch.exp(log_std).clamp_min(1e-4)
        quad = ((a - mu) / std).pow(2).sum(dim=-1)  # (B,K)
        log_det = 2.0 * torch.log(std).sum(dim=-1)  # (B,K)
        D = a.shape[-1]
        return -0.5 * (quad + log_det + D * math.log(2.0 * math.pi))

    def nll(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        logits_pi, mu, log_std = self.forward(z)
        log_pi = F.log_softmax(logits_pi, dim=-1)
        logp = self._log_normal_diag(a, mu, log_std)
        log_mix = torch.logsumexp(log_pi + logp, dim=-1)
        return (-log_mix).mean()

    def loss(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.nll(z, a)


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

_LOG_2PI = math.log(2.0 * math.pi)

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(int(n_layers)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class DeterministicProxyDecoder(nn.Module):
    """
    Deterministic regression decoder for proxy targets.
    API: predict(x)->yhat, loss(x,y)->scalar
    """
    def __init__(self, *, in_dim: int, hidden_dim: int, n_layers: int, out_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = _mlp(in_dim, hidden_dim, out_dim, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yhat = self.forward(x)
        return F.mse_loss(yhat, y)


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



import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroActionDecoder(nn.Module):
    """
    Macro-action decoder for frameskip=5 (outputs 10D = 5*(x,y) primitives).

    Designed for clean ablations:
      - disable_e:    remove dependence on e_t (state), uses a learned constant instead
      - disable_delta:remove dependence on delta (e_{t+1}-e_t), uses a learned constant instead
      - disable_z:    remove FiLM conditioning on z, uses identity FiLM (gamma=1, beta=0)

    Inputs:
      P_t       : [B, N, 384] patch tokens at time t
      P_t1_hat  : [B, N, 384] predicted patch tokens at time t+1 (or can pass P_t1 for oracle)
      z         : [B, 32] continuous codebook vector (can pass None if disable_z=True)

    Notes:
      - We keep trunk input dimension fixed at 256 by substituting learned constants when disabling e/delta.
      - This ensures ablations are comparable (same trunk/head capacity).
    """

    def __init__(
        self,
        token_dim: int = 384,
        z_dim: int = 32,
        hidden_e: int = 256,
        dec_dim: int = 128,
        trunk_dim: int = 256,
        out_dim: int = 10,
        disable_e: bool = False,
        disable_delta: bool = False,
        disable_z: bool = False,
    ):
        super().__init__()
        assert trunk_dim == 2 * dec_dim, "Expected trunk_dim == 2*dec_dim (concat of e and delta projections)."

        self.token_dim = token_dim
        self.z_dim = z_dim
        self.disable_e = disable_e
        self.disable_delta = disable_delta
        self.disable_z = disable_z

        # Attention pooling query (shared for P_t and P_t1_hat)
        self.pool_query = nn.Parameter(torch.randn(token_dim) * 0.02)

        # Projections from pooled tokens/delta into decoder space
        self.mlp_e = nn.Sequential(
            nn.Linear(token_dim, hidden_e),
            nn.GELU(),
            nn.Linear(hidden_e, dec_dim),
        )
        self.mlp_d = nn.Sequential(
            nn.Linear(token_dim, hidden_e),
            nn.GELU(),
            nn.Linear(hidden_e, dec_dim),
        )

        # Learned substitutes used when disabling e/delta (keeps capacity comparable)
        self.e_const = nn.Parameter(torch.zeros(dec_dim))
        self.d_const = nn.Parameter(torch.zeros(dec_dim))

        # FiLM from z: produce (gamma, beta) each of size trunk_dim
        # We generate FiLM params for the *first trunk linear output* (size trunk_dim)
        self.mlp_z = nn.Sequential(
            nn.Linear(z_dim, dec_dim),
            nn.GELU(),
            nn.Linear(dec_dim, 2 * trunk_dim),
        )

        # Trunk + head
        self.trunk_lin = nn.Linear(trunk_dim, trunk_dim)
        self.trunk_norm = nn.LayerNorm(trunk_dim)
        self.trunk_act = nn.GELU()
        self.trunk_lin2 = nn.Linear(trunk_dim, dec_dim)
        self.trunk_act2 = nn.GELU()

        self.head = nn.Linear(dec_dim, out_dim)

    def attend(self, P: torch.Tensor) -> torch.Tensor:
        """
        Attention pooling over patch tokens.
        P: [B, N, D]
        returns pooled: [B, D]
        """
        # logits: [B, N]
        logits = (P @ self.pool_query) / math.sqrt(self.token_dim)
        w = torch.softmax(logits, dim=1)
        return (w.unsqueeze(-1) * P).sum(dim=1)

    def forward(
        self,
        P_t: torch.Tensor,
        P_t1_hat: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = P_t.shape[0]

        # --- pooled token readouts ---
        e_t_raw = self.attend(P_t)          # [B, 384]
        e_t1_raw = self.attend(P_t1_hat)    # [B, 384]
        de_raw = e_t1_raw - e_t_raw         # [B, 384]

        # --- project to decoder space (or substitute constants) ---
        if self.disable_e:
            e = self.e_const.unsqueeze(0).expand(B, -1)  # [B, 128]
        else:
            e = self.mlp_e(F.layer_norm(e_t_raw, [self.token_dim]))  # [B, 128]

        if self.disable_delta:
            d = self.d_const.unsqueeze(0).expand(B, -1)  # [B, 128]
        else:
            d = self.mlp_d(F.layer_norm(de_raw, [self.token_dim]))   # [B, 128]

        h = torch.cat([e, d], dim=-1)  # [B, 256]

        # --- trunk + FiLM conditioning ---
        h = self.trunk_lin(h)          # [B, 256]

        if self.disable_z:
            gamma = torch.ones_like(h)
            beta = torch.zeros_like(h)
        else:
            if z is None:
                raise ValueError("z must be provided unless disable_z=True")
            if z.shape[-1] != self.z_dim:
                raise ValueError(f"Expected z.shape[-1]=={self.z_dim}, got {z.shape[-1]}")
            gamma_beta = self.mlp_z(z)              # [B, 512] if trunk_dim=256
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, 256]
            # Optional stabilization: start near identity
            gamma = 1.0 + 0.1 * torch.tanh(gamma)

        h = gamma * h + beta

        h = self.trunk_norm(h)
        h = self.trunk_act(h)
        h = self.trunk_lin2(h)         # [B, 128]
        h = self.trunk_act2(h)

        return self.head(h)


# -------------------------
# Example ablation configs:
# -------------------------
# Full model:
# dec = MacroActionDecoder(disable_e=False, disable_delta=False, disable_z=False)
#
# Remove z (state+delta only):
# dec = MacroActionDecoder(disable_z=True)
#
# Remove delta (state+z only):
# dec = MacroActionDecoder(disable_delta=True)
#
# Remove e_t (delta+z only)  <-- less meaningful but sometimes useful
# dec = MacroActionDecoder(disable_e=True)
#
# Code-only baseline (approx): disable_e=True, disable_delta=True  (then z only via FiLM)
# dec = MacroActionDecoder(disable_e=True, disable_delta=True, disable_z=False)
#
# State-only baseline: disable_delta=True, disable_z=True
# dec = MacroActionDecoder(disable_delta=True, disable_z=True)

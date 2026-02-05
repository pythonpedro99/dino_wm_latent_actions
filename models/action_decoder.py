
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



class MacroActionDecoderStd(nn.Module):
    """
    Standard macro-action decoder:
      - Attention pooling over patch tokens (learned query).
      - Feature concat + MLP head (no FiLM, no gamma clamps).

    Inputs:
      P_t      : [B, N, token_dim]
      P_t1_hat : [B, N, token_dim]
      z        : [B, z_dim]

    Output:
      a_macro  : [B, out_dim]   (e.g., out_dim=10 for frameskip=5 and 2D primitive)
    """

    def __init__(
        self,
        token_dim: int = 384,
        z_dim: int = 32,
        dec_dim: int = 128,      # projection dim for e and delta
        hid: int = 256,          # MLP hidden width
        out_dim: int = 10,
        use_e: bool = True,
        use_delta: bool = True,
        use_z: bool = True,
        learned_missing: bool = True,  # learned constants for ablated features
        bound_output: bool = False,    # optional tanh bounding
    ):
        super().__init__()
        self.token_dim = token_dim
        self.z_dim = z_dim
        self.dec_dim = dec_dim
        self.out_dim = out_dim

        self.use_e = use_e
        self.use_delta = use_delta
        self.use_z = use_z
        self.learned_missing = learned_missing
        self.bound_output = bound_output

        # --- attention pooling query (shared) ---
        self.pool_query = nn.Parameter(torch.randn(token_dim) * 0.02)

        # --- projections into decoder space ---
        self.proj_e = nn.Sequential(
            nn.Linear(token_dim, hid),
            nn.GELU(),
            nn.Linear(hid, dec_dim),
        )
        self.proj_d = nn.Sequential(
            nn.Linear(token_dim, hid),
            nn.GELU(),
            nn.Linear(hid, dec_dim),
        )

        # --- learned substitutes for missing features (keeps input dim fixed) ---
        if learned_missing:
            self.e_const = nn.Parameter(torch.zeros(dec_dim))
            self.d_const = nn.Parameter(torch.zeros(dec_dim))
            self.z_const = nn.Parameter(torch.zeros(z_dim))
        else:
            self.register_buffer("e_const", torch.zeros(dec_dim))
            self.register_buffer("d_const", torch.zeros(dec_dim))
            self.register_buffer("z_const", torch.zeros(z_dim))

        # --- MLP head on concatenated features ---
        in_dim = dec_dim + dec_dim + z_dim  # [e, d, z]
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, out_dim),
        )

        # Optional bounded output
        if bound_output:
            self.out_scale = nn.Parameter(torch.ones(out_dim))

    def attend(self, P: torch.Tensor) -> torch.Tensor:
        """Attention pooling over patch tokens. P: [B,N,D] -> [B,D]."""
        logits = (P @ self.pool_query) / math.sqrt(self.token_dim)  # [B,N]
        w = torch.softmax(logits, dim=1)
        return (w.unsqueeze(-1) * P).sum(dim=1)  # [B,D]

    def forward(self, P_t: torch.Tensor, P_t1_hat: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        B = P_t.shape[0]

        # pooled token readouts
        e_t_raw = self.attend(P_t)       # [B, token_dim]
        e_t1_raw = self.attend(P_t1_hat) # [B, token_dim]
        de_raw = e_t1_raw - e_t_raw      # [B, token_dim]

        # feature: e
        if self.use_e:
            e = self.proj_e(F.layer_norm(e_t_raw, [self.token_dim]))  # [B, dec_dim]
        else:
            e = self.e_const.unsqueeze(0).expand(B, -1)

        # feature: delta
        if self.use_delta:
            d = self.proj_d(F.layer_norm(de_raw, [self.token_dim]))   # [B, dec_dim]
        else:
            d = self.d_const.unsqueeze(0).expand(B, -1)

        # feature: z
        if self.use_z:
            if z is None:
                raise ValueError("z must be provided when use_z=True")
            if z.shape[-1] != self.z_dim:
                raise ValueError(f"Expected z.shape[-1]=={self.z_dim}, got {z.shape[-1]}")
            zz = z
        else:
            zz = self.z_const.unsqueeze(0).expand(B, -1)

        x = torch.cat([e, d, zz], dim=-1)  # [B, 2*dec_dim + z_dim]
        y = self.mlp(x)                    # [B, out_dim]

        if self.bound_output:
            y = torch.tanh(y) * self.out_scale

        return y


import math
from pathlib import Path
from typing import List, Tuple

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
        use_delta: bool = False,
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

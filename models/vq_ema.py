import torch
import torch.nn as nn
from torch import Tensor


class VectorQuantizerEMA(nn.Module):
    """
    Vector-quantized codebook with exponential moving average updates.

    - Supports `codebook_splits`: the last latent dimension is interpreted as
      `codebook_splits * code_dim`, and each split is quantized independently
      using the *same* codebook.
    - EMA tracks:
        * cluster_size[k]: EMA-smoothed count of assignments to code k
        * embedding_avg[k]: EMA-smoothed sum of encoder vectors for code k
      and normalizes them to update the codebook.
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        codebook_splits: int = 1,
        ema_decay: float = 0.99,
        commitment: float = 0.25,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.codebook_splits = codebook_splits
        self.num_codes = num_codes
        self.ema_decay = ema_decay
        self.commitment = commitment

        # Codebook: [num_codes, code_dim]
        self.embedding = nn.Parameter(torch.randn(num_codes, code_dim))

        # EMA buffers
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embedding_avg", self.embedding.data.clone())

    def forward(self, z: Tensor):
        """
        z: (..., codebook_splits * code_dim)
           e.g. (B, T, 1, codebook_splits * code_dim)

        Returns dict with:
            - "z_q_st": quantized latents with straight-through estimator
            - "loss": commitment loss
            - "indices": code indices per split, same shape as z without the last dim,
                         but with an extra `codebook_splits` axis.
        """
        orig_shape = z.shape
        last_dim = orig_shape[-1]

        # --- Check shape & splits semantics ---
        if last_dim % self.code_dim != 0:
            raise ValueError(
                f"Last dim {last_dim} must be divisible by code_dim={self.code_dim}"
            )
        inferred_splits = last_dim // self.code_dim
        if inferred_splits != self.codebook_splits:
            raise ValueError(
                f"Expected {self.codebook_splits} codebook splits, "
                f"but got {inferred_splits} from last dim={last_dim}"
            )

        # Reshape to (..., splits, code_dim) and flatten to [N, code_dim]
        z = z.view(*orig_shape[:-1], self.codebook_splits, self.code_dim)
        flat = z.reshape(-1, self.code_dim)  # [N, code_dim]

        # --- Compute distances to codebook entries ---
        # dist[i, k] = ||z_i - e_k||^2
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.T
            + self.embedding.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)  # [N]
        z_q = self.embedding[indices].view_as(z)  # same shape as z

        # --- EMA updates (codebook only) ---
        if self.training:
            with torch.no_grad():
                # One-hot encoding of assignments: [N, num_codes]
                enc_one_hot = torch.zeros(
                    indices.size(0),
                    self.num_codes,
                    device=z.device,
                    dtype=z.dtype,  # match z dtype (good for AMP)
                )
                enc_one_hot.scatter_(1, indices.unsqueeze(1), 1)

                # Update cluster_size (EMA of counts)
                self.cluster_size.mul_(self.ema_decay).add_(
                    (1.0 - self.ema_decay) * enc_one_hot.sum(0)
                )

                # Update embedding_avg (EMA of sums)
                # embed_sum: [num_codes, code_dim]
                embed_sum = enc_one_hot.t() @ flat
                self.embedding_avg.mul_(self.ema_decay).add_(
                    (1.0 - self.ema_decay) * embed_sum
                )

                # Total effective count
                n = self.cluster_size.sum()

                # Guard against pathological early steps (e.g., all zeros)
                if n.item() == 0:
                    # If we somehow have no assignments yet, just skip the update.
                    # This avoids dividing by zero or tiny n.
                    pass
                else:
                    # Normalize cluster_size to a probability-like vector,
                    # then rescale back to "count" scale to keep denominators sane.
                    cluster_size = (
                        (self.cluster_size + 1e-6)
                        / (n + self.num_codes * 1e-6)
                        * n
                    )
                    # Clamp to avoid division by extremely small values
                    cluster_size = torch.clamp(cluster_size, min=1e-6)

                    # Broadcast-normalize per code to produce updated codebook
                    embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)

                    # Revive codes that have effectively disappeared
                    dead_mask = cluster_size < 1e-3
                    if dead_mask.any() and flat.numel() > 0:
                        num_dead = int(dead_mask.sum().item())
                        rand_src = torch.randint(
                            0, flat.size(0), (num_dead,), device=z.device
                        )
                        embed_normalized[dead_mask] = flat[rand_src]

                    # Copy back into the parameter (no .data)
                    self.embedding.copy_(embed_normalized)

        # --- Commitment loss ---
        loss = self.commitment * ((z_q.detach() - z) ** 2).mean()

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # --- Restore shapes ---
        z_q_flat = z_q_st.view(*orig_shape)

        # Indices: reshape to original latent shape but with codebook_splits axis
        index_shape = (*orig_shape[:-1], self.codebook_splits)
        indices = indices.view(*index_shape)

        return {"z_q_st": z_q_flat, "loss": loss, "indices": indices}


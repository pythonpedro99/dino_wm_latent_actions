
import torch
import torch.nn as nn
from torch import Tensor

class VectorQuantizerEMA(nn.Module):
    """
    Vector-quantized codebook with exponential moving average updates.

    The EMA update keeps a running sum of how many times each code is selected
    (``cluster_size``) and the sum of all encoder vectors assigned to each code
    (``embedding_avg``). During each forward pass, the raw encoder output ``z``
    is flattened to ``[B*T, code_dim]`` and matched to the closest code
    embeddings. The codebook update works in three steps:

    1. Update ``cluster_size`` with the number of assignments per code.
    2. Update ``embedding_avg`` with the sum of encoder vectors per code,
       storing it as ``[num_codes, code_dim]`` to match the codebook layout.
    3. Normalize ``embedding_avg`` by the per-code counts to obtain the new
       codebook, avoiding division by zero with a small epsilon and copying the
       result into ``self.embedding``.

    This arrangement ensures that the EMA buffers match the shape of the
    codebook and prevents mismatches like the previously observed ``(32 vs 9)``
    dimension error.
    """

    def __init__(self, num_codes, code_dim, ema_decay=0.99, commitment=0.25):
        super().__init__()
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.ema_decay = ema_decay
        self.commitment = commitment

        self.embedding = nn.Parameter(torch.randn(num_codes, code_dim))
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embedding_avg", self.embedding.data.clone())

    def forward(self, z):
        # z: (B,T,1,code_dim)
        flat = z.reshape(-1, self.code_dim)

        # distances
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.T
            + self.embedding.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.embedding[indices].view_as(z)

        # EMA updates (codebook only)
        if self.training:
            with torch.no_grad():
                enc_one_hot = torch.zeros(
                    indices.size(0), self.num_codes, device=z.device
                )
                enc_one_hot.scatter_(1, indices.unsqueeze(1), 1)

                self.cluster_size.mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * enc_one_hot.sum(0)
                )
                # Sum encoder vectors for each code: [num_codes, code_dim]
                embed_sum = enc_one_hot.t() @ flat
                self.embedding_avg.mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * embed_sum
                )

                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + 1e-6) / (n + self.num_codes * 1e-6)
                # Broadcast-normalize per code to produce the updated codebook
                embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
                self.embedding.data.copy_(embed_normalized)

        # Commitment loss
        loss = self.commitment * ((z_q.detach() - z) ** 2).mean()

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        return {"z_q_st": z_q_st, "loss": loss, "indices": indices}
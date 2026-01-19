import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from models.blocks import SpatioTemporalTransformer

# @inproceedings{gao2025adaworld,
#  title={AdaWorld: Learning Adaptable World Models with Latent Actions},
#  author={Gao, Shenyuan and Zhou, Siyuan and Du, Yilun and Zhang, Jun and Gan, Chuang},
#  booktitle={International Conference on Machine Learning (ICML)},
#  year={2025}
# }

class LatentActionModel(nn.Module):
    """
    Latent action encoder operating on DINO patch tokens.

    Input:  visual_tokens (B, T, P, E)   # DINO patches
    Output: {"action_patches": (B, T, 1, E)}  # one action token per frame
    """

    def __init__(
        self,
        in_dim: int,          # should be encoder.emb_dim (DINO emb_dim)
        model_dim: int,       # usually same as in_dim
        latent_dim: int,      # unused now, kept for interface compatibility
        patch_size: int,      # unused now, kept for interface compatibility
        enc_blocks: int,
        dec_blocks: int,      # unused now (no decoder), kept for interface
        num_heads: int,
        dropout: float,
    ) -> None:
        super(LatentActionModel, self).__init__()

        self.model_dim = model_dim
        self.in_dim = in_dim

        # One learnable "action prompt" token per frame
        # Shape will be broadcast to (B, T, 1, in_dim)
        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, in_dim))
        nn.init.uniform_(self.action_prompt, a=-1.0, b=1.0)

        # SpatioTemporal transformer that works directly on patch tokens
        # Input / output dim is in_dim/model_dim (same in practice)
        self.encoder = SpatioTemporalTransformer(
            in_dim=in_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def encode(self, visual_tokens: Tensor) -> Dict[str, Tensor]:
        """
        visual_tokens: (B, T, P, E)  # DINO patch tokens

        Returns:
            {
                "action_patches": (B, T, 1, E)  # latent action token per frame
            }
        """
        B, T, P, E = visual_tokens.shape
        assert E == self.in_dim, f"Expected emb_dim {self.in_dim}, got {E}"

        # Add an action-prompt token per frame
        # action_prompt: (1, 1, 1, E) -> (B, T, 1, E)
        action_pad = self.action_prompt.expand(B, T, 1, E)

        # Concatenate action token + patch tokens along the token dimension
        # padded: (B, T, 1 + P, E)
        padded_tokens = torch.cat([action_pad, visual_tokens], dim=2)

        # Spatiotemporal encoding
        # encoded: (B, T, 1 + P, model_dim)
        encoded = self.encoder(padded_tokens)

        # Take the action token for each frame
        # action_patches: (B, T, 1, model_dim)
        action_patches = encoded[:, :, 0:1, :]  # keep singleton token dim

        return {"action_patches": action_patches}

    def forward(self, visual_tokens: Tensor) -> Dict[str, Tensor]:
        # For VWorldModel, we just call encode and return the dict
        return self.encode(visual_tokens)

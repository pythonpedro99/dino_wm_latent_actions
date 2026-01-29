import os
import sys
import math
import copy
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from omegaconf import OmegaConf
from models.action_decoder import DeterministicActionDecoder5Head, GaussianActionDecoder5Head, MDNActionDecoder5Head

# hard-disable wandb (Trainer calls wandb.init)
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["MPLBACKEND"] = "Agg"


def ensure_hydra_config(run_dir: Path):
    """Keeps Trainer/Hydra happy when running as a plain python script."""
    from hydra.core.hydra_config import HydraConfig
    from hydra.types import RunMode
    from hydra.conf import HydraConf
    from omegaconf import OmegaConf, open_dict

    if HydraConfig.initialized():
        return

    hydra_conf = OmegaConf.structured(HydraConf)
    with open_dict(hydra_conf):
        hydra_conf.mode = RunMode.RUN
        hydra_conf.runtime.output_dir = str(run_dir)
        hydra_conf.run.dir = str(run_dir)
        hydra_conf.job.name = "train_action_decoder"

    cfg = OmegaConf.create({"hydra": hydra_conf})
    HydraConfig.instance().set_config(cfg)


def add_repo_to_syspath():
    """
    Makes `metrics.*` and `dino_wm_latent_actions.*` importable even after os.chdir(run_dir).
    Assumes this script is stored in the repo root OR in /content with the repo cloned to /content/<repo>.
    """
    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir,
        script_dir / "dino_wm_latent_actions",
        Path("/content/dino_wm_latent_actions"),
        Path("/content"),
    ]
    candidates.insert(0, Path.cwd().resolve())

    for root in candidates:
        root = root.resolve()
        if (root / "metrics").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            break
        if (root / "dino_wm_latent_actions" / "metrics").is_dir():
            pkg = root / "dino_wm_latent_actions"
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            if str(pkg) not in sys.path:
                sys.path.insert(0, str(pkg))
            break


# =============================================================================
# Pair sampling (incremental) — builds (z,a) pair sets in terms of *pairs*
# =============================================================================

def _extract_latents(model, obs, act, latent_source: str) -> torch.Tensor:
    """
    model(obs, act) returns (..., encode_output) where encode_output is dict-like.
    latent_source:
      - "continuous": expects encode_output["latent_actions"]
      - "vq": expects encode_output["quantized_latent_actions"] (or common aliases)
    """
    *_rest, encode_output = model(obs, act)

    if latent_source == "continuous":
        keys = ["latent_actions"]
    elif latent_source == "vq":
        keys = ["quantized_latent_actions", "vq_latent_actions", "latent_actions_quantized"]
    else:
        raise ValueError(f"Unknown latent_source={latent_source!r}")

    for k in keys:
        if k in encode_output and encode_output[k] is not None:
            lat = encode_output[k]
            break
    else:
        raise KeyError(
            f"Could not find any of keys={keys} in encode_output. Keys available: {list(encode_output.keys())}"
        )

    if lat.ndim == 2:
        # Likely VQ indices [B,T]; this script expects continuous 32D vectors.
        raise RuntimeError(
            f"Latents have shape {tuple(lat.shape)} (ndim=2). Expected [B,T,D] continuous vectors."
        )

    return lat


def _align_latent_and_act(lat: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Requirement:
      - latent_actions typically has a dummy last action
      - use length-1 z,a pairs
    """
    if lat.ndim != 3:
        raise ValueError(f"Expected latents [B,T,D], got {tuple(lat.shape)}")
    if act.ndim == 2:
        act = act.unsqueeze(-1)
    if act.ndim != 3:
        raise ValueError(f"Expected actions [B,T,A], got {tuple(act.shape)}")

    B, T_lat, D_lat = lat.shape
    _, T_act, A_dim = act.shape

    if T_act == T_lat:
        lat_u = lat[:, :-1]
        act_u = act[:, :-1]
    elif T_act == T_lat - 1:
        lat_u = lat[:, :-1]
        act_u = act
    else:
        T_use = min(T_act, T_lat - 1)
        lat_u = lat[:, :T_use]
        act_u = act[:, :T_use]

    x = lat_u.reshape(-1, D_lat).contiguous()
    y = act_u.reshape(-1, A_dim).contiguous()
    return x, y


class PairStream:
    """Stateful stream of (z,a) *pairs* from Trainer dataloader."""
    def __init__(self, trainer, split: str, latent_source: str, device: torch.device):
        self.trainer = trainer
        self.split = split
        self.latent_source = latent_source
        self.device = device
        self.loader = trainer.dataloaders["train"] if split == "train" else trainer.dataloaders["valid"]
        self.it = iter(self.loader)

        self.model = trainer.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def next_pair_chunk(self) -> Tuple[torch.Tensor, torch.Tensor]:
        while True:
            try:
                batch = next(self.it)
                break
            except StopIteration:
                self.it = iter(self.loader)

        obs, act, *_ = batch

        if isinstance(obs, dict):
            obs = {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}
        else:
            obs = obs.to(self.device, non_blocking=True)
        act = act.to(self.device, non_blocking=True)

        lat = _extract_latents(self.model, obs, act, latent_source=self.latent_source)
        x, y = _align_latent_and_act(lat, act)

        return x.detach().cpu().float(), y.detach().cpu().float()


class PairBuffer:
    """Accumulates pair chunks on CPU without quadratic cat overhead."""
    def __init__(self):
        self.x_chunks: List[torch.Tensor] = []
        self.y_chunks: List[torch.Tensor] = []
        self.n_pairs: int = 0
        self.in_dim: Optional[int] = None
        self.out_dim: Optional[int] = None

    def append(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.ndim == 2 and y.ndim == 2
        assert x.shape[0] == y.shape[0]
        if self.in_dim is None:
            self.in_dim = int(x.shape[1])
            self.out_dim = int(y.shape[1])
        else:
            if int(x.shape[1]) != self.in_dim or int(y.shape[1]) != self.out_dim:
                raise ValueError(
                    f"Dim mismatch: got x={x.shape}, y={y.shape}, expected in_dim={self.in_dim}, out_dim={self.out_dim}"
                )

        self.x_chunks.append(x)
        self.y_chunks.append(y)
        self.n_pairs += int(x.shape[0])

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.x_chunks:
            raise RuntimeError("PairBuffer is empty")
        return torch.cat(self.x_chunks, dim=0), torch.cat(self.y_chunks, dim=0)


def fill_buffer_to(stream: PairStream, buf: PairBuffer, target_pairs: int, *, quiet: bool = False) -> None:
    if target_pairs <= buf.n_pairs:
        return

    if not quiet:
        print(f"  sampling {stream.split} pairs: {buf.n_pairs} -> {target_pairs}")

    while buf.n_pairs < target_pairs:
        x, y = stream.next_pair_chunk()
        remaining = target_pairs - buf.n_pairs
        if x.shape[0] > remaining:
            x = x[:remaining]
            y = y[:remaining]
        buf.append(x, y)

        if not quiet:
            if buf.n_pairs == target_pairs:
                print(f"    now {buf.n_pairs}/{target_pairs} pairs")

    if not quiet:
        print(f"  done sampling: {buf.n_pairs} pairs")


# =============================================================================
# Training / evaluation
# =============================================================================

@torch.no_grad()
def eval_quality(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, batch_size: int, device: torch.device) -> Dict[str, float]:
    """
    One clean quality metric across all models:
      - rmse: sqrt(mean((a_hat - a)^2))
    Also reports:
      - mse
      - mae
      - nll (if model has .nll)
    """
    model.eval()

    preds = []
    N = x.shape[0]
    for i in range(0, N, batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        pred = model.predict(xb).detach().cpu()
        preds.append(pred)

    y_pred = torch.cat(preds, dim=0)
    err = y_pred - y
    mse = float((err.pow(2).mean()).item())
    rmse = float(math.sqrt(mse))
    mae = float(err.abs().mean().item())

    out = {"rmse": rmse, "mse": mse, "mae": mae}

    if hasattr(model, "nll"):
        nll_sum = 0.0
        nll_count = 0
        for i in range(0, N, batch_size):
            xb = x[i:i + batch_size].to(device, non_blocking=True)
            yb = y[i:i + batch_size].to(device, non_blocking=True)
            nll_mean = float(model.nll(xb, yb).detach().cpu().item())
            bs = int(xb.shape[0])
            nll_sum += nll_mean * bs
            nll_count += bs
        out["nll"] = float(nll_sum / max(nll_count, 1))

    return out


@dataclass
class EarlyStopConfig:
    max_epochs: int = 2000
    patience: int = 30
    min_delta: float = 1e-4
    log_every: int = 1


def train_with_early_stopping(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    es: EarlyStopConfig,
    grad_clip: Optional[float] = None,
) -> Dict[str, object]:
    """
    Trains model.loss(z,a) with AdamW, early-stopping on val RMSE.
    Returns dict with:
      - model (best-state loaded)
      - best_epoch
      - epochs_trained
      - best_val (metrics)
    """
    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0

    val0 = eval_quality(model, x_val, y_val, batch_size=batch_size, device=device)
    best_rmse = val0["rmse"]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    if es.log_every > 0:
        extra = f" nll={val0['nll']:.4f}" if "nll" in val0 else ""
        print(f"    [epoch 0] val_rmse={val0['rmse']:.6f} val_mse={val0['mse']:.6f}{extra}")

    last_epoch = 0
    for epoch in range(1, es.max_epochs + 1):
        last_epoch = epoch
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            loss = model.loss(xb, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        val = eval_quality(model, x_val, y_val, batch_size=batch_size, device=device)
        val_rmse = val["rmse"]

        improved = (best_rmse - val_rmse) > es.min_delta
        if improved:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if es.log_every > 0 and (epoch % es.log_every == 0 or improved or bad_epochs >= es.patience):
            extra = f" val_nll={val['nll']:.4f}" if "nll" in val else ""
            flag = " *" if improved else ""
            print(
                f"    [epoch {epoch:4d}]{flag} "
                f"train_loss={train_loss:.6f}  val_rmse={val_rmse:.6f}  best={best_rmse:.6f}  bad={bad_epochs}/{es.patience}{extra}"
            )

        if bad_epochs >= es.patience:
            break

    model.load_state_dict(best_state)
    best_val = eval_quality(model, x_val, y_val, batch_size=batch_size, device=device)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "epochs_trained": last_epoch,
        "best_val": best_val,
    }


def build_decoders(in_dim: int, out_dim: int, *, hidden_dim: int, n_layers: int, n_primitives: int, mdn_components: int):
    assert out_dim == 2 * n_primitives, (
        f"Expected out_dim=2*n_primitives, got out_dim={out_dim}, n_primitives={n_primitives}"
    )
    return {
        "det": DeterministicActionDecoder5Head(
            in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_primitives=n_primitives
        ),
        "gauss": GaussianActionDecoder5Head(
            in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_primitives=n_primitives
        ),
        "mdn": MDNActionDecoder5Head(
            in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_primitives=n_primitives, n_components=mdn_components
        ),
    }


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def save_decoders_into_ckpt(
    ckpt_path: Path,
    out_path: Path,
    *,
    decoders: Dict[str, nn.Module],
    latent_source: str,
    train_pairs: int,
    meta: Dict[str, object],
):
    """
    Writes a new checkpoint file with an additional key:
      action_decoders_<latent_source> = {det, gauss, mdn, meta, train_pairs}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    key = f"action_decoders_{latent_source}"
    ckpt[key] = {
        "train_pairs": int(train_pairs),
        "meta": meta,
        "det": decoders["det"].state_dict(),
        "gauss": decoders["gauss"].state_dict(),
        "mdn": decoders["mdn"].state_dict(),
    }
    torch.save(ckpt, out_path)

# =============================================================================
# Diagnostics / checks (post-hoc, no saving/loading)
# =============================================================================

def _entropy_from_counts(counts: torch.Tensor) -> float:
    # counts: [K] long
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    p = counts.float() / float(total)
    p = p[p > 0]
    return float(-(p * p.log()).sum().item())


@torch.no_grad()
def per_head_rmse_from_preds(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    y_pred/y_true: [N,10] layout (x1,y1,...,x5,y5)
    Returns overall rmse + rmse_head1..5
    """
    assert y_true.ndim == 2 and y_true.shape[1] == 10, f"Expected y_true [N,10], got {tuple(y_true.shape)}"
    assert y_pred.shape == y_true.shape

    err = (y_pred - y_true).view(-1, 5, 2)  # [N,5,2]
    head_mse = err.pow(2).mean(dim=0).mean(dim=-1)  # [5]
    head_rmse = head_mse.sqrt()

    overall_rmse = float((y_pred - y_true).pow(2).mean().sqrt().item())
    out = {"rmse": overall_rmse}
    for h in range(5):
        out[f"rmse_head{h+1}"] = float(head_rmse[h].item())
    return out


@torch.no_grad()
def predict_all(decoder: nn.Module, x: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    decoder.eval()
    preds = []
    N = x.shape[0]
    for i in range(0, N, batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        preds.append(decoder.predict(xb).detach().cpu())
    return torch.cat(preds, dim=0)


def find_codebook_embeddings(root_model: nn.Module, expected_dim: int) -> torch.Tensor:
    """
    Best-effort: find a [K, expected_dim] tensor that looks like a VQ codebook.
    Prefers parameter/module names containing codebook/vq/embed.
    """
    candidates = []

    for name, p in root_model.named_parameters():
        if p is None or p.ndim != 2:
            continue
        if int(p.shape[1]) != expected_dim:
            continue
        lname = name.lower()
        score = 0
        if "codebook" in lname: score += 5
        if "vq" in lname: score += 3
        if "embed" in lname or "embedding" in lname: score += 2
        if score > 0:
            candidates.append((score, name, p.detach().cpu().float()))

    for name, m in root_model.named_modules():
        if isinstance(m, nn.Embedding) and m.weight.ndim == 2 and int(m.weight.shape[1]) == expected_dim:
            candidates.append((1, f"{name}.weight", m.weight.detach().cpu().float()))

    if not candidates:
        raise RuntimeError(
            f"[checks] Could not auto-find codebook embeddings with dim={expected_dim}. "
            f"Add a direct accessor for your model and pass it in explicitly."
        )

    candidates.sort(key=lambda t: (-t[0], -t[2].shape[0]))
    score, name, W = candidates[0]
    print(f"  [checks] codebook: {name}  shape={tuple(W.shape)}  score={score}")
    return W  # [K,D] on CPU


@torch.no_grad()
def nearest_code_indices(x: torch.Tensor, codebook: torch.Tensor, *, device: torch.device, x_batch: int = 4096) -> torch.Tensor:
    """
    x: [N,D] (typically quantized latent vectors)
    codebook: [K,D]
    returns idx: [N] long, nearest code in L2
    batched to avoid huge [N,K] allocations.
    """
    assert x.ndim == 2 and codebook.ndim == 2
    N, D = x.shape
    K, D2 = codebook.shape
    assert D == D2

    E = codebook.to(device, non_blocking=True)
    E_norm = (E ** 2).sum(dim=1).view(1, K)  # [1,K]

    idx_out = torch.empty(N, dtype=torch.long)
    for i in range(0, N, x_batch):
        xb = x[i:i + x_batch].to(device, non_blocking=True)       # [B,D]
        xb_norm = (xb ** 2).sum(dim=1, keepdim=True)              # [B,1]
        dist2 = xb_norm + E_norm - 2.0 * (xb @ E.t())             # [B,K]
        idx = dist2.argmin(dim=1).detach().cpu()
        idx_out[i:i + x_batch] = idx
    return idx_out


@torch.no_grad()
def decode_codes(decoder: nn.Module, codebook: torch.Tensor, *, device: torch.device, codes_batch: int = 4096) -> torch.Tensor:
    """
    returns A = decoder.predict(codebook) -> [K,10]
    """
    decoder.eval()
    K = codebook.shape[0]
    A_chunks = []
    for i in range(0, K, codes_batch):
        eb = codebook[i:i + codes_batch].to(device, non_blocking=True)
        A_chunks.append(decoder.predict(eb).detach().cpu())
    return torch.cat(A_chunks, dim=0)


@torch.no_grad()
def oracle_action_space_rmse(y: torch.Tensor, A: torch.Tensor, *, device: torch.device, y_batch: int = 4096) -> Tuple[float, torch.Tensor]:
    """
    y: [N,10]
    A: [K,10] where A[k] = decoded action for code k
    returns (oracle_rmse, oracle_idx [N])
    """
    assert y.ndim == 2 and y.shape[1] == 10
    assert A.ndim == 2 and A.shape[1] == 10
    N = y.shape[0]
    K = A.shape[0]
    D = 10

    A_dev = A.to(device, non_blocking=True)
    A_norm = (A_dev ** 2).sum(dim=1).view(1, K)  # [1,K]

    total_min = 0.0
    idx_out = torch.empty(N, dtype=torch.long)

    for i in range(0, N, y_batch):
        yb = y[i:i + y_batch].to(device, non_blocking=True)  # [B,10]
        y_norm = (yb ** 2).sum(dim=1, keepdim=True)          # [B,1]
        dist2 = y_norm + A_norm - 2.0 * (yb @ A_dev.t())     # [B,K]
        min_dist2, idx = dist2.min(dim=1)                    # [B]
        total_min += float(min_dist2.sum().item())
        idx_out[i:i + yb.shape[0]] = idx.detach().cpu()

    rmse = math.sqrt(total_min / (N * D))
    return float(rmse), idx_out


@torch.no_grad()
def within_code_action_dispersion(y: torch.Tensor, idx: torch.Tensor, K: int) -> Dict[str, float]:
    """
    Measures how diverse the ground-truth actions are within each assigned code.
    High dispersion => decoder forced to average => irreducible RMSE.
    y: [N,10], idx: [N] code indices
    Returns avg within-code std over all dims + per-head std (2D) averaged.
    """
    assert y.shape[1] == 10
    N = y.shape[0]

    counts = torch.zeros(K, dtype=torch.long)
    sum_ = torch.zeros(K, 10, dtype=torch.float64)
    sumsq = torch.zeros(K, 10, dtype=torch.float64)

    for i in range(N):
        k = int(idx[i].item())
        counts[k] += 1
        yi = y[i].double()
        sum_[k] += yi
        sumsq[k] += yi * yi

    valid = counts > 1
    if valid.sum().item() == 0:
        return {"within_std_all": 0.0, **{f"within_std_head{i}": 0.0 for i in range(1,6)}}

    c = counts[valid].double().unsqueeze(1)   # [M,1]
    mean = sum_[valid] / c                    # [M,10]
    var = (sumsq[valid] / c) - mean * mean    # [M,10]
    var = torch.clamp(var, min=0.0)
    std = var.sqrt()                          # [M,10]

    within_std_all = float(std.mean().item())

    # per head (mean over codes, mean over 2 dims)
    std_heads = std.view(-1, 5, 2).mean(dim=-1).mean(dim=0)  # [5]
    out = {"within_std_all": within_std_all}
    for h in range(5):
        out[f"within_std_head{h+1}"] = float(std_heads[h].item())
    return out


@torch.no_grad()
def run_checks(
    *,
    latent_source: str,
    trainer_model: nn.Module,
    decoder: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    pred_batch: int,
    codes_batch: int = 4096,
    y_batch: int = 4096,
) -> None:
    """
    Runs the minimal set of checks immediately after decoder training.
    For vq: requires access to the VQ codebook via trainer_model.
    """
    print("  [checks] running post-hoc checks...")

    # 1) Decoder predictions on val + per-head RMSE + normalization by target std
    y_pred = predict_all(decoder, x_val, batch_size=pred_batch, device=device)
    rmse = per_head_rmse_from_preds(y_pred, y_val)

    y_std = y_val.std(dim=0, unbiased=False)  # [10]
    nrmse = float(((y_pred - y_val).pow(2).mean().sqrt() / (y_std.mean().clamp(min=1e-8))).item())

    print(
        f"  [checks] pred rmse={rmse['rmse']:.4f}  nrmse~={nrmse:.4f}  "
        + "  ".join([f"h{i}={rmse[f'rmse_head{i}']:.4f}" for i in range(1, 6)])
    )

    # 2) VQ-specific checks
    if latent_source == "vq":
        codebook = find_codebook_embeddings(trainer_model, expected_dim=x_val.shape[1])  # [K,32]
        K = int(codebook.shape[0])

        # 2a) assigned code indices from x_val (should be exact-ish if x_val is quantized embeddings)
        idx_assigned = nearest_code_indices(x_val, codebook, device=device, x_batch=y_batch)
        counts = torch.bincount(idx_assigned, minlength=K)
        used = int((counts > 0).sum().item())
        ent = _entropy_from_counts(counts)

        print(f"  [checks] assigned code usage: used={used}/{K}  entropy={ent:.3f}")

        # 2b) decode each code once
        A = decode_codes(decoder, codebook, device=device, codes_batch=codes_batch)  # [K,10]

        # 2c) action-space oracle (tight lower bound for RMSE metric)
        oracle_rmse, idx_oracle = oracle_action_space_rmse(y_val, A, device=device, y_batch=y_batch)

        # compare assigned vs oracle
        y_assigned = A.index_select(0, idx_assigned)  # [N,10]
        assigned_rmse = float((y_assigned - y_val).pow(2).mean().sqrt().item())
        match = float((idx_assigned == idx_oracle).float().mean().item())

        oracle_head = per_head_rmse_from_preds(A.index_select(0, idx_oracle), y_val)
        assigned_head = per_head_rmse_from_preds(y_assigned, y_val)

        print(
            f"  [checks] ORACLE action-space rmse={oracle_rmse:.4f} | "
            f"assigned-via-codes rmse={assigned_rmse:.4f} | "
            f"match(assigned==oracle)={match*100:.2f}% | gap={assigned_rmse - oracle_rmse:+.4f}"
        )
        print(
            "  [checks] ORACLE per-head:   "
            + "  ".join([f"h{i}={oracle_head[f'rmse_head{i}']:.4f}" for i in range(1, 6)])
        )
        print(
            "  [checks] ASSIGNED per-head: "
            + "  ".join([f"h{i}={assigned_head[f'rmse_head{i}']:.4f}" for i in range(1, 6)])
        )

        # 2d) within-code dispersion (how much averaging is forced)
        disp = within_code_action_dispersion(y_val, idx_assigned, K=K)
        print(
            f"  [checks] within-code dispersion std_all={disp['within_std_all']:.4f}  "
            + "  ".join([f"h{i}={disp[f'within_std_head{i}']:.4f}" for i in range(1, 6)])
        )

    print("  [checks] done.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="checkpoints/model_latest.pth")

    # continuous vs vq (or both)
    ap.add_argument("--latent_source", type=str, default="continuous", choices=["continuous", "vq", "both"])

    # pair-based milestones (not dataset items)
    ap.add_argument("--train_pairs", type=str, default="5000,10000,50000")
    ap.add_argument("--val_pairs", type=int, default=20000)

    # model hyperparams
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_primitives", type=int, default=5)
    ap.add_argument("--mdn_components", type=int, default=5)
    ap.add_argument("--checks", action="store_true")

    # training hyperparams
    ap.add_argument("--pair_batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # early stopping
    ap.add_argument("--max_epochs", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=1)

    # saving
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_each_size", action="store_true")

    # logging
    ap.add_argument("--quiet_sampling", action="store_true")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    add_repo_to_syspath()
    os.chdir(run_dir)

    cfg_path = run_dir / "hydra.yaml"
    assert cfg_path.exists(), f"Missing hydra.yaml: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)

    ensure_hydra_config(run_dir)

    from dino_wm_latent_actions.train import Trainer
    trainer = Trainer(cfg)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (run_dir / ckpt_path).resolve()
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    trainer.load_ckpt(ckpt_path)

    device = trainer.device
    trainer.model.eval()
    for p in trainer.model.parameters():
        p.requires_grad_(False)

    sources = ["continuous", "vq"] if args.latent_source == "both" else [args.latent_source]
    train_pair_targets = parse_int_list(args.train_pairs)
    assert len(train_pair_targets) >= 1
    assert all(t > 0 for t in train_pair_targets)
    assert sorted(train_pair_targets) == train_pair_targets, "train_pairs must be increasing"

    es = EarlyStopConfig(
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        log_every=int(args.log_every),
    )
    grad_clip = float(args.grad_clip) if args.grad_clip and args.grad_clip > 0 else None

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")
    print(f"train_pairs targets: {train_pair_targets}")
    print(f"val_pairs: {args.val_pairs}")
    print("metric for quality (all models): val_rmse (lower is better)")
    print("")

    for latent_source in sources:
        print("=" * 90)
        print(f"[latent_source={latent_source}]")

        # 1) Build fixed validation pair set once
        val_stream = PairStream(trainer, split="valid", latent_source=latent_source, device=device)
        val_buf = PairBuffer()
        fill_buffer_to(val_stream, val_buf, target_pairs=args.val_pairs, quiet=args.quiet_sampling)
        x_val, y_val = val_buf.tensors()

        in_dim = int(x_val.shape[1])
        out_dim = int(y_val.shape[1])

        print(f"  val: pairs={x_val.shape[0]}  in_dim={in_dim}  out_dim={out_dim}")
        if out_dim != 2 * args.n_primitives:
            print(f"  WARNING: out_dim={out_dim} but n_primitives={args.n_primitives} -> expected {2*args.n_primitives}")

        # 2) Incrementally sample train pairs and train models at milestones
        train_stream = PairStream(trainer, split="train", latent_source=latent_source, device=device)
        train_buf = PairBuffer()

        last_models: Optional[Dict[str, nn.Module]] = None
        last_target: Optional[int] = None

        for target_pairs in train_pair_targets:
            fill_buffer_to(train_stream, train_buf, target_pairs=target_pairs, quiet=args.quiet_sampling)
            x_train, y_train = train_buf.tensors()

            if x_train.shape[1] != in_dim or y_train.shape[1] != out_dim:
                raise RuntimeError(
                    f"Dim mismatch: train x/y={x_train.shape}/{y_train.shape}, val x/y={x_val.shape}/{y_val.shape}"
                )

            print("-" * 90)
            print(f"  [train_pairs={target_pairs}] train_pairs={x_train.shape[0]}  batch_size={args.pair_batch_size}")

            # Train all three models (fresh for this milestone)
            models = build_decoders(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_primitives=args.n_primitives,
                mdn_components=args.mdn_components,
            )
            for m in models.values():
                m.to(device)

            results = {}
            for name, model in models.items():
                print(f"  -> training {name} ...")
                res = train_with_early_stopping(
                    model,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    batch_size=args.pair_batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                    es=es,
                    grad_clip=grad_clip,
                )
                results[name] = res

                best = res["best_val"]
                extra = f" nll={best['nll']:.4f}" if "nll" in best else ""
                print(
                    f"     [{name}] best_epoch={res['best_epoch']}  "
                    f"val_rmse={best['rmse']:.6f}  val_mse={best['mse']:.6f}{extra}"
                )

            # One clean metric summary
            det_rmse = results["det"]["best_val"]["rmse"]
            ga_rmse = results["gauss"]["best_val"]["rmse"]
            mdn_rmse = results["mdn"]["best_val"]["rmse"]
            print(f"  SUMMARY (val_rmse lower=better): det={det_rmse:.6f}  gauss={ga_rmse:.6f}  mdn={mdn_rmse:.6f}")
            
            if args.checks:
                # Usually deterministic is the cleanest for diagnostics because predict() is straightforward.
                run_checks(
                    latent_source=latent_source,
                    trainer_model=trainer.model,
                    decoder=results["det"]["model"],   # best-state already loaded
                    x_val=x_val,
                    y_val=y_val,
                    device=device,
                    pred_batch=args.pair_batch_size,
                    codes_batch=4096,
                    y_batch=4096,
                )
                
            last_models = {k: v["model"] for k, v in results.items()}
            last_target = target_pairs

            # Optional save at each milestone
            if args.save and args.save_each_size:
                out_path = ckpt_path if args.overwrite else ckpt_path.with_name(
                    ckpt_path.stem + f"_decoders_{latent_source}_{target_pairs}.pth"
                )
                meta = {
                    "latent_source": latent_source,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "hidden_dim": args.hidden_dim,
                    "n_layers": args.n_layers,
                    "n_primitives": args.n_primitives,
                    "mdn_components": args.mdn_components,
                    "train_pairs": target_pairs,
                    "val_pairs": int(x_val.shape[0]),
                    "metric": "val_rmse",
                }
                save_decoders_into_ckpt(
                    ckpt_path=ckpt_path,
                    out_path=out_path,
                    decoders=last_models,
                    latent_source=latent_source,
                    train_pairs=target_pairs,
                    meta=meta,
                )
                print(f"  saved decoders to: {out_path}")

        # Save final (largest) decoders by default
        if args.save and last_models is not None and last_target is not None and not args.save_each_size:
            out_path = ckpt_path if args.overwrite else ckpt_path.with_name(ckpt_path.stem + f"_decoders_{latent_source}.pth")
            meta = {
                "latent_source": latent_source,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "n_primitives": args.n_primitives,
                "mdn_components": args.mdn_components,
                "train_pairs": last_target,
                "val_pairs": int(x_val.shape[0]),
                "metric": "val_rmse",
            }
            save_decoders_into_ckpt(
                ckpt_path=ckpt_path,
                out_path=out_path,
                decoders=last_models,
                latent_source=latent_source,
                train_pairs=last_target,
                meta=meta,
            )
            print(f"  saved final decoders to: {out_path}")
        elif not args.save:
            print("  (save disabled)")

    print("\nDone.")


if __name__ == "__main__":
    main()

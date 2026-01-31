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
from models.action_decoder import DeterministicActionDecoder5Head, DeterministicProxyDecoder

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

def _standardize_targets(
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    *,
    target: str,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    if target == "action10":
        return y_train, y_val, None

    if target == "proxy_dir3":
        mean_dir = y_train[:, :2].mean(dim=0, keepdim=True)
        std_dir = y_train[:, :2].std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
        mean_mag = y_train[:, 2:].mean(dim=0, keepdim=True)
        std_mag = y_train[:, 2:].std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
        mean = torch.cat([mean_dir, mean_mag], dim=-1)
        std = torch.cat([std_dir, std_mag], dim=-1)
    else:
        mean = y_train.mean(dim=0, keepdim=True)
        std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)

    y_train_std = (y_train - mean) / std
    y_val_std = (y_val - mean) / std
    stats = {"mean": mean.squeeze(0), "std": std.squeeze(0)}
    return y_train_std, y_val_std, stats


def _unstandardize(y: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"].to(y.device, non_blocking=True)
    std = stats["std"].to(y.device, non_blocking=True)
    return y * std + mean


def print_baseline_rmse(y_train: torch.Tensor, y_val: torch.Tensor) -> float:
    mu = y_train.mean(dim=0, keepdim=True)
    rmse_baseline = ((y_val - mu).pow(2).mean().sqrt()).item()
    print("baseline_rmse(mean predictor):", rmse_baseline)
    return float(rmse_baseline)


@torch.no_grad()
def kmeans_elbow_sweep(
    P: torch.Tensor,
    *,
    max_k: int,
    iters: int = 20,
    seed: int = 0,
) -> List[Tuple[int, float, float]]:
    """
    Runs a simple k-means sweep on standardized proxy vectors.
    Returns list of (k, sse, ev) where ev = 1 - sse/total_var.
    """
    if max_k < 1:
        return []

    P = P.float()
    P0 = P - P.mean(dim=0, keepdim=True)
    P0 = P0 / (P0.std(dim=0, keepdim=True) + 1e-8)

    N = P0.shape[0]
    total_var = float((P0.pow(2).sum()).item())
    g = torch.Generator(device=P0.device)
    g.manual_seed(int(seed))

    out: List[Tuple[int, float, float]] = []
    for k in range(1, max_k + 1):
        if k >= N:
            break
        perm = torch.randperm(N, generator=g, device=P0.device)
        centers = P0.index_select(0, perm[:k]).clone()

        for _ in range(int(iters)):
            d2 = torch.cdist(P0, centers).pow(2)
            labels = d2.argmin(dim=1)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    centers[i] = P0[mask].mean(dim=0)
                else:
                    centers[i] = P0[perm[i % N]]

        d2_final = torch.cdist(P0, centers).pow(2)
        min_d2, _ = d2_final.min(dim=1)
        sse = float(min_d2.sum().item())
        ev = 1.0 - (sse / max(total_var, 1e-12))
        out.append((k, sse, ev))
    return out


@torch.no_grad()
def eval_quality(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    target_stats: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
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
    if target_stats is not None:
        y_pred = _unstandardize(y_pred, target_stats)
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
    loss_fn: Optional[callable] = None,
    target_stats: Optional[Dict[str, torch.Tensor]] = None,
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

    val0 = eval_quality(
        model,
        x_val,
        y_val,
        batch_size=batch_size,
        device=device,
        target_stats=target_stats,
    )
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
            if loss_fn is None:
                loss = model.loss(xb, yb)
            else:
                loss = loss_fn(model, xb, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        val = eval_quality(
            model,
            x_val,
            y_val,
            batch_size=batch_size,
            device=device,
            target_stats=target_stats,
        )
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
    best_val = eval_quality(
        model,
        x_val,
        y_val,
        batch_size=batch_size,
        device=device,
        target_stats=target_stats,
    )

    return {
        "model": model,
        "best_epoch": best_epoch,
        "epochs_trained": last_epoch,
        "best_val": best_val,
    }


def build_decoders(
    in_dim: int,
    out_dim: int,
    *,
    hidden_dim: int,
    n_layers: int,
    n_primitives: int,
    target: str,
):
    if target == "action10":
        assert out_dim == 2 * n_primitives, (
            f"Expected out_dim=2*n_primitives, got out_dim={out_dim}, n_primitives={n_primitives}"
        )
        return {
            "det": DeterministicActionDecoder5Head(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, n_primitives=n_primitives),
        }

    # proxy targets: generic vector decoders
    return {
        "det": DeterministicProxyDecoder(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, out_dim=out_dim),
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
      action_decoders_<latent_source> = {det, meta, train_pairs}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    key = f"action_decoders_{latent_source}"
    ckpt[key] = {
        "train_pairs": int(train_pairs),
        "meta": meta,
        "det": decoders["det"].state_dict(),
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
def proxy_rmse_from_preds(y_pred: torch.Tensor, y_true: torch.Tensor, *, target: str) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(err.pow(2).mean().sqrt().item())
    out = {"rmse": rmse}
    if target == "proxy_dir3":
        rmse_dir = float(err[:, :2].pow(2).mean().sqrt().item())
        rmse_mag = float(err[:, 2:].pow(2).mean().sqrt().item())
        out["rmse_dir"] = rmse_dir
        out["rmse_mag"] = rmse_mag
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
    target: str,
    target_stats: Optional[Dict[str, torch.Tensor]] = None,
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
    if target_stats is not None:
        y_pred = _unstandardize(y_pred, target_stats)

    y_std = y_val.std(dim=0, unbiased=False)
    nrmse = float(((y_pred - y_val).pow(2).mean().sqrt() / (y_std.mean().clamp(min=1e-8))).item())

    if target == "action10":
        rmse = per_head_rmse_from_preds(y_pred, y_val)
        print(
            f"  [checks] pred rmse={rmse['rmse']:.4f}  nrmse~={nrmse:.4f}  "
            + "  ".join([f"h{i}={rmse[f'rmse_head{i}']:.4f}" for i in range(1, 6)])
        )
    else:
        rmse = proxy_rmse_from_preds(y_pred, y_val, target=target)
        extra = ""
        if target == "proxy_dir3":
            extra = f"  dir_rmse={rmse['rmse_dir']:.4f}  mag_rmse={rmse['rmse_mag']:.4f}"
        print(
            f"  [checks] pred rmse={rmse['rmse']:.4f}  nrmse~={nrmse:.4f}{extra}"
        )

    # 2) VQ-specific checks
    if latent_source == "vq" and target == "action10":
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
    elif latent_source == "vq" and target != "action10":
        print("  [checks] vq codebook checks are action-only; skipping for proxy targets.")

    print("  [checks] done.")

# -----------------------------------------------------------------------------
# PCA + plotting (UPDATED)
# - discrete/distinct-ish colors with fixed vmin/vmax and colorbar ticks
# - prints std checks (max/min ratio) to decide whether to standardize
# - prints used codes + counts for the plotted subset
# -----------------------------------------------------------------------------
def _pca_2d_from_actions(
    y: torch.Tensor,
    *,
    standardize: bool = True,
    print_std_checks: bool = True,
) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """
    y: [N,10] actions (x1,y1,...,x5,y5)
    Returns:
      coords: [N,2] PCA coordinates
      evr: (evr1, evr2) explained variance ratios for PC1/PC2
    """
    assert y.ndim == 2, f"Expected y [N,10], got {tuple(y.shape)}"
    y = y.float()

    # --- diagnostics on raw y ---
    if print_std_checks:
        std = y.std(dim=0, unbiased=False)
        std_min = float(std.min().item())
        std_max = float(std.max().item())
        ratio = std_max / max(std_min, 1e-12)
        print(f"  [pca] raw action std per-dim: {[float(v) for v in std.tolist()]}")
        print(f"  [pca] std_min={std_min:.6g} std_max={std_max:.6g} max/min={ratio:.3f}")
        if ratio < 2.0:
            print("  [pca] std spread small -> standardize likely unnecessary.")
        elif ratio > 5.0:
            print("  [pca] std spread large -> standardize recommended.")

    # optional standardization
    if standardize:
        mu = y.mean(dim=0, keepdim=True)
        sig = y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        y = (y - mu) / sig

    # center
    y = y - y.mean(dim=0, keepdim=True)

    # SVD-based PCA (N x D)
    U, S, Vh = torch.linalg.svd(y, full_matrices=False)  # Vh: [D,D]
    V = Vh.transpose(0, 1)  # [D,D]

    comps = V[:, :2]          # [D,2]
    coords = y @ comps        # [N,2]

    # explained variance ratio
    N = y.shape[0]
    eig = (S ** 2) / max(N - 1, 1)
    total = eig.sum().clamp_min(1e-12)
    evr1 = float((eig[0] / total).item())
    evr2 = float((eig[1] / total).item())

    return coords, (evr1, evr2)


def plot_action_pca_colored_by_code(
    y_val: torch.Tensor,
    code_idx: torch.Tensor,
    *,
    out_path: Path,
    dpi: int = 600,
    max_points: int = 0,
    seed: int = 0,
    standardize: bool = False,
    K: Optional[int] = None,
    cmap: str = "tab20",
) -> None:
    """
    Saves ONE figure:
      PCA(actions 10D->2D), scatter colored by VQ code index.

    - discrete-ish colors using fixed vmin/vmax (code bins)
    - prints code usage counts for the plotted subset
    - prints action std checks (raw) to guide standardization choice
    """
    import matplotlib.pyplot as plt

    assert y_val.ndim == 2 and y_val.shape[1] == 10, f"Expected y_val [N,10], got {tuple(y_val.shape)}"
    assert code_idx.ndim == 1 and code_idx.shape[0] == y_val.shape[0], (
        f"Expected code_idx [N], got {tuple(code_idx.shape)} vs N={y_val.shape[0]}"
    )

    # Determine K (number of codes) for stable color binning
    if K is None:
        K = int(code_idx.max().item()) + 1
    K = int(K)

    # Optional subsample (keeps pairing)
    N = y_val.shape[0]
    if max_points and max_points > 0 and max_points < N:
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(N, generator=g)[: int(max_points)]
        y = y_val.index_select(0, perm)
        idx = code_idx.index_select(0, perm)
        print(f"  [plot] subsample: {N} -> {y.shape[0]} (seed={seed})")
    else:
        y = y_val
        idx = code_idx

    # Print code usage on the plotted subset
    counts = torch.bincount(idx, minlength=K)
    used = int((counts > 0).sum().item())
    nz = torch.nonzero(counts > 0).view(-1)
    nz_pairs = [(int(k.item()), int(counts[k].item())) for k in nz]
    print(f"  [plot] codes used in plotted set: {used}/{K}")
    print(f"  [plot] nonzero code counts (code:count): {nz_pairs}")

    # PCA on actions
    coords, (evr1, evr2) = _pca_2d_from_actions(y, standardize=bool(standardize), print_std_checks=True)

    x = coords[:, 0].cpu().numpy()
    yy = coords[:, 1].cpu().numpy()
    c = idx.cpu().numpy()

    # Discrete binning for integer codes
    vmin = -0.5
    vmax = K - 0.5

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        x, yy,
        c=c,
        s=2,
        alpha=0.9,
        linewidths=0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar with integer ticks
    cbar = plt.colorbar(sc, ticks=list(range(K)))
    cbar.set_label("VQ code index")

    plt.xlabel(f"PC1 (EVR {evr1*100:.1f}%)")
    plt.ylabel(f"PC2 (EVR {evr2*100:.1f}%)")
    plt.title(f"Action PCA (10D→2D) colored by VQ code (val pairs) | standardize={standardize}")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.as_posix(), dpi=int(dpi), bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved: {out_path} (dpi={dpi}, points={len(c)})")


def make_target(y_action: torch.Tensor, *, target: str, n_primitives: int, eps: float = 1e-8) -> torch.Tensor:
    """
    y_action: [N, 2*n_primitives] (default n_primitives=5 => [N,10])
    Returns y_target with shape:
      - action10: [N,10]
      - proxy_r2: [N,2]      (resultant vector)
      - proxy_dir3: [N,3]    (unit direction + magnitude)
      - proxy_endbend3: [N,3] (endpoint displacement + bend scalar)
    """
    y = y_action.float()
    N, A = y.shape
    assert A == 2 * n_primitives, f"Expected y_action dim {2*n_primitives}, got {A}"

    if target == "action10":
        return y

    a = y.view(N, n_primitives, 2)        # [N,5,2]
    r = a.sum(dim=1)                      # [N,2]

    if target == "proxy_r2":
        return r

    if target == "proxy_dir3":
        m = torch.linalg.norm(r, dim=-1, keepdim=True)     # [N,1]
        d = r / (m + eps)                                  # [N,2]
        return torch.cat([d, m], dim=-1)                    # [N,3]

    if target == "proxy_endbend3":
        # cumulative positions p_t in pusher-relative coords
        p = torch.cumsum(a, dim=1)      # [N,5,2]
        Delta = p[:, -1, :]             # [N,2]
        norm = torch.linalg.norm(Delta, dim=-1, keepdim=True)  # [N,1]
        u = Delta / (norm + eps)        # [N,2]
        n = torch.stack([-u[:, 1], u[:, 0]], dim=-1)  # [N,2] perpendicular

        # deviation from straight line at t=1..(H-1)
        H = n_primitives
        t = torch.arange(1, H, device=y.device, dtype=y.dtype).view(1, H-1, 1)  # [1,4,1]
        lin = (t / float(H)) * Delta.view(N, 1, 2)                               # [N,4,2]
        dev = p[:, :H-1, :] - lin                                                # [N,4,2]
        b = (dev * n.view(N, 1, 2)).sum(dim=-1).mean(dim=1, keepdim=True)        # [N,1]
        return torch.cat([Delta, b], dim=-1)                                     # [N,3]

    raise ValueError(f"Unknown target={target!r}")


# -----------------------------------------------------------------------------
# main() (UPDATED call site)
# - adds CLI flags to control standardization + colormap
# - passes K=codebook.shape[0] so colors are stable even if some codes unused
# -----------------------------------------------------------------------------

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
    ap.add_argument("--checks", action="store_true")
    ap.add_argument(
        "--target",
        type=str,
        default="proxy_dir3",
        choices=["action10", "proxy_r2", "proxy_dir3", "proxy_endbend3"],
        help="Training target: action10 (5x2) or proxy variants.",
    )


    # training hyperparams
    ap.add_argument("--pair_batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # early stopping
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=1)

    # saving
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_each_size", action="store_true")

    # logging
    ap.add_argument("--quiet_sampling", action="store_true")

    # plotting: action PCA colored by VQ code
    ap.add_argument("--plot_action_pca", action="store_true")
    ap.add_argument("--plot_path", type=str, default="action_pca_by_code.png")
    ap.add_argument("--plot_dpi", type=int, default=600)
    ap.add_argument("--plot_max_points", type=int, default=0, help="0 = use all points")
    ap.add_argument("--plot_seed", type=int, default=0)
    ap.add_argument("--plot_standardize", action="store_true", help="Standardize per action dim before PCA")
    ap.add_argument("--plot_cmap", type=str, default="tab20", help="Matplotlib colormap name (e.g., tab20)")

    # proxy diagnostics
    ap.add_argument("--proxy_kmeans_max_k", type=int, default=16, help="Run k-means sweep on proxy vectors up to K.")
    ap.add_argument("--proxy_kmeans_iters", type=int, default=20, help="Iterations per k-means run.")

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
        
        x_val, y_val_action = val_buf.tensors()
        y_val_raw = make_target(y_val_action, target=args.target, n_primitives=args.n_primitives)
        if args.target != "action10" and args.proxy_kmeans_max_k > 0:
            sweep = kmeans_elbow_sweep(
                y_val_raw,
                max_k=int(args.proxy_kmeans_max_k),
                iters=int(args.proxy_kmeans_iters),
                seed=int(args.plot_seed),
            )
            if sweep:
                print("  [proxy] k-means sweep (standardized proxy vectors):")
                for k, sse, ev in sweep:
                    print(f"    k={k:2d}  sse={sse:.4f}  ev={ev:.4f}")


        # --- Q1 figure: PCA(actions)->2D colored by VQ code (val set) ---
        if args.plot_action_pca:
            if latent_source != "vq":
                print("  [plot] plot_action_pca requested, but latent_source != 'vq' -> skipping (need codes).")
            else:
                codebook = find_codebook_embeddings(trainer.model, expected_dim=x_val.shape[1])  # [K,D]
                idx_assigned = nearest_code_indices(x_val, codebook, device=device, x_batch=4096)  # [N]

                plot_path = Path(args.plot_path)
                if not plot_path.is_absolute():
                    plot_path = (Path.cwd() / plot_path).resolve()

                plot_action_pca_colored_by_code(
                    y_val=y_val_action,
                    code_idx=idx_assigned,
                    out_path=plot_path,
                    dpi=args.plot_dpi,
                    max_points=args.plot_max_points,
                    seed=args.plot_seed,
                    standardize=args.plot_standardize,
                    K=int(codebook.shape[0]),
                    cmap=args.plot_cmap,
                )

        in_dim = int(x_val.shape[1])
        out_dim = int(y_val_raw.shape[1])

        print(f"  val: pairs={x_val.shape[0]}  in_dim={in_dim}  out_dim={out_dim}")
        if args.target == "action10" and out_dim != 2 * args.n_primitives:
            print(f"  WARNING: out_dim={out_dim} but n_primitives={args.n_primitives} -> expected {2*args.n_primitives}")

        # 2) Incrementally sample train pairs and train models at milestones
        train_stream = PairStream(trainer, split="train", latent_source=latent_source, device=device)
        train_buf = PairBuffer()

        last_models: Optional[Dict[str, nn.Module]] = None
        last_target: Optional[int] = None

        for target_pairs in train_pair_targets:
            fill_buffer_to(train_stream, train_buf, target_pairs=target_pairs, quiet=args.quiet_sampling)
            x_train, y_train_action = train_buf.tensors()
            y_train_raw = make_target(y_train_action, target=args.target, n_primitives=args.n_primitives)
            y_train, _y_val_scaled, target_stats = _standardize_targets(
                y_train_raw,
                y_val_raw,
                target=args.target,
            )


            if x_train.shape[1] != in_dim or y_train.shape[1] != out_dim:
                raise RuntimeError(
                    f"Dim mismatch: train x/y={x_train.shape}/{y_train.shape}, val x/y={x_val.shape}/{y_val_raw.shape}"
                )

            print("-" * 90)
            print(f"  [train_pairs={target_pairs}] train_pairs={x_train.shape[0]}  batch_size={args.pair_batch_size}")

            print_baseline_rmse(y_train_raw, y_val_raw)

            # Train deterministic decoder (fresh for this milestone)
            models = build_decoders(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_primitives=args.n_primitives,
                target=args.target,
            )

            for m in models.values():
                m.to(device)

            results = {}
            def proxy_dir_loss(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
                pred = model.forward(xb)
                loss_dir = F.mse_loss(pred[:, :2], yb[:, :2], reduction="mean")
                loss_mag = F.mse_loss(pred[:, 2:], yb[:, 2:], reduction="mean")
                return loss_dir + loss_mag

            loss_fn = proxy_dir_loss if args.target == "proxy_dir3" else None

            for name, model in models.items():
                print(f"  -> training {name} ...")
                res = train_with_early_stopping(
                    model,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val_raw,
                    batch_size=args.pair_batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                    es=es,
                    grad_clip=grad_clip,
                    loss_fn=loss_fn,
                    target_stats=target_stats,
                )
                results[name] = res

                best = res["best_val"]
                extra = f" nll={best['nll']:.4f}" if "nll" in best else ""
                print(
                    f"     [{name}] best_epoch={res['best_epoch']}  "
                    f"val_rmse={best['rmse']:.6f}  val_mse={best['mse']:.6f}{extra}"
                )

            det_rmse = results["det"]["best_val"]["rmse"]
            print(f"  SUMMARY (val_rmse lower=better): det={det_rmse:.6f}")

            if args.checks:
                run_checks(
                    latent_source=latent_source,
                    trainer_model=trainer.model,
                    decoder=results["det"]["model"],
                    x_val=x_val,
                    y_val=y_val_raw,
                    device=device,
                    pred_batch=args.pair_batch_size,
                    target=args.target,
                    target_stats=target_stats,
                    codes_batch=4096,
                    y_batch=4096,
                )

            last_models = {k: v["model"] for k, v in results.items()}
            last_target = target_pairs

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
                    "train_pairs": target_pairs,
                    "val_pairs": int(x_val.shape[0]),
                    "metric": "val_rmse",
                    "target": args.target,
                    "target_mean": None if target_stats is None else target_stats["mean"].tolist(),
                    "target_std": None if target_stats is None else target_stats["std"].tolist(),
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

        if args.save and last_models is not None and last_target is not None and not args.save_each_size:
            out_path = ckpt_path if args.overwrite else ckpt_path.with_name(
                ckpt_path.stem + f"_decoders_{latent_source}.pth"
            )
            meta = {
                "latent_source": latent_source,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "n_primitives": args.n_primitives,
                "train_pairs": last_target,
                "val_pairs": int(x_val.shape[0]),
                "metric": "val_rmse",
                "target": args.target,
                "target_mean": None if target_stats is None else target_stats["mean"].tolist(),
                "target_std": None if target_stats is None else target_stats["std"].tolist(),
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

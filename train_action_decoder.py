import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from omegaconf import OmegaConf
from models.action_decoder import MacroActionDecoder

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
        hydra_conf.job.name = "train_action_encoder"

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


def _parse_pair_sizes(value: str) -> List[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        sizes = [int(part) for part in parts]
        if not sizes:
            raise ValueError("train_pairs must have at least one value.")
        return sizes
    raise TypeError(f"Unsupported train_pairs value: {value!r}")


def _extract_latents(
    encode_output: Dict[str, torch.Tensor],
    latent_source: str,
) -> torch.Tensor:
    """
    model.encode(obs, act) returns dict-like outputs.
    latent_source:
      - "continuous": expects encode_output["latent_actions"]
      - "vq": expects encode_output["quantized_latent_actions"] (or common aliases)
    """
    if latent_source == "continuous":
        keys = ["latent_actions"]
    elif latent_source == "vq":
        keys = ["quantized_latent_actions"]
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

    if lat.ndim != 3:
        raise RuntimeError(
            f"Latents have shape {tuple(lat.shape)} (ndim={lat.ndim}). Expected [B,T,D] continuous vectors."
        )

    return lat


def _extract_latents_and_visuals(
    model,
    obs,
    act,
    *,
    latent_source: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encode_output = model.encode(obs, act)

    if "visual_embs" not in encode_output or encode_output["visual_embs"] is None:
        raise KeyError("encode_output['visual_embs'] missing; ensure model.encode returns visual_embs.")
    if "z" not in encode_output or encode_output["z"] is None:
        raise KeyError("encode_output['z'] missing; ensure model.encode returns z.")
    if getattr(model, "predictor", None) is None:
        raise RuntimeError("model.predictor is required to compute P_t1_hat.")

    z = encode_output["z"]
    num_pred = int(getattr(model, "num_pred", 1))
    t_pred = z.shape[1] - num_pred
    if t_pred <= 0:
        raise RuntimeError(f"Not enough steps to predict: z.shape[1]={z.shape[1]}, num_pred={num_pred}")

    z_src = z[:, :t_pred]
    z_pred = model.predict(z_src)
    pred_obs, _ = model.separate_emb(z_pred)

    if "visual" not in pred_obs or pred_obs["visual"] is None:
        raise KeyError("pred_obs['visual'] missing after model.separate_emb(z_pred).")

    latents = _extract_latents(encode_output, latent_source=latent_source)
    p_t = encode_output["visual_embs"][:, :t_pred]
    p_t1_hat = pred_obs["visual"]
    return latents, p_t, p_t1_hat


def _align_macro_pairs(
    p_t: torch.Tensor,
    p_t1_hat: torch.Tensor,
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align P_t, P_t1_hat, z, and a to the same temporal length and flatten into pairs.
    """
    if p_t.ndim != 4:
        raise ValueError(f"Expected p_t [B,T,P,D], got {tuple(p_t.shape)}")
    if p_t1_hat.ndim != 4:
        raise ValueError(f"Expected p_t1_hat [B,T,P,D], got {tuple(p_t1_hat.shape)}")
    if latents.ndim != 3:
        raise ValueError(f"Expected latents [B,T,D], got {tuple(latents.shape)}")
    if actions.ndim == 2:
        actions = actions.unsqueeze(-1)
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [B,T,A], got {tuple(actions.shape)}")

    t_pair = min(p_t.shape[1], p_t1_hat.shape[1], latents.shape[1], actions.shape[1])
    p_t = p_t[:, :t_pair]
    p_t1_hat = p_t1_hat[:, :t_pair]
    latents = latents[:, :t_pair]
    actions = actions[:, :t_pair]

    b, t, p, d = p_t.shape
    z_dim = latents.shape[-1]
    a_dim = actions.shape[-1]

    p_t = p_t.reshape(b * t, p, d).contiguous()
    p_t1_hat = p_t1_hat.reshape(b * t, p, d).contiguous()
    latents = latents.reshape(b * t, z_dim).contiguous()
    actions = actions.reshape(b * t, a_dim).contiguous()

    return p_t, p_t1_hat, latents, actions


class MacroPairStream:
    """Stateful stream of (P_t, P_t1_hat, z, a) pairs from Trainer dataloader."""
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
    def next_pair_chunk(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        latents, p_t, p_t1_hat = _extract_latents_and_visuals(
            self.model,
            obs,
            act,
            latent_source=self.latent_source,
        )

        p_t, p_t1_hat, z, y = _align_macro_pairs(p_t, p_t1_hat, latents, act)

        # move to CPU for writing; keep float32 here; writer will cast if requested
        return (
            p_t.detach().cpu().float(),
            p_t1_hat.detach().cpu().float(),
            z.detach().cpu().float(),
            y.detach().cpu().float(),
        )


# -----------------------------
# Disk-backed memmap cache (A)
# -----------------------------
class MemmapPairWriter:
    """
    Disk-backed writer for (p_t, p_t1, z, y) pairs.
    Writes float16 by default to cut size ~2x.
    """
    def __init__(
        self,
        root: Path,
        prefix: str,
        n_pairs: int,
        p: int,
        d: int,
        z_dim: int,
        a_dim: int,
        dtype: np.dtype = np.float16,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.n_pairs = int(n_pairs)
        self.p = int(p)
        self.d = int(d)
        self.z_dim = int(z_dim)
        self.a_dim = int(a_dim)
        self.dtype = np.dtype(dtype)

        self.meta_path = self.root / f"{prefix}_meta.json"
        self.p_t_path = self.root / f"{prefix}_p_t.dat"
        self.p_t1_path = self.root / f"{prefix}_p_t1.dat"
        self.z_path = self.root / f"{prefix}_z.dat"
        self.y_path = self.root / f"{prefix}_y.dat"

        self.p_t = np.memmap(self.p_t_path, mode="w+", dtype=self.dtype, shape=(self.n_pairs, self.p, self.d))
        self.p_t1 = np.memmap(self.p_t1_path, mode="w+", dtype=self.dtype, shape=(self.n_pairs, self.p, self.d))
        self.z = np.memmap(self.z_path, mode="w+", dtype=self.dtype, shape=(self.n_pairs, self.z_dim))
        self.y = np.memmap(self.y_path, mode="w+", dtype=self.dtype, shape=(self.n_pairs, self.a_dim))

        self.i = 0  # write cursor

    def append(self, p_t: torch.Tensor, p_t1: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> None:
        if self.i >= self.n_pairs:
            return
        n = int(p_t.shape[0])
        j = min(self.i + n, self.n_pairs)
        take = j - self.i
        if take <= 0:
            return

        # Cast on CPU just-in-time; only slice what we need.
        p_t_np = p_t[:take].contiguous().to(torch.float16).cpu().numpy()
        p_t1_np = p_t1[:take].contiguous().to(torch.float16).cpu().numpy()
        z_np = z[:take].contiguous().to(torch.float16).cpu().numpy()
        y_np = y[:take].contiguous().to(torch.float16).cpu().numpy()

        self.p_t[self.i:j] = p_t_np
        self.p_t1[self.i:j] = p_t1_np
        self.z[self.i:j] = z_np
        self.y[self.i:j] = y_np

        self.i = j

    def finalize(self) -> None:
        self.p_t.flush()
        self.p_t1.flush()
        self.z.flush()
        self.y.flush()
        meta = {
            "n_pairs": self.n_pairs,
            "p": self.p,
            "d": self.d,
            "z_dim": self.z_dim,
            "a_dim": self.a_dim,
            "dtype": str(self.dtype),
            "files": {
                "p_t": self.p_t_path.name,
                "p_t1": self.p_t1_path.name,
                "z": self.z_path.name,
                "y": self.y_path.name,
            },
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)


class MemmapPairDataset(Dataset):
    """Reads pairs lazily from disk-backed memmaps."""
    def __init__(self, root: Path, prefix: str):
        root = Path(root)
        meta_path = root / f"{prefix}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing cache meta: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.root = root
        self.prefix = prefix
        self.n_pairs = int(meta["n_pairs"])
        self.p = int(meta["p"])
        self.d = int(meta["d"])
        self.z_dim = int(meta["z_dim"])
        self.a_dim = int(meta["a_dim"])
        self.dtype = np.dtype(meta["dtype"])

        self.p_t = np.memmap(
            root / meta["files"]["p_t"],
            mode="r",
            dtype=self.dtype,
            shape=(self.n_pairs, self.p, self.d),
        )
        self.p_t1 = np.memmap(
            root / meta["files"]["p_t1"],
            mode="r",
            dtype=self.dtype,
            shape=(self.n_pairs, self.p, self.d),
        )
        self.z = np.memmap(
            root / meta["files"]["z"],
            mode="r",
            dtype=self.dtype,
            shape=(self.n_pairs, self.z_dim),
        )
        self.y = np.memmap(
            root / meta["files"]["y"],
            mode="r",
            dtype=self.dtype,
            shape=(self.n_pairs, self.a_dim),
        )

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int):
        # Return torch tensors (float16 on CPU). We'll cast to float32 on GPU in training/eval.
        return (
            torch.from_numpy(self.p_t[idx]),
            torch.from_numpy(self.p_t1[idx]),
            torch.from_numpy(self.z[idx]),
            torch.from_numpy(self.y[idx]),
        )


def build_memmap_cache(
    stream: MacroPairStream,
    cache_dir: Path,
    prefix: str,
    target_pairs: int,
    *,
    quiet: bool,
    rebuild: bool,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{prefix}_meta.json"

    if rebuild:
        for fp in cache_dir.glob(f"{prefix}_*.dat"):
            try:
                fp.unlink()
            except FileNotFoundError:
                pass
        try:
            meta_path.unlink()
        except FileNotFoundError:
            pass

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        existing_pairs = int(meta.get("n_pairs", 0))
        if existing_pairs >= target_pairs:
            if not quiet:
                print(f"[cache] {prefix}: exists ({meta_path})")
            return
        if not quiet:
            print(f"[cache] {prefix}: exists but too small ({existing_pairs} < {target_pairs}), rebuilding.")
        for fp in cache_dir.glob(f"{prefix}_*.dat"):
            try:
                fp.unlink()
            except FileNotFoundError:
                pass
        try:
            meta_path.unlink()
        except FileNotFoundError:
            pass

    # probe to get shapes
    p_t, p_t1, z, y = stream.next_pair_chunk()
    p = int(p_t.shape[1])
    d = int(p_t.shape[2])
    z_dim = int(z.shape[1])
    a_dim = int(y.shape[1])

    writer = MemmapPairWriter(
        cache_dir,
        prefix=prefix,
        n_pairs=target_pairs,
        p=p,
        d=d,
        z_dim=z_dim,
        a_dim=a_dim,
        dtype=np.float16,
    )

    writer.append(p_t, p_t1, z, y)
    if not quiet:
        print(f"[cache] {prefix}: wrote {writer.i}/{target_pairs}")

    while writer.i < target_pairs:
        p_t, p_t1, z, y = stream.next_pair_chunk()
        writer.append(p_t, p_t1, z, y)
        if not quiet:
            print(f"[cache] {prefix}: wrote {writer.i}/{target_pairs}")

    writer.finalize()
    if not quiet:
        print(f"[cache] {prefix}: done -> {cache_dir}")


# -----------------------------
# Training / Eval
# -----------------------------
@dataclass
class EarlyStopConfig:
    max_epochs: int
    patience: int
    min_delta: float
    log_every: int


def _batch_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str, huber_delta: float) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target, reduction="mean")
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, target, reduction="mean", beta=huber_delta)
    raise ValueError(f"Unknown loss_type={loss_type}")


@torch.no_grad()
def eval_rmse_loader(
    model: MacroActionDecoder,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Streaming RMSE: does NOT store all predictions in RAM (fixes eval RAM spikes).
    """
    model.eval()
    sse = 0.0
    n_el = 0
    batch_rmse_mean = 0.0
    batch_rmse_m2 = 0.0
    n_batches = 0
    for xb, xb1, zb, yb in loader:
        xb = xb.to(device, non_blocking=True, dtype=torch.float32)
        xb1 = xb1.to(device, non_blocking=True, dtype=torch.float32)
        zb = zb.to(device, non_blocking=True, dtype=torch.float32)
        yb = yb.to(device, non_blocking=True, dtype=torch.float32)

        pred = model(xb, xb1, zb)
        diff = pred - yb
        batch_rmse = float(diff.pow(2).mean().sqrt().item())
        sse += float(diff.pow(2).sum().item())
        n_el += int(diff.numel())
        n_batches += 1
        delta = batch_rmse - batch_rmse_mean
        batch_rmse_mean += delta / n_batches
        batch_rmse_m2 += delta * (batch_rmse - batch_rmse_mean)

    rmse = float((sse / max(n_el, 1)) ** 0.5)
    if n_batches > 1:
        rmse_std = float((batch_rmse_m2 / (n_batches - 1)) ** 0.5)
    else:
        rmse_std = 0.0
    return rmse, rmse_std


def train_with_early_stopping(
    model: MacroActionDecoder,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    es: EarlyStopConfig,
    grad_clip: Optional[float],
    loss_type: str,
    huber_delta: float,
) -> Dict[str, object]:
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, es.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, xb1, zb, yb in train_loader:
            xb = xb.to(device, non_blocking=True, dtype=torch.float32)
            xb1 = xb1.to(device, non_blocking=True, dtype=torch.float32)
            zb = zb.to(device, non_blocking=True, dtype=torch.float32)
            yb = yb.to(device, non_blocking=True, dtype=torch.float32)

            optim.zero_grad(set_to_none=True)
            pred = model(xb, xb1, zb)
            loss = _batch_loss(pred, yb, loss_type=loss_type, huber_delta=huber_delta)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        val_rmse, val_rmse_std = eval_rmse_loader(model, val_loader, device=device)

        improved = val_rmse < best_rmse - es.min_delta
        if improved:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if es.log_every > 0 and (epoch % es.log_every == 0 or improved or bad_epochs >= es.patience):
            flag = " *" if improved else ""
            print(
                f"    [epoch {epoch:4d}]{flag} train_loss={train_loss:.6f} "
                f"val_rmse={val_rmse:.6f}±{val_rmse_std:.6f} best={best_rmse:.6f} "
                f"bad={bad_epochs}/{es.patience}"
            )

        if bad_epochs >= es.patience:
            break

    model.load_state_dict(best_state)
    final_rmse, final_rmse_std = eval_rmse_loader(model, val_loader, device=device)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "final_rmse": final_rmse,
        "final_rmse_std": final_rmse_std,
    }


def main():
    ap = argparse.ArgumentParser(description="Train MacroActionDecoder on patch tokens + latent actions (disk-backed cache).")
    ap.add_argument("--run_dir", type=str, help="Path to Hydra run directory.")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint filename or absolute path.")
    ap.add_argument("--latent_source", type=str, default="continuous", choices=["continuous", "vq"])
    ap.add_argument("--train_pairs", type=str, default="50000")
    ap.add_argument("--val_pairs", type=int, default=10000)

    ap.add_argument("--pair_batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)
    ap.add_argument("--loss_type", type=str, default="huber", choices=["huber", "mse"])
    ap.add_argument("--huber_delta", type=float, default=1.0)

    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=1)

    ap.add_argument("--disable_e", action="store_true")
    ap.add_argument("--disable_delta", action="store_true")
    ap.add_argument("--disable_z", action="store_true")

    ap.add_argument("--quiet_sampling", action="store_true")

    # cache / dataloader knobs
    ap.add_argument("--pair_cache_dir", type=str, default="macro_pair_cache", help="Directory under run_dir to store memmap cache.")
    ap.add_argument("--rebuild_cache", action="store_true", help="Delete and rebuild cached pairs.")
    ap.add_argument("--loader_workers", type=int, default=2, help="DataLoader workers for memmap dataset.")
    ap.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory for memmap loaders.")

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

    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")
    print("")

    pair_cache_dir = Path(args.pair_cache_dir)
    cache_dir = pair_cache_dir if pair_cache_dir.is_absolute() else (run_dir / pair_cache_dir).resolve()

    pin_memory = (not args.no_pin_memory) and torch.cuda.is_available()
    train_pair_sizes = _parse_pair_sizes(args.train_pairs)
    max_train_pairs = max(train_pair_sizes)

    # Build / reuse validation cache
    val_stream = MacroPairStream(trainer, split="valid", latent_source=args.latent_source, device=device)
    build_memmap_cache(
        val_stream,
        cache_dir=cache_dir,
        prefix="valid",
        target_pairs=int(args.val_pairs),
        quiet=args.quiet_sampling,
        rebuild=bool(args.rebuild_cache),
    )

    # Build / reuse train cache
    train_stream = MacroPairStream(trainer, split="train", latent_source=args.latent_source, device=device)
    build_memmap_cache(
        train_stream,
        cache_dir=cache_dir,
        prefix="train",
        target_pairs=int(max_train_pairs),
        quiet=args.quiet_sampling,
        rebuild=bool(args.rebuild_cache),
    )

    # Create datasets/loaders from memmaps
    val_ds = MemmapPairDataset(cache_dir, "valid")
    train_ds = MemmapPairDataset(cache_dir, "train")

    # Basic shape consistency check (cheap)
    if val_ds.p != train_ds.p or val_ds.d != train_ds.d or val_ds.z_dim != train_ds.z_dim or val_ds.a_dim != train_ds.a_dim:
        raise RuntimeError(
            "Token/latent/action shape mismatch between train and valid caches:\n"
            f"  train: P={train_ds.p}, D={train_ds.d}, z_dim={train_ds.z_dim}, a_dim={train_ds.a_dim}\n"
            f"  valid: P={val_ds.p}, D={val_ds.d}, z_dim={val_ds.z_dim}, a_dim={val_ds.a_dim}"
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.pair_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.loader_workers),
        pin_memory=pin_memory,
        persistent_workers=(int(args.loader_workers) > 0),
    )

    token_dim = int(train_ds.d)
    z_dim = int(train_ds.z_dim)
    out_dim = int(train_ds.a_dim)

    es = EarlyStopConfig(
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        log_every=int(args.log_every),
    )
    grad_clip = float(args.grad_clip) if args.grad_clip and args.grad_clip > 0 else None

    for train_pairs in train_pair_sizes:
        if train_pairs > len(train_ds):
            raise ValueError(
                f"Requested train_pairs={train_pairs} exceeds cached pairs={len(train_ds)}. "
                "Rebuild cache with a larger train_pairs value."
            )

        if train_pairs < len(train_ds):
            subset_ds = torch.utils.data.Subset(train_ds, range(int(train_pairs)))
        else:
            subset_ds = train_ds

        train_loader = DataLoader(
            subset_ds,
            batch_size=int(args.pair_batch_size),
            shuffle=True,
            drop_last=False,
            num_workers=int(args.loader_workers),
            pin_memory=pin_memory,
            persistent_workers=(int(args.loader_workers) > 0),
        )

        model = MacroActionDecoder(
            token_dim=token_dim,
            z_dim=z_dim,
            out_dim=out_dim,
            disable_e=args.disable_e,
            disable_delta=args.disable_delta,
            disable_z=args.disable_z,
        )

        results = train_with_early_stopping(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=int(args.pair_batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=device,
            es=es,
            grad_clip=grad_clip,
            loss_type=str(args.loss_type),
            huber_delta=float(args.huber_delta),
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_key = f"action_decoder_{int(train_pairs)}"
        ckpt[ckpt_key] = {k: v.detach().cpu() for k, v in results["model"].state_dict().items()}
        torch.save(ckpt, ckpt_path)

        print("")
        print(
            f"Done ({train_pairs} pairs). best_epoch={results['best_epoch']} "
            f"best_rmse={results['best_rmse']:.6f} final_rmse={results['final_rmse']:.6f}"
            f"±{results['final_rmse_std']:.6f}"
        )


if __name__ == "__main__":
    main()

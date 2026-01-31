import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

        return (
            p_t.detach().cpu().float(),
            p_t1_hat.detach().cpu().float(),
            z.detach().cpu().float(),
            y.detach().cpu().float(),
        )


class MacroPairBuffer:
    """Accumulates macro-action pairs on CPU without quadratic cat overhead."""
    def __init__(self):
        self.p_t_chunks: List[torch.Tensor] = []
        self.p_t1_chunks: List[torch.Tensor] = []
        self.z_chunks: List[torch.Tensor] = []
        self.y_chunks: List[torch.Tensor] = []
        self.n_pairs: int = 0
        self.token_dim: Optional[int] = None
        self.z_dim: Optional[int] = None
        self.action_dim: Optional[int] = None

    def append(self, p_t: torch.Tensor, p_t1: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> None:
        assert p_t.ndim == 3 and p_t1.ndim == 3 and z.ndim == 2 and y.ndim == 2
        assert p_t.shape == p_t1.shape
        assert p_t.shape[0] == z.shape[0] == y.shape[0]
        if self.token_dim is None:
            self.token_dim = int(p_t.shape[-1])
            self.z_dim = int(z.shape[1])
            self.action_dim = int(y.shape[1])
        self.p_t_chunks.append(p_t)
        self.p_t1_chunks.append(p_t1)
        self.z_chunks.append(z)
        self.y_chunks.append(y)
        self.n_pairs += int(p_t.shape[0])

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p_t = torch.cat(self.p_t_chunks, dim=0) if self.p_t_chunks else torch.empty(0)
        p_t1 = torch.cat(self.p_t1_chunks, dim=0) if self.p_t1_chunks else torch.empty(0)
        z = torch.cat(self.z_chunks, dim=0) if self.z_chunks else torch.empty(0)
        y = torch.cat(self.y_chunks, dim=0) if self.y_chunks else torch.empty(0)
        return p_t, p_t1, z, y


def fill_buffer_to(stream: MacroPairStream, buf: MacroPairBuffer, *, target_pairs: int, quiet: bool) -> None:
    while buf.n_pairs < target_pairs:
        p_t, p_t1, z, y = stream.next_pair_chunk()
        buf.append(p_t, p_t1, z, y)
        if not quiet:
            print(f"  collected {buf.n_pairs}/{target_pairs} pairs")


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
def eval_rmse(
    model: MacroActionDecoder,
    p_t: torch.Tensor,
    p_t1: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    preds = []
    n = p_t.shape[0]
    for i in range(0, n, batch_size):
        xb = p_t[i : i + batch_size].to(device, non_blocking=True)
        xb1 = p_t1[i : i + batch_size].to(device, non_blocking=True)
        zb = z[i : i + batch_size].to(device, non_blocking=True)
        pred = model(xb, xb1, zb).detach().cpu()
        preds.append(pred)
    pred_all = torch.cat(preds, dim=0)
    rmse = float((pred_all - y).pow(2).mean().sqrt().item())
    return rmse


def train_with_early_stopping(
    model: MacroActionDecoder,
    *,
    p_t_train: torch.Tensor,
    p_t1_train: torch.Tensor,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    p_t_val: torch.Tensor,
    p_t1_val: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
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

    train_ds = TensorDataset(p_t_train, p_t1_train, z_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_rmse = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, es.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, xb1, zb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            xb1 = xb1.to(device, non_blocking=True)
            zb = zb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad()
            pred = model(xb, xb1, zb)
            loss = _batch_loss(pred, yb, loss_type=loss_type, huber_delta=huber_delta)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_rmse = eval_rmse(
            model,
            p_t_val,
            p_t1_val,
            z_val,
            y_val,
            batch_size=batch_size,
            device=device,
        )

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
                f"val_rmse={val_rmse:.6f} best={best_rmse:.6f} bad={bad_epochs}/{es.patience}"
            )

        if bad_epochs >= es.patience:
            break

    model.load_state_dict(best_state)
    final_rmse = eval_rmse(
        model,
        p_t_val,
        p_t1_val,
        z_val,
        y_val,
        batch_size=batch_size,
        device=device,
    )

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "final_rmse": final_rmse,
    }


def main():
    ap = argparse.ArgumentParser(description="Train MacroActionDecoder on patch tokens + latent actions.")
    ap.add_argument("--run_dir", type=str, help="Path to Hydra run directory.")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint filename or absolute path.")
    ap.add_argument("--latent_source", type=str, default="continuous", choices=["continuous", "vq"])
    ap.add_argument("--train_pairs", type=int, default=50000)
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

    val_stream = MacroPairStream(trainer, split="valid", latent_source=args.latent_source, device=device)
    val_buf = MacroPairBuffer()
    fill_buffer_to(val_stream, val_buf, target_pairs=args.val_pairs, quiet=args.quiet_sampling)
    p_t_val, p_t1_val, z_val, y_val = val_buf.tensors()

    train_stream = MacroPairStream(trainer, split="train", latent_source=args.latent_source, device=device)
    train_buf = MacroPairBuffer()
    fill_buffer_to(train_stream, train_buf, target_pairs=args.train_pairs, quiet=args.quiet_sampling)
    p_t_train, p_t1_train, z_train, y_train = train_buf.tensors()

    if p_t_train.shape[1:] != p_t_val.shape[1:]:
        raise RuntimeError(f"Token shape mismatch: train {p_t_train.shape} vs val {p_t_val.shape}")

    token_dim = int(p_t_train.shape[-1])
    z_dim = int(z_train.shape[-1])
    out_dim = int(y_train.shape[-1])

    model = MacroActionDecoder(
        token_dim=token_dim,
        z_dim=z_dim,
        out_dim=out_dim,
        disable_e=args.disable_e,
        disable_delta=args.disable_delta,
        disable_z=args.disable_z,
    )

    es = EarlyStopConfig(
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        log_every=int(args.log_every),
    )
    grad_clip = float(args.grad_clip) if args.grad_clip and args.grad_clip > 0 else None

    results = train_with_early_stopping(
        model,
        p_t_train=p_t_train,
        p_t1_train=p_t1_train,
        z_train=z_train,
        y_train=y_train,
        p_t_val=p_t_val,
        p_t1_val=p_t1_val,
        z_val=z_val,
        y_val=y_val,
        batch_size=args.pair_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        es=es,
        grad_clip=grad_clip,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
    )

    print("")
    print(
        f"Done. best_epoch={results['best_epoch']} "
        f"best_rmse={results['best_rmse']:.6f} final_rmse={results['final_rmse']:.6f}"
    )


if __name__ == "__main__":
    main()

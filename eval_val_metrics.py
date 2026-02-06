import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict
import math

import torch
from omegaconf import OmegaConf

# --- disable wandb hard ---
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["MPLBACKEND"] = "Agg"


def ensure_hydra_config(run_dir: Path):
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
        hydra_conf.job.name = "eval_val_metrics"

    # Top-level config that contains a structured HydraConf at key "hydra"
    cfg = OmegaConf.create({"hydra": hydra_conf})
    HydraConfig.instance().set_config(cfg)


def add_repo_to_syspath():
    """
    Makes `metrics.*` and `dino_wm_latent_actions.*` importable even after os.chdir(run_dir).
    Assumes this script is stored in the repo root OR in /content with the repo cloned to /content/<repo>.
    """
    # 1) If the script lives in the repo root, this is the correct root.
    script_dir = Path(__file__).resolve().parent

    # Candidate repo roots to try
    candidates = [
        script_dir,
        script_dir / "dino_wm_latent_actions",          # if script is one level above package
        Path("/content/dino_wm_latent_actions"),        # if repo cloned directly into /content
        Path("/content"),                               # fallback
    ]

    # Also try the current working dir before we chdir to run_dir (colab often starts in /content/<repo>)
    candidates.insert(0, Path.cwd().resolve())

    for root in candidates:
        root = root.resolve()
        # case 1: metrics/ exists at repo root
        if (root / "metrics").is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            break
        # case 2: metrics/ exists inside package dir
        if (root / "dino_wm_latent_actions" / "metrics").is_dir():
            pkg = root / "dino_wm_latent_actions"
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            if str(pkg) not in sys.path:
                sys.path.insert(0, str(pkg))
            break


def aggregate_epoch_log(epoch_log):
    out = {}
    for k, (n, s, ssq) in epoch_log.items():
        mean = s / n
        var = ssq / n - mean ** 2
        std = math.sqrt(max(var, 0.0))
        se = std / math.sqrt(n)
        ci95 = 1.96 * se

        out[k] = {
            "mean": mean,
            "std": std,
            "se": se,
            "ci95": ci95,
            "n": n,
        }
    return out


def _compute_latent_prior_stats(trainer, cfg):
    # if not bool(getattr(cfg, "compute_prior_stats", False)):
    #     print("[latent prior] skipped: set compute_prior_stats=true in config.")
    #     return None

    # if not getattr(trainer, "use_lam", False) or getattr(trainer, "use_vq", False):
    #     print("[latent prior] skipped: requires trainer.use_lam=True and trainer.use_vq=False.")
    #     return None

    model = trainer.model
    device = trainer.accelerator.device
    loader = trainer.dataloaders["valid"]
    action_dim = int(getattr(model, "latent_action_dim", 0))
    if action_dim <= 0:
        raise RuntimeError("latent_action_dim must be > 0 to compute latent prior stats.")

    total = 0
    sum_latents = torch.zeros(action_dim, device=device)
    sumsq_latents = torch.zeros(action_dim, device=device)

    with torch.no_grad():
        for batch in loader:
            obs, act, *_ = batch
            if isinstance(obs, dict):
                obs = {k: v.to(device, non_blocking=True) for k, v in obs.items()}
            else:
                obs = obs.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)

            encode_output = model.encode(obs, act)
            u_inv = encode_output.get("latent_actions")
            if u_inv is None:
                raise RuntimeError("latent_actions missing while computing latent prior stats.")

            u_inv = u_inv.float()
            total += u_inv.shape[0] * u_inv.shape[1]
            sum_latents += u_inv.sum(dim=(0, 1))
            sumsq_latents += (u_inv ** 2).sum(dim=(0, 1))

    mu = sum_latents / max(total, 1)
    var = (sumsq_latents / max(total, 1)) - mu ** 2
    sigma = torch.sqrt(var.clamp_min(0.0)) + 1e-6

    stats = {"mu": mu.detach().cpu(), "sigma": sigma.detach().cpu()}
    print(
        f"[latent prior] mu shape={tuple(mu.shape)} sigma shape={tuple(sigma.shape)} "
        f"mean={mu.mean().item():.4f} std={sigma.mean().item():.4f}"
    )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to the hydra run folder (contains hydra.yaml, checkpoints/)")
    ap.add_argument("--ckpt", type=str, default="checkpoints/model_latest.pth", help="Checkpoint path relative to run_dir, or absolute")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    # IMPORTANT: Trainer sets cfg['saved_folder']=os.getcwd()
    os.chdir(run_dir)

    cfg_path = run_dir / "hydra.yaml"
    assert cfg_path.exists(), f"Missing hydra.yaml: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)

    ensure_hydra_config(run_dir)

    # Import Trainer AFTER sys.path is correct (assumes you run from repo root or have module path)
    # If Trainer is not in train.py, change this import accordingly.
    add_repo_to_syspath()
    from dino_wm_latent_actions.train import Trainer  # <-- adjust if needed

    trainer = Trainer(cfg)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (run_dir / ckpt_path).resolve()
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    trainer.load_ckpt(ckpt_path)

    # clean accumulator for exact eval
    trainer.epoch_log = OrderedDict()
    print(f"Global step from checkpoint: {trainer.global_step}")

    #trainer.val()

    metrics = aggregate_epoch_log(trainer.epoch_log)

    print(f"run_dir:    {run_dir}")
    print(f"checkpoint: {ckpt_path}")
    print(f"loaded epoch={trainer.epoch} global_step={trainer.global_step}")
    print("\n=== aggregated validation metrics ===")
    for k in sorted(metrics.keys()):
        m = metrics[k]
        print(
            f"{k:45s} "
            f"mean={m['mean']:.6f} "
            f"std={m['std']:.6f} "
            f"ci95=±{m['ci95']:.6f} "
            f"(N={m['n']})"
        )

    prior_stats = _compute_latent_prior_stats(trainer, cfg)
    if prior_stats is not None:
        stats_path = run_dir / "latent_prior_stats.pt"
        torch.save(prior_stats, stats_path)
        mu_list = [float(x) for x in prior_stats["mu"].tolist()]
        sigma_list = [float(x) for x in prior_stats["sigma"].tolist()]
        print(f"\n[latent prior] saved stats to {stats_path}")
        print("[latent prior] paste into plan config:")
        print("latent_prior:")
        print(f"  mu: {mu_list}")
        print(f"  sigma: {sigma_list}")


if __name__ == "__main__":
    main()

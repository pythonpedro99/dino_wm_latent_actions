import os
import time
import math
import hydra
import torch
import torch.nn as nn
import wandb
import logging
import warnings
import threading
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from collections import deque
import torch.nn.functional as F

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

def two_phase_schedule(global_step, warmup_steps, mix_steps, target_commit):
    warmup_steps = max(int(warmup_steps), 0)
    mix_steps = max(int(mix_steps), 0)

    if global_step < warmup_steps:
        alpha = 0.0
    elif mix_steps == 0:
        alpha = 1.0
    else:
        alpha = (global_step - warmup_steps) / mix_steps

    alpha = float(max(0.0, min(1.0, alpha)))

    if global_step < warmup_steps:
        lambda_commit = 0.0
    elif mix_steps == 0:
        lambda_commit = float(target_commit)
    else:
        lambda_commit = alpha * float(target_commit)

    return alpha, lambda_commit

class JsonlLogger:
    """
    Minimal JSONL logger (one JSON object per line), safe for:
      - DDP/Accelerate (writes only on main process if provided)
      - multi-threaded usage (internal lock)

    Usage:
      logger = JsonlLogger("metrics.jsonl", is_main_process=accelerator.is_main_process)
      logger.log({"step": 10, "loss": 0.123})
    """
    def __init__(self, path: str, is_main_process: bool = True):
        self.is_main_process = bool(is_main_process)
        self.path = Path(path)
        self._lock = threading.Lock()

        if self.is_main_process:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered append
            self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        else:
            self._fh = None

    def log(self, obj: dict):
        """Append obj as a single JSON line. No-op on non-main processes."""
        if not self.is_main_process or self._fh is None:
            return

        with self._lock:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()  # keep it minimal but robust

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.dataset.frameskip}_h{self.cfg.model.num_hist}_p{self.cfg.model.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        log_path = Path(self.cfg.saved_folder) / "diagnosis" / "metrics.jsonl"
        self.jsonl = JsonlLogger(str(log_path), is_main_process=self.accelerator.is_main_process)

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        self.global_step = 0
        
        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.model.num_hist,
            num_pred=self.cfg.model.num_pred,
            frameskip=self.cfg.dataset.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        # Deterministic shuffling for train; deterministic (non-shuffled) order for valid
        g_train = torch.Generator().manual_seed(self.cfg.training.seed)

        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                self.datasets["train"],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=True,
                generator=g_train,
                num_workers=self.cfg.env.num_workers,
                prefetch_factor=2 if self.cfg.env.num_workers > 0 else None,
                collate_fn=None,
                persistent_workers=True,
                pin_memory=True
            ),
            "valid": torch.utils.data.DataLoader(
                self.datasets["valid"],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False,
                num_workers=0,
                prefetch_factor=None,
                collate_fn=None,
                persistent_workers=False,
                pin_memory=False
            ),
        }


        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.predictor = None
        self.decoder = None
        self.latent_action_model = None
        self.vq_model = None
        self.latent_action_down = None
        self.model = None

        self.encoder_optimizer = None
        self.predictor_optimizer = None
        self.decoder_optimizer = None
        self.action_encoder_optimizer = None
        self.latent_optimizer = None


        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        self.train_lam = self.cfg.model.train_lam
        self.train_action_encoder = self.cfg.model.train_action_encoder

        self.use_action_encoder = self.cfg.model.use_action_encoder
        self.use_lam = self.cfg.model.use_lam
        self.plan_action_type = self.cfg.model.plan_action_type
        self.swap_check_every_n_steps = int(self.cfg.metrics.swap_check_every_n_steps)
        self.ppl_check_every_n_steps = int(self.cfg.metrics.ppl_check_every_n_steps)
        self.deadcode_check_every_n_steps = int(self.cfg.metrics.deadcode_check_every_n_steps)

        self.ppl_batch_window = int(self.cfg.metrics.ppl_batch_window)
        self.deadcode_batch_window = int(self.cfg.metrics.deadcode_batch_window)
        self.swap_bad_streak_to_flag = int(self.cfg.metrics.swap_bad_streak_to_flag)

        self.shuffle_u_threshold = float(self.cfg.metrics.shuffle_u_threshold)
        self.shuffle_z_threshold = float(self.cfg.metrics.shuffle_z_threshold)
        self.ppl_norm_threshold = float(self.cfg.metrics.ppl_norm_threshold)
        self.deadcode_threshold = float(self.cfg.metrics.deadcode_threshold)

        self._swap_bad_streak = 0
        self._ppl_idx_window = deque(maxlen=self.ppl_batch_window)
        self._dead_idx_window = deque(maxlen=self.deadcode_batch_window)
        self.codebook_size = self.cfg.model.num_codes


        if self.use_action_encoder == self.use_lam:
            raise ValueError("Invalid config: choose exactly one: model.use_action_encoder XOR model.use_lam")

        if not self.use_action_encoder and self.train_action_encoder:
            raise ValueError("train_action_encoder=True but use_action_encoder=False")
        if not self.use_lam and self.train_lam:
            raise ValueError("train_lam=True but use_lam=False")

        log.info(
            "Mode: use_action_encoder=%s use_lam=%s | train_encoder=%s train_predictor=%s train_decoder=%s train_action_encoder=%s train_lam=%s",
            self.use_action_encoder, self.use_lam,
            self.train_encoder, self.train_predictor, self.train_decoder,
            self.train_action_encoder, self.train_lam
        )

    
        self.init_models()
        self.init_optimizers()

        ckpt_dir = Path(self.cfg.saved_folder) / "checkpoints"
        model_ckpt = ckpt_dir / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")


        self.epoch_log = OrderedDict()


    def save_ckpt(self):
        self.accelerator.wait_for_everyone()

        ckpt_dir = Path(self.cfg.saved_folder) / "checkpoints"
        if self.accelerator.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        ckpt = {
            "epoch": int(getattr(self, "epoch", 0)),
            "global_step": int(getattr(self, "global_step", 0)),
            "model": unwrapped_model.state_dict(),

            "opt_encoder": self.encoder_optimizer.state_dict() if getattr(self, "encoder_optimizer", None) else None,
            "opt_predictor": self.predictor_optimizer.state_dict() if getattr(self, "predictor_optimizer", None) else None,
            "opt_decoder": self.decoder_optimizer.state_dict() if getattr(self, "decoder_optimizer", None) else None,
            "opt_action_encoder": self.action_encoder_optimizer.state_dict() if getattr(self, "action_encoder_optimizer", None) else None,
            "opt_lam": self.latent_optimizer.state_dict() if getattr(self, "latent_optimizer", None) else None,
        }

        if self.accelerator.is_main_process:
            latest_path = ckpt_dir / "model_latest.pth"
            epoch_path = ckpt_dir / f"model_{self.epoch}.pth"

            self.accelerator.save(ckpt, str(latest_path))
            self.accelerator.save(ckpt, str(epoch_path))

            log.info("Saved checkpoint(s) to %s", str(ckpt_dir.resolve()))
            ckpt_path = str(epoch_path.resolve())
        else:
            ckpt_path = None

        self.accelerator.wait_for_everyone()

        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch


    def load_ckpt(self, filename, strict: bool = True):
        filename = str(filename)

        ckpt = torch.load(filename, map_location="cpu")

        if getattr(self, "model", None) is None:
            raise RuntimeError("self.model is None. Call init_models() before load_ckpt().")

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        res = unwrapped_model.load_state_dict(ckpt["model"], strict=strict)
        if hasattr(res, "missing_keys") and self.accelerator.is_main_process:
            if res.missing_keys or res.unexpected_keys:
                log.warning(
                    "Checkpoint load (strict=%s): missing=%s unexpected=%s",
                    strict, res.missing_keys, res.unexpected_keys
                )

        self.epoch = int(ckpt.get("epoch", 0))
        self.global_step = int(ckpt.get("global_step", 0))

        # Load optimizer states if optimizers exist (must be created already)
        if getattr(self, "encoder_optimizer", None) and ckpt.get("opt_encoder") is not None:
            self.encoder_optimizer.load_state_dict(ckpt["opt_encoder"])
        if getattr(self, "predictor_optimizer", None) and ckpt.get("opt_predictor") is not None:
            self.predictor_optimizer.load_state_dict(ckpt["opt_predictor"])
        if getattr(self, "decoder_optimizer", None) and ckpt.get("opt_decoder") is not None:
            self.decoder_optimizer.load_state_dict(ckpt["opt_decoder"])
        if getattr(self, "action_encoder_optimizer", None) and ckpt.get("opt_action_encoder") is not None:
            self.action_encoder_optimizer.load_state_dict(ckpt["opt_action_encoder"])
        if getattr(self, "latent_optimizer", None) and ckpt.get("opt_lam") is not None:
            self.latent_optimizer.load_state_dict(ckpt["opt_lam"])

        self.accelerator.wait_for_everyone()
        return ckpt



    def init_models(self):

        # -------------------------
        # initialize encoder 
        # -------------------------
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(self.cfg.encoder)

        if not self.train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # -------------------------
        # action encoder 
        # -------------------------
        if self.cfg.model.use_action_encoder:
            if self.action_encoder is None:
                self.action_encoder = hydra.utils.instantiate(
                    self.cfg.action_encoder,
                    in_chans=self.datasets["train"].action_dim,
                    emb_dim=self.cfg.model.action_emb_dim,
                )

            action_emb_dim = self.action_encoder.emb_dim
            log.info(f"Action encoder type: {type(self.action_encoder)}  emb_dim={action_emb_dim}")

            # If you have a flag, freeze here (before model prepare)
            if hasattr(self, "train_action_encoder") and not self.train_action_encoder:
                for p in self.action_encoder.parameters():
                    p.requires_grad = False
                self.action_encoder.eval()
        else:
            self.action_encoder = None
            action_emb_dim = 0
            log.info("Action encoder disabled (use_action_encoder=False).")

        # -------------------------
        # predictor (dynamic dim) (PLAIN)
        # -------------------------
        if self.encoder.latent_ndim == 1:
            num_patches = 1
        else:
            decoder_scale = 16
            num_side_patches = self.cfg.dataset.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.model.concat_dim == 0:
            num_patches += 1

        cond_dim_per_step = 0
        if self.cfg.model.concat_dim != 0:
            if self.cfg.model.use_action_encoder:
                cond_dim_per_step = action_emb_dim * self.cfg.model.num_action_repeat
            elif self.cfg.model.use_lam:
                cond_dim_per_step = self.cfg.model.latent_action_dim * self.cfg.model.num_action_repeat
            else:
                cond_dim_per_step = 0

        predictor_dim = self.encoder.emb_dim + (cond_dim_per_step * self.cfg.model.concat_dim)

        if self.cfg.model.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.model.num_hist,
                    dim=predictor_dim,
                )

            if not self.train_predictor:
                for p in self.predictor.parameters():
                    p.requires_grad = False
        else:
            self.predictor = None

        log.info(
            "Predictor dim=%s (encoder=%s, cond_per_step=%s, concat_dim=%s, num_action_repeat=%s, use_action_encoder=%s, use_lam=%s)",
            predictor_dim,
            self.encoder.emb_dim,
            cond_dim_per_step,
            self.cfg.model.concat_dim,
            self.cfg.model.num_action_repeat,
            getattr(self.cfg.model, "use_action_encoder", None),
            getattr(self.cfg.model, "use_lam", None),
        )

        # -------------------------
        # decoder (PLAIN)
        # -------------------------
        if self.cfg.model.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(self.base_path, self.cfg.env.decoder_path)
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,
                    )

            if not self.train_decoder:
                for p in self.decoder.parameters():
                    p.requires_grad = False
        else:
            self.decoder = None

        # -------------------------
        # latent action model (LAM) (optional) (PLAIN)
        # -------------------------
        if self.cfg.model.use_lam:
            if self.latent_action_model is None:
                self.latent_action_model = hydra.utils.instantiate(
                    self.cfg.latent_action_model,
                    in_dim=self.encoder.emb_dim,
                    model_dim=self.encoder.emb_dim,
                    patch_size=getattr(self.encoder, "patch_size", 1),
                )

            self.vq_model = None

            if self.cfg.model.use_lam and self.cfg.model.use_vq:
                self.vq_model = hydra.utils.instantiate(self.cfg.vq_model)

            # latent dimensionality is representation-dependent
            if self.cfg.model.use_vq:
                latent_dim = (
                    self.cfg.model.codebook_splits * self.cfg.model.codebook_dim
                )
            else:
                latent_dim = self.cfg.model.latent_action_dim


            if self.latent_action_down is None:
                self.latent_action_down = nn.Linear(self.encoder.emb_dim, latent_dim)

            if not self.train_lam:
                for module in [self.latent_action_model, self.vq_model, self.latent_action_down]:
                    if module is None:
                        continue
                    for p in module.parameters():
                        p.requires_grad = False


            log.info(
                "LAM enabled (use_lam=True). train_lam=%s, latent_dim=%s",
                self.train_lam,
                latent_dim,
            )
        else:
            self.latent_action_model = None
            self.vq_model = None
            self.latent_action_down = None
            log.info("LAM disabled (use_lam=False).")

        # -------------------------
        # build composite model (PLAIN)
        # -------------------------
        model_kwargs = dict(
        image_size=self.cfg.dataset.img_size,
        num_hist=self.cfg.model.num_hist,
        num_pred=self.cfg.model.num_pred,
        encoder=self.encoder,
        decoder=self.decoder,
        predictor=self.predictor,
        codebook_splits=self.cfg.model.codebook_splits,
        codebook_dim=self.cfg.model.codebook_dim,
        action_dim=self.datasets["train"].action_dim,
        concat_dim=self.cfg.model.concat_dim,
        latent_action_dim=self.cfg.model.latent_action_dim,
        num_action_repeat=self.cfg.model.num_action_repeat,
        train_encoder=self.train_encoder,
        train_predictor=self.train_predictor,
        train_decoder=self.train_decoder,
        train_lam=self.train_lam,
        use_action_encoder=self.cfg.model.use_action_encoder,
        use_lam=self.cfg.model.use_lam,
        use_vq=self.cfg.model.use_vq,
        plan_action_type=self.cfg.model.plan_action_type,
        is_training=self.cfg.model.is_training,
    )

        if self.action_encoder is not None:
            model_kwargs["action_encoder"] = self.action_encoder
        if self.latent_action_model is not None:
            model_kwargs["latent_action_model"] = self.latent_action_model
        if self.vq_model is not None:
            model_kwargs["vq_model"] = self.vq_model
        if self.latent_action_down is not None:
            model_kwargs["latent_action_down"] = self.latent_action_down

        self.model = hydra.utils.instantiate(
                    {"_target_": self.cfg.model._target_},
                    **model_kwargs,
)


        # -------------------------
        # prepare ONCE (no nested DDP)
        # -------------------------
        self.model = self.accelerator.prepare(self.model)

        # Optional: watch AFTER prepare. Also watch submodules through self.model to avoid stale refs.
        if self.accelerator.is_main_process and getattr(self, "wandb_run", None) is not None:
            if getattr(self.cfg.model, "use_action_encoder", False) and hasattr(self.model, "action_encoder"):
                self.wandb_run.watch(self.model.action_encoder)


    def init_optimizers(self):
        # Sanity: init_models must have run
        if getattr(self, "model", None) is None:
            raise RuntimeError("init_models() must be called before init_optimizers().")

        # Encoder optimizer
        if self.train_encoder:
            if getattr(self, "encoder_optimizer", None) is None:
                self.encoder_optimizer = torch.optim.Adam(
                    self.model.encoder.parameters(),
                    lr=self.cfg.training.encoder_lr,
                )
                self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)

        # Predictor optimizer
        if self.cfg.model.has_predictor and getattr(self.model, "predictor", None) is not None and self.train_predictor:
            if getattr(self, "predictor_optimizer", None) is None:
                self.predictor_optimizer = torch.optim.AdamW(
                    self.model.predictor.parameters(),
                    lr=self.cfg.training.predictor_lr,
                )
                self.predictor_optimizer = self.accelerator.prepare(self.predictor_optimizer)

        # Action encoder optimizer (baseline mode)
        if self.cfg.model.use_action_encoder:
            if getattr(self.model, "action_encoder", None) is None:
                raise RuntimeError("use_action_encoder=True but model.action_encoder is None.")

            if getattr(self, "train_action_encoder", False):
                if getattr(self, "action_encoder_optimizer", None) is None:
                    self.action_encoder_optimizer = torch.optim.AdamW(
                        self.model.action_encoder.parameters(),
                        lr=self.cfg.training.action_encoder_lr,
                    )
                    self.action_encoder_optimizer = self.accelerator.prepare(self.action_encoder_optimizer)

        # Decoder optimizer
        if self.cfg.model.has_decoder and getattr(self.model, "decoder", None) is not None and self.train_decoder:
            if getattr(self, "decoder_optimizer", None) is None:
                self.decoder_optimizer = torch.optim.Adam(
                    self.model.decoder.parameters(),
                    lr=self.cfg.training.decoder_lr,
                )
                self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

        # LAM optimizer (LAM mode)
        if self.cfg.model.use_lam and self.train_lam:

            # --- Sanity checks (only what is semantically required) ---
            if getattr(self.model, "latent_action_model", None) is None:
                raise RuntimeError(
                    "train_lam=True but latent_action_model is None"
                )

            if getattr(self.model, "latent_action_down", None) is None:
                raise RuntimeError(
                    "train_lam=True but latent_action_down is None"
                )

            if self.cfg.model.use_vq and getattr(self.model, "vq_model", None) is None:
                raise RuntimeError(
                    "use_vq=True but vq_model is None"
                )

            # --- Optimizer initialization ---
            if getattr(self, "latent_optimizer", None) is None:

                latent_param_groups = [
                    self.model.latent_action_model.parameters(),
                    self.model.latent_action_down.parameters(),
                ]

                # Only include VQ parameters if VQ is actually used
                if self.cfg.model.use_vq:
                    latent_param_groups.append(
                        self.model.vq_model.parameters()
                    )

                latent_params = itertools.chain(*latent_param_groups)

                self.latent_optimizer = torch.optim.AdamW(
                    latent_params,
                    lr=self.cfg.training.latent_lr,
                )
                self.latent_optimizer = self.accelerator.prepare(
                    self.latent_optimizer
                )




    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        for epoch in range(self.epoch + 1, self.total_epochs + 1):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.logs_flash(step=self.global_step)
            self.val()
            self.logs_flash(step=self.global_step)
            if self.cfg.training.save_every_x_epoch > 0 and (self.epoch % self.cfg.training.save_every_x_epoch == 0):
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(
                self.dataloaders["train"],
                desc=f"Epoch {self.epoch} Train",
                disable=not self.accelerator.is_main_process,
                position=0,
            )
        ):
            self.global_step += 1
            save_every_x_steps = int(getattr(self.cfg.training, "save_every_x_steps", 0))
            if save_every_x_steps > 0 and (self.global_step % save_every_x_steps == 0):
                _ckpt_path, _model_name, _model_epoch = self.save_ckpt()
            obs, act,_ = data
            plot = False  # dont plot at all
            self.model.train()
            if not self.train_encoder:
                self.model.encoder.eval()
            two_phase_cfg = getattr(self.cfg, "two_phase", None)
            two_phase_enabled = bool(getattr(two_phase_cfg, "enabled", False)) if two_phase_cfg is not None else False
            warmup_steps = int(getattr(two_phase_cfg, "warmup_steps", 0)) if two_phase_cfg is not None else 0
            mix_steps = int(getattr(two_phase_cfg, "mix_steps", 0)) if two_phase_cfg is not None else 0
            if two_phase_enabled and (warmup_steps > 0 or mix_steps > 0):
                alpha, lambda_commit = two_phase_schedule(
                    self.global_step,
                    warmup_steps,
                    mix_steps,
                    self.cfg.model.commitment,
                )
                if self.model.use_vq and getattr(self.model, "vq_model", None) is not None:
                    self.model.vq_model.commitment = lambda_commit
                self.model.two_phase_enabled = True
                self.model.two_phase_alpha = alpha
                if warmup_steps > 0 and self.global_step < warmup_steps:
                    assert math.isclose(alpha, 0.0), "Expected alpha=0 during warmup."
                    assert math.isclose(lambda_commit, 0.0), "Expected lambda_commit=0 during warmup."
            else:
                self.model.two_phase_enabled = False
                self.model.two_phase_alpha = None
            (
                z_out,
                visual_out,
                visual_reconstructed,
                loss,
                loss_components,
                encode_output,
            ) = self.model(obs, act)

            if self.encoder_optimizer: self.encoder_optimizer.zero_grad()
            if self.decoder_optimizer: self.decoder_optimizer.zero_grad()
            if self.predictor_optimizer: self.predictor_optimizer.zero_grad()
            if self.action_encoder_optimizer: self.action_encoder_optimizer.zero_grad()
            if self.latent_optimizer: self.latent_optimizer.zero_grad()
            
            self.accelerator.backward(loss)

            if self.encoder_optimizer and self.model.train_encoder:
                self.encoder_optimizer.step()

            if self.decoder_optimizer and self.cfg.model.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()

            if self.predictor_optimizer and self.cfg.model.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()

            if self.action_encoder_optimizer:
                self.action_encoder_optimizer.step()

            if self.latent_optimizer:
                self.latent_optimizer.step()


            loss = self.accelerator.gather_for_metrics(loss).mean()
            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            # ---- collapse metrics ----
            if self.use_lam:
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    
                    z = encode_output["z"].detach()             # [B,T,P',D']
                    z_src = z[:, : self.model.num_hist]         # [B,H,P',D']
                    z_tgt = z[:, self.model.num_pred :]         # [B,H,P',D']

                    # base action vectors aligned with history axis H
                    if self.model.use_vq and (encode_output.get("quantized_latent_actions") is not None):
                        act_base_h = encode_output["quantized_latent_actions"][:, : self.model.num_hist]
                    elif (encode_output.get("latent_actions") is not None):
                        act_base_h = encode_output["latent_actions"][:, : self.model.num_hist]
                    else:
                        _, act_base_h = self.model.separate_emb(z_src)

                   
                    # ----------------------------
                    # swap + shuffle checks (streak-based over consecutive batches after trigger)
                    # log ONLY after the full window
                    # ----------------------------
                    m = self.global_step % self.swap_check_every_n_steps

                    # reset streak at the start of each check-window so "streak" means consecutive batches
                    if m == 0:
                        self._swap_bad_streak = 0

                    # run for the first `swap_bad_streak_to_flag` batches after each trigger
                    if m < self.swap_bad_streak_to_flag:
                        swap_s = float(self.metric_z_swap_score(z_src, z_tgt, act_base_h))
                        delta, mse_base, mse_shuf = self.metric_action_shuffle_delta(z_src, z_tgt, act_base_h)
                        delta = float(delta); mse_base = float(mse_base); mse_shuf = float(mse_shuf)

                        swap_bad = self._swap_is_bad(swap_s, delta)
                        self._swap_bad_streak = (self._swap_bad_streak + 1) if swap_bad else 0
                        swap_flagged = (self._swap_bad_streak >= self.swap_bad_streak_to_flag)

                        # log only at the end of the window
                        if m == (self.swap_bad_streak_to_flag - 1):
                            self.jsonl.log({
                                "step": int(self.global_step),
                                "swap_s": swap_s,
                                "shuffle_delta": delta,
                                "mse_base": mse_base,
                                "mse_shuf": mse_shuf,
                                "swap_bad": bool(swap_bad),
                                "swap_bad_streak": int(self._swap_bad_streak),
                                "swap_flagged": bool(swap_flagged),
                            })


                    # ----------------------------
                    # VQ usage checks (windowed)
                    # ----------------------------
                    if self.model.use_vq and (encode_output.get("vq_outputs") is not None):
                        vq_idx = encode_output["vq_outputs"].get("indices", None)  # [B,T] typically

                        # always append indices when available
                        if vq_idx is not None:
                            self._append_vq_idx(vq_idx)

                        # ppl computed over last ppl_batch_window appended batches
                        if vq_idx is not None and (self.global_step % self.ppl_check_every_n_steps == 0):
                            idx_win = self._cat_window(self._ppl_idx_window)
                            if idx_win is not None:
                                ppl, ppl_norm = self.metric_codebook_ppl(idx_win)
                                ppl = float(ppl); ppl_norm = float(ppl_norm)
                                ppl_bad = (ppl_norm < self.ppl_norm_threshold)

                                self.jsonl.log({
                                    "step": int(self.global_step),
                                    "ppl": ppl,
                                    "ppl_norm": ppl_norm,
                                    "ppl_bad": bool(ppl_bad),
                                    "ppl_window_steps": int(len(self._ppl_idx_window)),
                                })

                        # deadcode computed over last deadcode_batch_window appended batches
                        if vq_idx is not None and (self.global_step % self.deadcode_check_every_n_steps == 0):
                            idx_win = self._cat_window(self._dead_idx_window)
                            if idx_win is not None:
                                dead_rate = float(self.metric_deadcode_rate(idx_win))
                                dead_bad = (dead_rate > self.deadcode_threshold)

                                self.jsonl.log({
                                    "step": int(self.global_step),
                                    "dead_rate": dead_rate,
                                    "dead_bad": bool(dead_bad),
                                    "deadcode_window_steps": int(len(self._dead_idx_window)),
                                })

                if was_training:
                    self.model.train()

            
            if self.cfg.model.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.model.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                    err_logs = self.err_eval(z_obs_out, z_tgt)
                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and len(self.train_traj_dset) > 0 and self.cfg.model.has_predictor:
            with torch.no_grad():
                logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                logs = {f"val_{k}": [float(v.detach().mean().cpu()) if torch.is_tensor(v) else float(v)]
                        for k, v in logs.items()}
                self.logs_update(logs)


        self.accelerator.wait_for_everyone()
        with torch.no_grad():
            for i, data in enumerate(
                tqdm(
                    self.dataloaders["valid"],
                    desc=f"Epoch {self.epoch} Valid",
                    disable=not self.accelerator.is_main_process,
                    position=0,
                )
            ):
                obs, act, _ = data
                plot = False  # disable plotting during val for speed
                z_out, visual_out, visual_reconstructed, loss, loss_components, encode_output = self.model(
                    obs, act
                )

                loss = self.accelerator.gather_for_metrics(loss).mean()

                loss_components = self.accelerator.gather_for_metrics(loss_components)
                loss_components = {
                    key: value.mean().item() for key, value in loss_components.items()
                }

                if self.cfg.model.has_decoder and plot:
                    # only eval images when plotting due to speed
                    if self.cfg.model.has_predictor:
                        z_obs_out, z_act_out = self.model.separate_emb(z_out)
                        z_gt = self.model.encode_obs(obs)
                        z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                        err_logs = self.err_eval(z_obs_out, z_tgt)

                        err_logs = self.accelerator.gather_for_metrics(err_logs)
                        err_logs = {
                            key: value.mean().item() for key, value in err_logs.items()
                        }
                        err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                        self.logs_update(err_logs)

                    if visual_out is not None:
                        for t in range(
                            self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                        ):
                            img_pred_scores = eval_images(
                                visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                            )
                            img_pred_scores = self.accelerator.gather_for_metrics(
                                img_pred_scores
                            )
                            img_pred_scores = {
                                f"val_img_{k}_pred": [v.mean().item()]
                                for k, v in img_pred_scores.items()
                            }
                            self.logs_update(img_pred_scores)

                    if visual_reconstructed is not None:
                        for t in range(obs["visual"].shape[1]):
                            img_reconstruction_scores = eval_images(
                                visual_reconstructed[:, t], obs["visual"][:, t]
                            )
                            img_reconstruction_scores = self.accelerator.gather_for_metrics(
                                img_reconstruction_scores
                            )
                            img_reconstruction_scores = {
                                f"val_img_{k}_reconstructed": [v.mean().item()]
                                for k, v in img_reconstruction_scores.items()
                            }
                            self.logs_update(img_reconstruction_scores)

                    self.plot_samples(
                        obs["visual"],
                        visual_out,
                        visual_reconstructed,
                        self.epoch,
                        batch=i,
                        num_samples=self.num_reconstruct_samples,
                        phase="valid",
                    )
                loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
                self.logs_update(loss_components)


    def openloop_rollout(
            self, dset, num_rollout=200, rand_start_end=True, min_horizon=5, mode="train"
        ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.model.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        logs = {}

        num_past = [(self.cfg.model.num_hist, ""), (1, "_1framestart")]

        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, _,_ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.dataset.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.dataset.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.dataset.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.dataset.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start :
                    start + horizon * self.cfg.dataset.frameskip + 1 :
                    self.cfg.dataset.frameskip
                ]
            act = act[start : start + horizon * self.cfg.dataset.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.dataset.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0) if self.model.use_action_encoder else None

            for past in num_past:
                n_past, postfix = past

                obs_full = {}
                for k in obs.keys():
                    obs_full[k] = obs[k].unsqueeze(0).to(self.device)

                z_obses, z = self.model.rollout(obs_full, actions, num_obs_init=n_past)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(div_loss[k])
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [div_loss[k]]

                if self.cfg.model.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )

        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs
    
    def _append_vq_idx(self, vq_idx: torch.Tensor):
        # store flattened indices on CPU to avoid GPU memory growth
        idx = vq_idx.detach().reshape(-1).to("cpu", non_blocking=True)
        self._ppl_idx_window.append(idx)
        self._dead_idx_window.append(idx)

    def _cat_window(self, window):
        if len(window) == 0:
            return None
        return torch.cat(list(window), dim=0)

    def _swap_is_bad(self, swap_s: float, shuffle_delta: float) -> bool:
        # shuffle_u_threshold: 0.05 (<) => bad if delta < 0.05
        # shuffle_z_threshold: 0.65 (>) => bad if swap_s > 0.65
        return (shuffle_delta < self.shuffle_u_threshold) or (swap_s > self.shuffle_z_threshold)
    
    @torch.no_grad()
    def metric_z_swap_score(self, z_src, z_tgt, act_base_h) -> float:
        """
        Information-bypass (z-swap) score in visual-token space.

        Improvements (no API change):
        1A) average over multiple random permutations (K)
        1C) margin-based comparison (tau)
        1D) feature normalization before distance computations
        """
        tgt_obs, _ = self.model.separate_emb(z_tgt.detach())
        tgt_vis = tgt_obs["visual"]  # [B,H,P,D_vis]

        B, H = act_base_h.shape[:2]
        N = B * H

        # --- config via attributes (optional); safe defaults if not present ---
        K = int(getattr(self, "swap_perm_k", 8))                 # 1A: number of perms to average
        tau_mode = getattr(self, "swap_margin_mode", "relative") # {"none","absolute","relative"}
        tau_abs = float(getattr(self, "swap_margin_abs", 0.0))   # used if tau_mode=="absolute"
        tau_rel = float(getattr(self, "swap_margin_rel", 1e-3))  # used if tau_mode=="relative"
        do_norm = bool(getattr(self, "swap_feat_norm", True))    # 1D

        # flatten once
        tgt_f = tgt_vis.reshape(N, -1)

        # 1D: normalize features to reduce scale effects
        if do_norm:
            tgt_f = F.normalize(tgt_f, dim=1, eps=1e-8)

        act_flat = act_base_h.reshape(N, -1)

        score_acc = 0.0
        # (optional) compute a robust scale for relative margin from target features once
        # We'll compute tau per perm from distances for stability; see below.

        for _ in range(K):
            perm = torch.randperm(N, device=z_src.device)

            act_swapped = act_flat[perm].reshape_as(act_base_h)
            z_src_cf = self.model.replace_actions_from_z(z_src.clone(), act_swapped)
            z_hat = self.model.predict(z_src_cf)

            hat_obs, _ = self.model.separate_emb(z_hat)
            hat_vis = hat_obs["visual"]  # [B,H,P,D_vis]
            hat_f = hat_vis.reshape(N, -1)

            if do_norm:
                hat_f = F.normalize(hat_f, dim=1, eps=1e-8)

            # compute distances
            d_to_j = torch.norm(hat_f - tgt_f[perm], dim=1)
            d_to_i = torch.norm(hat_f - tgt_f, dim=1)

            # 1C: margin-based comparison (optional)
            if tau_mode == "none":
                wins = (d_to_j < d_to_i)
            elif tau_mode == "absolute":
                # require d_i - d_j > tau_abs
                wins = ((d_to_i - d_to_j) > tau_abs)
            else:
                # relative: scale tau by a typical distance in this perm to avoid tuning to a fixed magnitude
                # Use median of d_to_i as a robust scale proxy
                tau = tau_rel * (d_to_i.median().clamp_min(1e-8))
                wins = ((d_to_i - d_to_j) > tau)

            score_acc += wins.float().mean().item()

        return float(score_acc / max(K, 1))


    @torch.no_grad()
    def metric_action_shuffle_delta(self, z_src, z_tgt, act_base_h, eps: float = 1e-8):
        """
        Action-ignoring (action-shuffle) delta in visual-token space:
          delta = (MSE_shuf - MSE_base)/(MSE_base+eps)

        Returns: (delta, mse_base, mse_shuf)
        """
        tgt_obs, _ = self.model.separate_emb(z_tgt.detach())
        tgt_vis = tgt_obs["visual"]

        z_pred = self.model.predict(z_src)
        pred_obs, _ = self.model.separate_emb(z_pred)
        pred_vis = pred_obs["visual"]
        mse_base = float(F.mse_loss(pred_vis, tgt_vis, reduction="mean").item())

        B, H = act_base_h.shape[:2]
        N = B * H
        perm = torch.randperm(N, device=z_src.device)

        act_flat = act_base_h.reshape(N, -1)
        act_shuf = act_flat[perm].reshape_as(act_base_h)

        z_src_shuf = self.model.replace_actions_from_z(z_src.clone(), act_shuf)
        z_pred_shuf = self.model.predict(z_src_shuf)

        pred_obs_shuf, _ = self.model.separate_emb(z_pred_shuf)
        pred_vis_shuf = pred_obs_shuf["visual"]
        mse_shuf = float(F.mse_loss(pred_vis_shuf, tgt_vis, reduction="mean").item())

        delta = float((mse_shuf - mse_base) / (mse_base + eps))
        return delta, mse_base, mse_shuf

    def metric_codebook_ppl(self, vq_indices: torch.Tensor, eps: float = 1e-8):
        """
        Constant-code collapse metric:
          PPL = exp( -sum_k p(k) log(p(k)+eps) )
          PPL_norm = PPL / K

        Uses self.codebook_size as K.
        """
        K = int(self.codebook_size)
        idx = vq_indices.reshape(-1).to(torch.long)
        counts = torch.bincount(idx, minlength=K).float()
        p = counts / (counts.sum() + eps)
        entropy = -(p * (p + eps).log()).sum().item()
        ppl = math.exp(entropy)
        ppl_norm = ppl / float(K)
        return float(ppl), float(ppl_norm)

    def metric_deadcode_rate(self, vq_indices: torch.Tensor):
        """
        Dead-code rate:
          dead_rate = 1 - (#codes used)/K

        Uses self.codebook_size as K.
        """
        K = int(self.codebook_size)
        idx = vq_indices.reshape(-1).to(torch.long)
        counts = torch.bincount(idx, minlength=K)
        used = int((counts > 0).sum().item())
        return float(1.0 - used / float(K))


    def log_train_errors(self, z_out, obs):
        z_obs_out, _ = self.model.separate_emb(z_out)
        z_gt = self.model.encode_obs(obs)
        z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

        err_logs = self.err_eval(z_obs_out, z_tgt)

        err_logs = self.accelerator.gather_for_metrics(err_logs)
        err_logs = {key: value.mean().item() for key, value in err_logs.items()}
        err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}
        self.logs_update(err_logs)


    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["global_step"] = step
        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
            self.jsonl.log(dict(epoch_log))  
        self.epoch_log = OrderedDict()


    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

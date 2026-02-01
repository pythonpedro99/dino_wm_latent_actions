import os
import gym
import json
import hydra
import random
import torch
import torch.nn as nn
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from models.action_decoder import MacroActionDecoder
from utils import cfg_to_dict, seed
from typing import Any, Iterable


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "action_encoder",
    "latent_action_model",
    "vq_model",
    "latent_action_down",
    "latent_decoder"
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=None,
    goal_source=["dset"],
    goal_H=None,
    alpha=None,
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        action_decoder: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.action_decoder = action_decoder
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] + n for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.plan_action_type = cfg_dict["plan_action_type"]
        if self.plan_action_type in {"latent", "discrete"}:
            self.action_dim = self.wm.act_feat_dim
        else:
            self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            plan_action_type=self.plan_action_type,
            action_decoder=self.action_decoder,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
            plan_action_type=self.plan_action_type,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        # valid_traj = [
        #     self.dset[i][0]["visual"].shape[0]
        #     for i in range(len(self.dset))
        #     if self.dset[i][0]["visual"].shape[0] >= traj_len
        # ]
        # if len(valid_traj) == 0:
        #     raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path: Path, required_keys: Iterable[str]) -> dict[str, Any]:
    """
    Loads ONLY the entries listed in required_keys from the checkpoint payload (if present),
    plus mandatory metadata 'epoch'.

    Assumes checkpoints store state_dicts / plain Python containers (recommended):
      - Loads on CPU for robustness.
      - Does not call .to(device) on payload entries (state_dicts are dicts).
    """
    snapshot_path = Path(snapshot_path)
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location="cpu")

    if "epoch" not in payload:
        raise KeyError(f"Checkpoint missing 'epoch' field: {snapshot_path}")

    required_keys_set = set(required_keys)

    result: dict[str, Any] = {"epoch": int(payload["epoch"])}
    loaded_keys: list[str] = []

    for k in required_keys_set:
        if k in payload:
            result[k] = payload[k]
            loaded_keys.append(k)

    result["_loaded_keys"] = sorted(loaded_keys)
    result["_missing_keys"] = sorted(required_keys_set - set(loaded_keys))
    return result


def load_model(
    model_ckpt: Path,
    model_cfg,
    cfg_dict: dict,
    num_action_repeat: int,
    required_keys: set[str],
    device,
):
    """
    Strictly load a VWorldModel checkpoint by instantiating the full composite
    module graph (encoder/predictor/decoder/action_encoder/LAM/VQ/down) exactly
    as training does, then loading with strict=True.

    Returns:
      (model, action_decoder)
    """
    model_ckpt = Path(model_ckpt)
    if not model_ckpt.exists():
        raise FileNotFoundError(
            f"Strict loading requires a checkpoint, but it does not exist: {model_ckpt}"
        )

    # 1) Load checkpoint first
    result = load_ckpt(model_ckpt, required_keys=required_keys)
    print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if result["_missing_keys"]:
        present = sorted(
            [k for k in result.keys() if k not in {"epoch", "_loaded_keys", "_missing_keys"}]
        )
        raise ValueError(
            "Strict loading failed: checkpoint does not contain all required components.\n"
            f"  checkpoint: {model_ckpt}\n"
            f"  missing: {result['_missing_keys']}\n"
            f"  present: {present}\n"
        )

    action_decoder = None

    use_action_encoder = bool(getattr(model_cfg.model, "use_action_encoder"))
    use_lam          = bool(getattr(model_cfg.model, "use_lam"))
    use_vq           = bool(getattr(model_cfg.model, "use_vq"))
    plan_action_type = cfg_dict.get("plan_action_type")
    is_training = cfg_dict.get("is_training")


    encoder = hydra.utils.instantiate(
        model_cfg.encoder,
    )

    # num_patches
    if getattr(encoder, "latent_ndim", None) == 1:
        num_patches = 1
    else:
        decoder_scale = 16
        img_size = int(getattr(model_cfg.dataset, "img_size", 224))
        num_side_patches = img_size // decoder_scale
        num_patches = num_side_patches**2

    # training code: if cfg.concat_dim == 0: num_patches += 2
    concat_dim = int(getattr(model_cfg.model, "concat_dim"))
    if concat_dim == 0:
        num_patches += 1

   
    cond_dim_per_step = 0
    if concat_dim != 0:
        if use_action_encoder:
            action_emb_dim = int(getattr(model_cfg.model, "action_emb_dim", 0) or 0)
            if action_emb_dim <= 0:
                raise ValueError(
                    "use_action_encoder=True but model_cfg.model.action_emb_dim is missing/invalid."
                )
            cond_dim_per_step = action_emb_dim * int(num_action_repeat)
        elif use_lam:
            cond_dim_per_step = int(model_cfg.model.latent_action_dim) * int(num_action_repeat)
        else:
            cond_dim_per_step = 0


    # predictor_dim
    predictor_dim = int(getattr(encoder, "emb_dim")) + (cond_dim_per_step * concat_dim)

    # instantiate predictor (only if enabled)
    predictor = None
    has_predictor = bool(getattr(model_cfg.model, "has_predictor", True))
    if has_predictor:
        predictor = hydra.utils.instantiate(
            model_cfg.predictor,
            num_patches=num_patches,
            num_frames=int(getattr(model_cfg.model, "num_hist", 1)),
            dim=predictor_dim,
        )


    action_encoder = None
    if use_action_encoder:
        action_encoder_cfg = getattr(model_cfg, "action_encoder", None)
        if action_encoder_cfg is None:
            raise ValueError("use_action_encoder=True but model_cfg.action_encoder is not configured.")


        action_dim = cfg_dict.get("action_dim", 0)
        if action_dim <= 0:
            raise ValueError(
                "Action encoder requires a valid action_dim. Provide cfg_dict['action_dim'] "
                "or set model_cfg.dataset.action_dim."
            )

        frameskip = int(getattr(getattr(model_cfg, "dataset", None), "frameskip", 1))

        action_encoder = hydra.utils.instantiate(
            action_encoder_cfg,
            in_chans=action_dim * frameskip,
            emb_dim=int(model_cfg.model.action_emb_dim),
        )


    latent_action_model = None
    vq_model = None
    latent_action_down = None
    if use_lam:
        lam_cfg = getattr(model_cfg, "latent_action_model", None)
        if lam_cfg is None:
            raise ValueError("use_lam=True but model_cfg.latent_action_model is not configured.")
        latent_action_model = hydra.utils.instantiate(
        lam_cfg,
        in_dim=int(getattr(encoder, "emb_dim")),
        model_dim=int(getattr(encoder, "emb_dim")),
        patch_size=int(getattr(encoder, "patch_size", 1)),
        )

        if use_vq:
            vq_cfg = getattr(model_cfg, "vq_model", None)
            if vq_cfg is None:
                raise ValueError("use_vq=True but model_cfg.vq_model is not configured.")
            vq_model = hydra.utils.instantiate(vq_cfg)
            latent_dim = int(model_cfg.model.codebook_splits) * int(model_cfg.model.codebook_dim)
        else:
            latent_dim = int(model_cfg.model.latent_action_dim)

        latent_action_down = nn.Linear(int(getattr(encoder, "emb_dim")), latent_dim)

    model = hydra.utils.instantiate(
        {"_target_": model_cfg.model._target_},
        encoder=encoder,
        predictor=predictor,
        decoder=None,
        action_encoder=action_encoder,
        latent_action_model=latent_action_model,
        vq_model=vq_model,
        latent_action_down=latent_action_down,
        image_size=model_cfg.dataset.img_size,
        num_hist=1,
        num_pred=model_cfg.model.num_pred,
        codebook_splits=model_cfg.model.codebook_splits,
        codebook_dim=model_cfg.model.codebook_dim,
        action_dim=model_cfg.model.action_emb_dim,
        concat_dim=model_cfg.model.concat_dim,
        latent_action_dim=model_cfg.model.latent_action_dim,
        num_action_repeat=num_action_repeat,
        train_encoder=model_cfg.model.train_encoder,
        train_predictor=model_cfg.model.train_predictor,
        train_decoder=model_cfg.model.train_decoder,
        train_lam=model_cfg.model.train_lam,
        use_action_encoder=use_action_encoder,
        use_lam=use_lam,
        use_vq=use_vq,
        plan_action_type=plan_action_type,
        is_training=is_training,
    )


    
    model.load_state_dict(result["model"], strict=True)

    model.to(device)
    model.eval()

    if plan_action_type in {"latent", "discrete"}:
        action_decoder = MacroActionDecoder(
            token_dim=int(getattr(encoder, "emb_dim")),
            z_dim=int(model.act_feat_dim),
            out_dim=int(cfg_dict.get("action_dim", 0)) * int(model_cfg.dataset.frameskip),
            disable_e=bool(cfg_dict.get("action_decoder_disable_e", False)),
            disable_delta=bool(cfg_dict.get("action_decoder_disable_delta", False)),
            disable_z=bool(cfg_dict.get("action_decoder_disable_z", False)),
        )
        if "action_decoder" in result:
            action_decoder.load_state_dict(result["action_decoder"], strict=True)
        action_decoder.to(device)
        action_decoder.eval()
    return model, action_decoder




class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    required_keys = {"model"}
    plan_action_type = cfg_dict["plan_action_type"]
    use_action_encoder = model_cfg["use_action_encoder"]
    

    # if (not use_action_encoder) and plan_action_type in {"latent", "discrete"}:
    #     required_keys.add("action_decoder")

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.model.num_hist,
        num_pred=model_cfg.model.num_pred,
        frameskip=model_cfg.dataset.frameskip,
    )
    dset = dset["valid"]
    cfg_dict["action_dim"] = dset.action_dim

    num_action_repeat = model_cfg.model.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model, action_decoder = load_model(model_ckpt, model_cfg,cfg_dict, num_action_repeat,required_keys, device=device)

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        action_decoder=action_decoder,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.dataset.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan_pusht")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()

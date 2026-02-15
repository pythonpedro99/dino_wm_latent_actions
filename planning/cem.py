import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        plan_action_type,
        logging_prefix="plan_0",
        log_filename="logs.json",
        latent_prior_mu=None,
        latent_prior_sigma=None,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
            plan_action_type,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

        self.latent_prior_mu = (
            torch.as_tensor(latent_prior_mu) if latent_prior_mu is not None else None
        )
        self.latent_prior_sigma = (
            torch.as_tensor(latent_prior_sigma) if latent_prior_sigma is not None else None
        )
        self.latent_prior_k = float(kwargs.get("latent_prior_k", 2.0))
        self.latent_prior_lambda = float(kwargs.get("latent_prior_lambda", 0.01))

        # keep name for backwards compatibility: in Pattern A this means "use prior init + constraints"
        self.use_prior_sampling = bool(kwargs.get("use_prior_sampling", True))

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])

        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions

        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)

        # Pattern A: initialize the CEM distribution from the latent action model prior
        # (only when planning in latent space and no warm-start is provided).
        if (
            actions is None
            and self.use_prior_sampling
            and self.latent_prior_mu is not None
            and self.latent_prior_sigma is not None
            and self.plan_action_type == "latent"
        ):
            prior_mu = self.latent_prior_mu.to(mu.device).view(1, 1, -1)
            prior_sigma = self.latent_prior_sigma.to(mu.device).view(1, 1, -1)
            mu = prior_mu.expand(n_evals, self.horizon, self.action_dim).clone()
            sigma = prior_sigma.expand(n_evals, self.horizon, self.action_dim).clone()

        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        prior_mu = None
        prior_sigma = None
        if (
            self.latent_prior_mu is not None
            and self.latent_prior_sigma is not None
            and self.plan_action_type == "latent"
        ):
            prior_mu = self.latent_prior_mu.to(self.device).view(1, 1, -1)
            prior_sigma = self.latent_prior_sigma.to(self.device).view(1, 1, -1)
            prior_sigma = prior_sigma.clamp_min(1e-6)

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }

                # Pattern A: ALWAYS sample from the current CEM Gaussian N(mu, sigma)
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
                    * sigma[traj]
                    + mu[traj]
                )

                # include mean plan as a candidate (baseline trick)
                action[0] = mu[traj]

                # prior is used as a constraint (clamp) to stay in-distribution
                if self.use_prior_sampling and prior_mu is not None and prior_sigma is not None:
                    lower = prior_mu - self.latent_prior_k * prior_sigma
                    upper = prior_mu + self.latent_prior_k * prior_sigma
                    action = torch.clamp(action, lower, upper)

                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs=cur_trans_obs_0,
                        act=action,
                        num_obs_init=self.wm.num_hist,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)

                # prior is also used as a soft regularizer (optional)
                if (
                    self.use_prior_sampling
                    and prior_mu is not None
                    and prior_sigma is not None
                    and self.latent_prior_lambda > 0.0
                ):
                    prior_penalty = (((action - prior_mu) / prior_sigma) ** 2).sum(dim=-1)
                    prior_penalty = prior_penalty.mean(dim=-1)
                    loss = loss + self.latent_prior_lambda * prior_penalty

                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())

                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                # keep mu in prior box (defensive)
                if self.use_prior_sampling and prior_mu is not None and prior_sigma is not None:
                    lower = prior_mu.squeeze(0) - self.latent_prior_k * prior_sigma.squeeze(0)
                    upper = prior_mu.squeeze(0) + self.latent_prior_k * prior_sigma.squeeze(0)
                    mu[traj] = torch.clamp(mu[traj], lower, upper)

            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1})

            if self.plan_action_type == "latent":
                u_mean = mu.mean().item()
                u_std = mu.std().item()
                u_min = mu.min().item()
                u_max = mu.max().item()
                print(
                    f"[{self.logging_prefix}] step={i + 1} "
                    f"u.mean={u_mean:.4f} u.std={u_std:.4f} u.min={u_min:.4f} u.max={u_max:.4f}"
                )

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu, np.full(n_evals, np.inf)  # all actions are valid

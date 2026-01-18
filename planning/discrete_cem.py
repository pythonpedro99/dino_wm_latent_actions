import torch
import torch.nn.functional as F
import numpy as np
from einops import repeat

from .base_planner import BasePlanner
from utils import move_to_device


class DiscreteCEMPlanner(BasePlanner):
    """
    Discrete CEM planner over the VQ codebook on self.wm.vq_model.

    Assumptions (matches your vq_model):
      - self.wm.vq_model.embedding: nn.Parameter of shape (K, code_dim)
      - self.wm.vq_model.codebook_splits = S
      - self.wm.vq_model.code_dim = D
      - self.wm.vq_model.num_codes = K
      - action_dim == S * D

    Planning is over discrete indices per (timestep, split):
      - categorical pi[b, t, s, k]
      - sample indices j ~ Cat(pi[t,s,:]) (factorized across t and s)
      - decode to latent-action vectors by concatenating codebook embeddings per split
    """

    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,  # kept for signature compatibility (unused)
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
        self.horizon = int(horizon)
        self.topk = int(topk)               # M elites
        self.num_samples = int(num_samples) # N population
        self.var_scale = var_scale          # unused; retained for compatibility
        self.opt_steps = int(opt_steps)     # I iterations
        self.eval_every = int(eval_every)
        self.logging_prefix = logging_prefix

        # Discrete-CEM hyperparams
        self.pseudocount = float(kwargs.get("pseudocount"))
        self.momentum = float(kwargs.get("momentum"))
        self.min_prob = float(kwargs.get("min_prob"))
        self.init_smoothing = float(kwargs.get("init_smoothing"))
        self.sample_temperature = float(kwargs.get("sample_temperature"))
        self.use_mode_as_first_sample = bool(kwargs.get("use_mode_as_first_sample"))

        # Return selection:
        #  - return_mode=True returns argmax(pi) plan (discrete analogue of mean in continuous CEM)
        #  - return_mode=False returns best sampled plan seen so far
        self.return_mode = bool(kwargs.get("return_mode", True))

    # -------------------------
    # VQ model / codebook access
    # -------------------------
    def _get_vq(self):
        if not hasattr(self.wm, "vq_model"):
            raise AttributeError("DiscreteCEMPlanner expects self.wm.vq_model to exist.")
        vq = self.wm.vq_model
        needed = ["embedding", "code_dim", "codebook_splits", "num_codes"]
        for n in needed:
            if not hasattr(vq, n):
                raise AttributeError(f"self.wm.vq_model is missing attribute '{n}'.")
        return vq

    def _get_codebook(self) -> torch.Tensor:
        """
        Returns codebook embedding matrix of shape (K, code_dim) on self.device.
        """
        vq = self._get_vq()
        cb = vq.embedding
        if not isinstance(cb, torch.Tensor):
            cb = torch.as_tensor(cb)
        cb = cb.to(device=self.device, dtype=torch.float32)
        K, D = int(vq.num_codes), int(vq.code_dim)
        if cb.ndim != 2 or cb.shape[0] != K or cb.shape[1] != D:
            raise ValueError(f"Expected vq_model.embedding shape ({K}, {D}), got {tuple(cb.shape)}.")
        return cb

    def _check_action_dim(self):
        vq = self._get_vq()
        expected = int(vq.codebook_splits) * int(vq.code_dim)
        if int(self.action_dim) != expected:
            raise ValueError(
                f"action_dim mismatch: action_dim={int(self.action_dim)} "
                f"but vq_model.codebook_splits*code_dim={expected} "
                f"(splits={int(vq.codebook_splits)}, code_dim={int(vq.code_dim)})."
            )

    # -------------------------
    # Discrete-CEM helpers
    # -------------------------
    def _apply_temperature(self, probs: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature is None or float(temperature) == 1.0:
            return probs
        logits = (probs.clamp_min(self.min_prob)).log() / float(temperature)
        return torch.softmax(logits, dim=-1)

    def _decode_indices_to_actions(self, indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        indices: (..., S) int64
        codebook: (K, D)
        returns: (..., S*D) float32
        """
        z = codebook[indices]  # (..., S, D)
        return z.reshape(*z.shape[:-2], z.shape[-2] * z.shape[-1])

    def _mle_update(self, elite_indices: torch.Tensor, K: int) -> torch.Tensor:
        """
        elite_indices: (M, H, S) int64
        Returns pi_mle: (H, S, K)
        """
        onehot = F.one_hot(elite_indices, num_classes=K).to(torch.float32)  # (M,H,S,K)
        counts = onehot.sum(dim=0)  # (H,S,K)
        if self.pseudocount > 0:
            counts = counts + self.pseudocount
        pi_mle = counts / counts.sum(dim=-1, keepdim=True)
        pi_mle = pi_mle.clamp_min(self.min_prob)
        pi_mle = pi_mle / pi_mle.sum(dim=-1, keepdim=True)
        return pi_mle

    def _momentum_update(self, pi_old: torch.Tensor, pi_mle: torch.Tensor) -> torch.Tensor:
        m = self.momentum
        pi_new = m * pi_old + (1.0 - m) * pi_mle
        pi_new = pi_new.clamp_min(self.min_prob)
        pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True)
        return pi_new

    def _pi_mode_actions(self, pi: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        pi: (B, H, S, K) -> mode indices (B, H, S) -> actions (B, H, S*D)
        """
        idx = torch.argmax(pi, dim=-1)  # (B,H,S)
        return self._decode_indices_to_actions(idx, codebook)

    def _nearest_code_indices_splitwise(
        self, actions: torch.Tensor, codebook: torch.Tensor, S: int, D: int
    ) -> torch.Tensor:
        """
        actions: (B, T, S*D)
        returns indices: (B, T, S) via splitwise nearest neighbor in codebook.
        """
        B, T, AD = actions.shape
        if AD != S * D:
            raise ValueError(f"Expected actions last dim {S*D}, got {AD}.")
        z = actions.reshape(B, T, S, D).to(torch.float32)     # (B,T,S,D)
        flat = z.reshape(B * T * S, D)                        # (B*T*S, D)
        dists = torch.cdist(flat, codebook)                   # (B*T*S, K)
        idx = torch.argmin(dists, dim=-1)
        return idx.reshape(B, T, S)

    # -------------------------
    # Initialization
    # -------------------------
    def init_pi(self, obs_0, actions=None) -> torch.Tensor:
        """
        Initialize factorized categorical distribution pi over (timestep, split).
        pi shape: (B, H, S, K)
        Optional warm-start from actions by mapping to nearest codes per split.
        """
        self._check_action_dim()
        vq = self._get_vq()
        S, D, K = int(vq.codebook_splits), int(vq.code_dim), int(vq.num_codes)
        codebook = self._get_codebook()

        n_evals = obs_0["visual"].shape[0]
        pi = torch.full((n_evals, self.horizon, S, K), 1.0 / K, dtype=torch.float32, device=self.device)

        if actions is None:
            return pi

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = actions.to(self.device)

        T = min(int(actions.shape[1]), self.horizon)
        if T <= 0:
            return pi

        idx = self._nearest_code_indices_splitwise(actions[:, :T], codebook, S=S, D=D)  # (B,T,S)
        onehot = F.one_hot(idx, num_classes=K).to(torch.float32)                        # (B,T,S,K)

        eps = float(self.init_smoothing)
        pi[:, :T] = onehot * (1.0 - eps) + eps / K
        pi[:, :T] = pi[:, :T] / pi[:, :T].sum(dim=-1, keepdim=True)
        return pi

    # -------------------------
    # Planning
    # -------------------------
    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: optional warm-start action sequence (normalized),
                     (B, T, action_dim), T <= horizon

        Returns:
            planned_actions: (B, H, action_dim) torch.Tensor
            validity: np.ndarray (B,) filled with +inf (mirrors existing planner behavior)
        """
        self._check_action_dim()
        vq = self._get_vq()
        S, D, K = int(vq.codebook_splits), int(vq.code_dim), int(vq.num_codes)
        codebook = self._get_codebook()

        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        pi = self.init_pi(obs_0, actions=actions)  # (B, H, S, K)
        n_evals = pi.shape[0]

        best_loss = torch.full((n_evals,), float("inf"), device=self.device)
        best_indices = torch.zeros((n_evals, self.horizon, S), dtype=torch.long, device=self.device)

        for i in range(self.opt_steps):
            iter_best_losses = []

            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }

                probs = pi[traj]  # (H, S, K)
                sample_probs = self._apply_temperature(probs, self.sample_temperature)

                dist = torch.distributions.Categorical(probs=sample_probs)  # batch over (H,S)
                indices = dist.sample((self.num_samples,))  # (N, H, S)

                if self.use_mode_as_first_sample:
                    indices[0] = torch.argmax(probs, dim=-1)  # (H,S)

                cand_actions = self._decode_indices_to_actions(indices, codebook)  # (N, H, action_dim)

                with torch.no_grad():
                    i_z_obses, _ = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=cand_actions,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g).reshape(-1)  # (N,)

                M = min(self.topk, self.num_samples)
                elite_rank = torch.argsort(loss)[:M]
                elite_indices = indices[elite_rank]  # (M, H, S)
                elite_losses = loss[elite_rank]      # (M,)

                if elite_losses[0] < best_loss[traj]:
                    best_loss[traj] = elite_losses[0]
                    best_indices[traj] = elite_indices[0]

                iter_best_losses.append(elite_losses[0].item())

                pi_mle = self._mle_update(elite_indices, K=K)      # (H,S,K)
                pi[traj] = self._momentum_update(pi[traj], pi_mle)

            mean_loss = float(np.mean(iter_best_losses)) if iter_best_losses else float("nan")
            with torch.no_grad():
                ent = -(pi * (pi.clamp_min(self.min_prob)).log()).sum(dim=-1)  # (B,H,S)
                mean_ent = ent.mean().item()

            self.wandb_run.log(
                {
                    f"{self.logging_prefix}/loss": mean_loss,
                    f"{self.logging_prefix}/entropy": mean_ent,
                    "step": i + 1,
                }
            )

            if self.evaluator is not None and (i % self.eval_every == 0):
                if self.return_mode:
                    eval_actions = self._pi_mode_actions(pi, codebook)
                else:
                    eval_actions = self._decode_indices_to_actions(best_indices, codebook)

                logs, successes, _, _ = self.evaluator.eval_actions(
                    eval_actions, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        if self.return_mode:
            planned_actions = self._pi_mode_actions(pi, codebook)
        else:
            planned_actions = self._decode_indices_to_actions(best_indices, codebook)

        return planned_actions, np.full(n_evals, np.inf)

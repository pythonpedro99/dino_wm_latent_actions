import os
import torch
import torch.nn.functional as F
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils
from models.action_decoder import MacroActionDecoder


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        plan_action_type,
        action_decoder=None,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device
        self.plan_action_type = plan_action_type
        self.action_decoder = action_decoder
        if self.action_decoder is not None:
            self.action_decoder.eval()

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        # last_index: -1 where length==inf, else length-1
        last_index = np.where(length == np.inf, -1, length - 1).astype(np.int64)

        if isinstance(traj_data, torch.Tensor):
            B = traj_data.shape[0]
            idx0 = torch.arange(B, device=traj_data.device)
            idx1 = torch.as_tensor(last_index, device=traj_data.device, dtype=torch.long)
            traj_last = traj_data[idx0, idx1].unsqueeze(1)
            return traj_last
        else:
            traj_last = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
            return traj_last


    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
            
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs=trans_obs_0,
                act=actions,
                num_obs_init=self.wm.num_hist
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        # rollout in env
        print(f"[pre-decode] actions: shape={tuple(actions.shape)} dtype={actions.dtype} device={actions.device} min={actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).min().item():.6g} max={actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).max().item():.6g} mean={actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).mean().item():.6g} std={actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).std(unbiased=False).item():.6g} nan={torch.isnan(actions).sum().item()} inf={torch.isinf(actions).sum().item()}")
        env_actions = actions
        if self.plan_action_type in {"latent", "discrete"}:
            if self.action_decoder is None:
                raise ValueError(
                    "plan_action_type requires an action_decoder for env rollout, but none was provided."
                )
            with torch.no_grad():
                if isinstance(self.action_decoder, MacroActionDecoder):
                    print(f"[macro] i_z_obses['visual']: shape={tuple(i_z_obses['visual'].shape)} env_actions(latent): shape={tuple(env_actions.shape)} frameskip={self.frameskip}")
                    env_actions = self._decode_macro_actions(
                        i_z_obses["visual"],
                        env_actions,
                    )
                    print(f"[post-decode] env_actions: shape={tuple(env_actions.shape)} dtype={env_actions.dtype} device={env_actions.device} min={env_actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).min().item():.6g} max={env_actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).max().item():.6g} mean={env_actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).mean().item():.6g} std={env_actions.detach().nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).std(unbiased=False).item():.6g} nan={torch.isnan(env_actions).sum().item()} inf={torch.isinf(env_actions).sum().item()}")
                else:
                    env_actions = self.action_decoder(env_actions)


        exec_actions = rearrange(
            env_actions.detach().cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        print(
            f"[pre-denorm] exec_actions(tensor): shape={tuple(exec_actions.shape)} dtype={exec_actions.dtype} device={exec_actions.device} "
            f"min={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).min().item():.6g} "
            f"max={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).max().item():.6g} "
            f"mean={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).mean().item():.6g} "
            f"std={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).std(unbiased=False).item():.6g} "
            f"nan={torch.isnan(exec_actions).sum().item()} inf={torch.isinf(exec_actions).sum().item()}"
        )

        exec_actions = self.preprocessor.denormalize_actions(exec_actions)
        print(
            f"[post-denorm] exec_actions(tensor): shape={tuple(exec_actions.shape)} dtype={exec_actions.dtype} device={exec_actions.device} "
            f"min={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).min().item():.6g} "
            f"max={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).max().item():.6g} "
            f"mean={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).mean().item():.6g} "
            f"std={exec_actions.nan_to_num(nan=0.0,posinf=0.0,neginf=0.0).std(unbiased=False).item():.6g} "
            f"nan={torch.isnan(exec_actions).sum().item()} inf={torch.isinf(exec_actions).sum().item()}"
        )

        exec_actions = exec_actions.cpu().numpy()

        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
            z_obs_g=z_obs_g,
        )

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _decode_macro_actions(self, visual_tokens, z_actions):
        if visual_tokens.shape[1] < z_actions.shape[1] + 1:
            raise ValueError(
                "Not enough rollout frames to decode actions: "
                f"got {visual_tokens.shape[1]} frames for {z_actions.shape[1]} actions."
            )
        batch_size, horizon = z_actions.shape[:2]
        p_t = visual_tokens[:, :horizon]
        p_t1 = visual_tokens[:, 1 : horizon + 1]
        p_t = p_t.reshape(batch_size * horizon, *p_t.shape[2:])
        p_t1 = p_t1.reshape(batch_size * horizon, *p_t1.shape[2:])
        z_flat = z_actions.reshape(batch_size * horizon, z_actions.shape[-1])
        decoded_flat = self.action_decoder(p_t, p_t1, z_flat)
        return decoded_flat.reshape(batch_size, horizon, -1)

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs, z_obs_g):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        goal_emb_mse = F.mse_loss(i_z_obs["visual"], z_obs_g["visual"]).item()
        goal_emb_l2 = torch.norm(i_z_obs["visual"] - z_obs_g["visual"]).item()

        logs.update({
                "mean_pixel_l2": mean_visual_dist,
                "mean_wm_env_emb_l2": div_visual_emb,
                "mean_wm_emb_l2": goal_emb_l2,

            })
        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )

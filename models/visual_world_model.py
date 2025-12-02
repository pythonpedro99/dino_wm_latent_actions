import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        latent_action_model,
        latent_vq_model,
        latent_action_down,
        latent_action_up,
        ema_decay=0.99,
        commitment=0.25,
        codebook_splits=None,
        codebook_dim=None,
        proprio_dim=384,
        action_dim=384,
        concat_dim=0,
        latent_action_dim=32,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.latent_action_model = latent_action_model
        self.latent_vq_model = latent_vq_model
        self.latent_action_down = latent_action_down
        self.latent_action_up = latent_action_up
        self.latent_action_norm = nn.LayerNorm(getattr(self.latent_action_down, "out_features", latent_action_dim))
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
        self.latent_action_dim = latent_action_dim
        self.encoder_emb_dim = self.encoder.emb_dim
        self.codebook_splits = codebook_splits
        self.codebook_dim = codebook_dim

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"latent_action_model: {self.latent_action_model}")
        print(f"latent_vq_model: {self.latent_vq_model}")
        print(f"latent_action_down: {self.latent_action_down}")
        print(f"latent_action_up: {self.latent_action_up}")
        print(f"latent_action_norm: {self.latent_action_norm}")
        print(f"latent_action_dim: {self.latent_action_dim}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"encoder_emb_dim: {self.encoder_emb_dim}")


        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act):
        # Encode observation
        z_dct = self.encode_obs(obs)
        act = self.encode_act(act)

        # ----- latent actions (z_a, z_a_down, z_q) -----
        # z_a: output of latent_action_model before downsampling
        z_a = self.latent_action_model(z_dct["visual"])["action_patches"]           # (B, T, F?, A)
        z_a_down = self.latent_action_down(z_a)                                     # (B, T, 1, A) typically

        # Temporal shift / re-alignment as before
        z_a_down_shifted = torch.cat(
            [z_a_down[:, 1:, :, :], z_a_down[:, -1:, :, :]],
            dim=1
        )

        vq_outputs = self.latent_vq_model(z_a_down_shifted)
        quantized_latent_actions = vq_outputs["z_q_st"].squeeze(2)                  # z_q: (B, T, A)
        vq_outputs["indices"] = vq_outputs["indices"].squeeze(2)                    # (B, T)

        proprio_token = torch.zeros_like(z_dct["proprio"].unsqueeze(2))             # (B, T, 1, A_p)

        # ----- concat modes -----
        if self.concat_dim == 0:
            # concat along token-dim; use quantized actions as before
            act_token = quantized_latent_actions.unsqueeze(2)                       # (B, T, 1, A)
            z = torch.cat([z_dct["visual"], proprio_token, act_token], dim=2)

        elif self.concat_dim == 1:
            # concat along feature-dim; gate between z_a_down and z_q per flag
            proprio_tiled = repeat(
                proprio_token, "b t 1 a -> b t f a",
                f=z_dct["visual"].shape[2]
            )
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)

            # decide which latent action representation to use
            use_z_q = getattr(self, "use_z_q_in_concat", False)
            if use_z_q:
                act_base = quantized_latent_actions                                  # (B, T, A)
            else:
                act_base = act #z_a_down_shifted.squeeze(2)                               # (B, T, A) from z_a_down

            act_tiled = repeat(
                act_base.unsqueeze(2), "b t 1 a -> b t f a",
                f=z_dct["visual"].shape[2]
            )
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)

            z = torch.cat([z_dct["visual"], proprio_repeated, act_repeated], dim=3)
        else:
            raise ValueError(f"Unsupported concat_dim: {self.concat_dim}")

        # ----- stats every X steps -----
        # hardcoded interval, using a self.counter as requested
        LOG_INTERVAL = 100
        if not hasattr(self, "counter"):
            self.counter = 0
        self.counter += 1

        if self.counter % LOG_INTERVAL == 0:
            with torch.no_grad():
                stats = {}

                def add_tensor_stats(name, t):
                    # t can be 3D or 4D, we flatten everything except last dim
                    # mean/std are scalar; dim_zero_frac is per-dimension
                    t_flat = t.view(-1, t.shape[-1])
                    stats[f"{name}_mean"] = t_flat.mean().item()
                    stats[f"{name}_std"] = t_flat.std().item()
                    dim_zero_frac = (t_flat == 0).float().mean(dim=0)                # (D,)
                    stats[f"{name}_dim_zero_frac"] = dim_zero_frac.detach().cpu()

                # z_a: pre-downsample, z_a_down: after downsample+shift, z_q: quantized
                add_tensor_stats("z_a", z_a)
                add_tensor_stats("z_a_down", z_a_down_shifted.squeeze(2))
                add_tensor_stats("z_q", quantized_latent_actions)

                # VQ loss (if present in outputs)
                if "loss" in vq_outputs:
                    stats["vq_loss"] = vq_outputs["loss"].detach().cpu()

                # code usage statistics from indices
                indices = vq_outputs["indices"]                                      # (B, T)
                idx_flat = indices.view(-1)
                # infer codebook size if not explicitly stored
                codebook_size = getattr(
                    self.latent_vq_model, "num_embeddings",
                    int(idx_flat.max().item()) + 1
                )
                counts = torch.bincount(idx_flat, minlength=codebook_size)           # (num_codes,)
                stats["code_counts"] = counts.detach().cpu()
                stats["num_used_codes"] = int((counts > 0).sum().item())

                # entropy over code distribution
                total = counts.sum()
                if total > 0:
                    probs = counts.float() / total
                    nonzero = probs > 0
                    entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
                    stats["code_entropy"] = entropy.detach().cpu()
                else:
                    stats["code_entropy"] = torch.tensor(0.0)

                # attach to vq_outputs so the caller can log them
                vq_outputs["stats"] = stats

        return {
            "z": z,
            "vq_outputs": vq_outputs,
            "latent_actions": z_a_down_shifted.squeeze(2),          # z_a_down (as before, but explicit)
            "quantized_latent_actions": quantized_latent_actions,   # z_q
            "visual_embs": z_dct["visual"],
        }




    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        encode_output = self.encode(obs, act)  
        z = encode_output["z"]  # (b, num_frames, num_patches, emb_dim)
        vq_output = encode_output["vq_outputs"]
        loss_components["vq_loss"] = vq_output["loss"]

        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components, encode_output

    def replace_actions_from_z(self, z, act):
        if self.concat_dim == 0:
            z[:, :, -1, :] = act
        elif self.concat_dim == 1:
            act_tiled = repeat(act.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z



    def rollout(self, obs, act, num_obs_init):
        """
        input:
            obs  (dict): (b, t+n, 3, H, W)  # FULL trajectory
            act  (tensor): ignored (API compatibility)
            num_obs_init (int): how many initial frames act as context

        output:
            z_obses, z
        """
        encode_output = self.encode(obs, act)
        quantized_latent_actions = encode_output["quantized_latent_actions"]

        z = encode_output["z"][:, :num_obs_init]
        act = self.encode_act(act)
        action = act[:, num_obs_init:]

        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist:])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t:t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist:])
        z_new = z_pred[:, -1:, ...]
        z = torch.cat([z, z_new], dim=1)

        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

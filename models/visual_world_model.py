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
        decoder,
        predictor,
        codebook_splits,
        codebook_dim,
        action_dim,
        concat_dim,
        latent_action_dim,
        num_action_repeat,
        train_encoder,
        train_predictor,
        train_decoder,
        train_lam,
        action_encoder=None,
        latent_action_model=None,
        vq_model=None,
        latent_action_down=None,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.latent_action_model = latent_action_model
        self.vq_model = vq_model
        self.latent_action_down = latent_action_down
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim ) * (concat_dim) # Not used
        self.latent_action_dim = latent_action_dim
        self.encoder_emb_dim = self.encoder.emb_dim
        self.codebook_splits = codebook_splits
        self.codebook_dim = codebook_dim

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"action encoder: {action_encoder}")
        print(f"latent_action_model: {self.latent_action_model}")
        print(f"latent_vq_model: {self.vq_model}")
        print(f"latent_action_down: {self.latent_action_down}")
        print(f"latent_action_dim: {self.latent_action_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"encoder_emb_dim: {self.encoder_emb_dim}")


        self.concat_dim = concat_dim  # 0 or 1
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
        self.use_z_q_in_concat = True  # whether to use z_q in concat mode (dim=1) or z_a_down

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
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

        vq_outputs = self.vq_model(z_a_down_shifted)
        quantized_latent_actions = vq_outputs["z_q_st"].squeeze(2)                  # z_q: (B, T, A)
        vq_outputs["indices"] = vq_outputs["indices"].squeeze(2)                    # (B, T)

        # ----- concat modes -----
        if self.concat_dim == 0:
            # concat along token-dim; use quantized actions as before
            act_token = quantized_latent_actions.unsqueeze(2)                       # (B, T, 1, A)
            z = torch.cat([z_dct["visual"], act_token], dim=2)

        elif self.concat_dim == 1:
            # concat along feature-dim; gate between z_a_down and z_q per flag
            

            # decide which latent action representation to use
            use_z_q = getattr(self, "use_z_q_in_concat", False)
            if use_z_q:
                act_base = quantized_latent_actions                                  # (B, T, A)
            else:
                act_base = z_a_down_shifted.squeeze(2)                               # (B, T, A) from z_a_down

            act_tiled = repeat(
                act_base.unsqueeze(2), "b t 1 a -> b t f a",
                f=z_dct["visual"].shape[2]
            )
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)

            z = torch.cat([z_dct["visual"], act_repeated], dim=3)
        else:
            raise ValueError(f"Unsupported concat_dim: {self.concat_dim}")

        # ----- stats every X steps -----
        # hardcoded interval, using a self.counter as requested
        LOG_INTERVAL = 50
        if not hasattr(self, "counter"):
            self.counter = 0
        self.counter += 1

        if self.counter % LOG_INTERVAL == 0:
            with torch.no_grad():
                stats = {}

                def add_tensor_stats(name, t):
                    # Flatten everything except last dim
                    t_flat = t.reshape(-1, t.shape[-1])
                    stats[f"{name}_mean"] = float(t_flat.mean().item())
                    stats[f"{name}_std"] = float(t_flat.std().item())

                    dim_zero_frac = (t_flat == 0).float().mean(dim=0)  # (D,)
                    # store full vector for later programmatic logging, but don't print full vector
                    stats[f"{name}_dim_zero_frac"] = dim_zero_frac.detach().cpu()

                    # compact scalar summaries for printing
                    stats[f"{name}_dim_zero_frac_mean"] = float(dim_zero_frac.mean().item())
                    stats[f"{name}_dim_zero_frac_max"] = float(dim_zero_frac.max().item())

                # z_a: pre-downsample, z_a_down: after downsample+shift, z_q: quantized
                add_tensor_stats("z_a", z_a)
                add_tensor_stats("z_a_down", z_a_down_shifted.squeeze(2))
                add_tensor_stats("z_q", quantized_latent_actions)

                # VQ loss (if present)
                if isinstance(vq_outputs, dict) and ("loss" in vq_outputs) and (vq_outputs["loss"] is not None):
                    vq_loss_val = vq_outputs["loss"]
                    stats["vq_loss"] = float(vq_loss_val.detach().float().mean().cpu().item())

                # code usage statistics from indices
                indices = vq_outputs.get("indices", None) if isinstance(vq_outputs, dict) else None
                if indices is not None:
                    idx_flat = indices.reshape(-1).detach()

                    # infer codebook size if not explicitly stored
                    codebook_size = getattr(
                        self.vq_model,
                        "num_embeddings",
                        int(idx_flat.max().item()) + 1 if idx_flat.numel() > 0 else 0,
                    )

                    if codebook_size > 0 and idx_flat.numel() > 0:
                        counts = torch.bincount(idx_flat, minlength=codebook_size)  # (num_codes,)
                        stats["code_counts"] = counts.detach().cpu()

                        used = (counts > 0).sum().item()
                        stats["num_used_codes"] = int(used)

                        total = counts.sum().item()
                        if total > 0:
                            probs = counts.float() / counts.sum()
                            nonzero = probs > 0
                            entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
                            stats["code_entropy"] = float(entropy.detach().cpu().item())
                        else:
                            stats["code_entropy"] = 0.0

                        # compact summaries for printing
                        stats["code_topk_counts"] = (
                            torch.topk(counts, k=min(5, counts.numel())).values.detach().cpu().tolist()
                        )
                    else:
                        stats["num_used_codes"] = 0
                        stats["code_entropy"] = 0.0
                        stats["code_topk_counts"] = []
                else:
                    stats["num_used_codes"] = 0
                    stats["code_entropy"] = 0.0
                    stats["code_topk_counts"] = []

                # attach to vq_outputs so caller can log them
                vq_outputs["stats"] = stats

                # ---- SAFE PRINT (rank 0 only) ----
                is_rank0 = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    is_rank0 = (torch.distributed.get_rank() == 0)

                if is_rank0:
                    msg = (
                        f"[encode stats @ {self.counter}] "
                        f"z_a(mean={stats['z_a_mean']:.4f}, std={stats['z_a_std']:.4f}, "
                        f"zero_frac_mean={stats['z_a_dim_zero_frac_mean']:.4f}, zero_frac_max={stats['z_a_dim_zero_frac_max']:.4f}) | "
                        f"z_a_down(mean={stats['z_a_down_mean']:.4f}, std={stats['z_a_down_std']:.4f}, "
                        f"zero_frac_mean={stats['z_a_down_dim_zero_frac_mean']:.4f}, zero_frac_max={stats['z_a_down_dim_zero_frac_max']:.4f}) | "
                        f"z_q(mean={stats['z_q_mean']:.4f}, std={stats['z_q_std']:.4f}, "
                        f"zero_frac_mean={stats['z_q_dim_zero_frac_mean']:.4f}, zero_frac_max={stats['z_q_dim_zero_frac_max']:.4f}) | "
                        f"vq_loss={stats.get('vq_loss', float('nan')):.4f} | "
                        f"used_codes={stats['num_used_codes']} entropy={stats['code_entropy']:.4f} "
                        f"top5_counts={stats.get('code_topk_counts', [])}"
                    )
                    print(msg, flush=True)


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
        return {"visual": visual_embs}

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
        obs = {"visual": visual}
        return obs, diff

    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
                z_visual, z_act = z[:, :, :-1, :], z[:, :, -1, :]
                z_obs = {"visual": z_visual}
                return z_obs, z_act

        elif self.concat_dim == 1:
            z_visual = z[..., :-self.action_dim]
            z_act = z[..., -self.action_dim:]
            # remove tiled dims
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
            z_obs = {"visual": z_visual}
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
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
                z_loss = z_visual_loss

            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim],
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )
                z_loss = z_visual_loss


            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
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
        #latent_actions = encode_output["latent_actions"][:, :-1]  # (b, t+n-1, latent_action_dim)
        quantized_latent_actions = encode_output["quantized_latent_actions"][:, :-1]  # (b, t+n-1, latent_action_dim)

        z = encode_output["z"][:, :num_obs_init]
        act = quantized_latent_actions #self.encode_act(act)
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

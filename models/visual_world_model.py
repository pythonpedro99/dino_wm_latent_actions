import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from typing import Optional

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
        use_action_encoder,
        use_lam,
        use_vq,
        plan_action_type: Optional[str] = None,
        action_encoder=None,
        latent_action_model=None,
        vq_model=None,
        latent_action_down=None
    ):
        super().__init__()

        self.plan_action_type = plan_action_type
        self.use_action_encoder = use_action_encoder
        self.use_lam = use_lam
        self.use_vq = use_vq

        self.encoder = encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.latent_action_model = latent_action_model
        self.vq_model = vq_model
        self.latent_action_down = latent_action_down
        self.concat_dim = concat_dim 

        self.train_lam = train_lam
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder

        if self.use_action_encoder == self.use_lam:
            raise ValueError("Invalid config: choose exactly one: use_action_encoder XOR use_lam")

        if self.use_action_encoder and self.action_encoder is None:
            raise ValueError("use_action_encoder=True but action_encoder is None")

        if self.use_lam:
            if self.latent_action_model is None or self.latent_action_down is None:
                raise ValueError("use_lam=True but latent_action_model or latent_action_down is None")
            if self.use_vq and self.vq_model is None:
                raise ValueError("use_vq=True but vq_model is None")

        self.num_hist = num_hist
        self.num_pred = num_pred
        self.num_action_repeat = num_action_repeat 
        self.encoder_emb_dim = self.encoder.emb_dim
        self.codebook_splits = codebook_splits
        self.codebook_dim = codebook_dim
        self.num_action_repeat = num_action_repeat
        self.latent_action_dim = latent_action_dim

        # Canonical per-step action feature dim (before repeat)
        if self.use_action_encoder:
            # Prefer the actual module attribute if present
            self.act_feat_dim = getattr(self.action_encoder, "emb_dim", action_dim)
        else:
            if self.use_vq:
                # VQ quantizes vectors of size codebook_splits * code_dim
                self.act_feat_dim = self.codebook_splits * self.codebook_dim
            else:
                self.act_feat_dim = latent_action_dim

        # This is the dimension that appears in z when concat_dim==1
        self.action_dim = self.act_feat_dim * self.num_action_repeat
        self.emb_dim = self.encoder.emb_dim + (self.action_dim if self.concat_dim == 1 else 0)



        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"action encoder: {action_encoder}")
        print(f"latent_action_model: {self.latent_action_model}")
        print(f"latent_vq_model: {self.vq_model}")
        print(f"latent_action_down: {self.latent_action_down}")
        print(f"latent_action_dim: {self.latent_action_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"encoder_emb_dim: {self.encoder_emb_dim}")


        
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
        if self.action_encoder is not None:
            self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        if self.action_encoder is not None:
            self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act):
        z_dct = self.encode_obs(obs)
        z_visual = z_dct["visual"]  # (B, T, P, D)

        vq_outputs = None
        latent_actions = None
        quantized_latent_actions = None

        if self.plan_action_type == "raw":
            if self.use_action_encoder:
                act_base = self.encode_act(act)  # (B, T, A_feat)
            elif self.use_lam:
                z_a = self.latent_action_model(z_visual)["action_patches"]
                z_a_down = self.latent_action_down(z_a)
                z_a_down_shifted = torch.cat([z_a_down[:, 1:], z_a_down[:, -1:]], dim=1)
                latent_actions = z_a_down_shifted.squeeze(2)

                if self.use_vq:
                    vq_outputs = self.vq_model(z_a_down_shifted)
                    quantized_latent_actions = vq_outputs["z_q_st"].squeeze(2)
                    vq_outputs["indices"] = vq_outputs["indices"].squeeze(2)
                    act_base = quantized_latent_actions
                else:
                    act_base = latent_actions
            else:
                raise RuntimeError("plan_action_type='raw' but neither action encoder nor LAM is enabled.")

        elif self.plan_action_type in ("latent", "discrete"):
            # Explicit: use provided action vectors as-is; do NOT encode and do NOT infer from obs
            act_base = act

        else:
            raise ValueError(f"Unsupported plan_action_type: {self.plan_action_type}")

        if act_base.shape[-1] != self.act_feat_dim:
            raise ValueError(
                f"Action dim mismatch: got {act_base.shape[-1]}, expected {self.act_feat_dim}. "
                f"plan_action_type={self.plan_action_type}. "
                "If using discrete/latent, pass codebook vectors in the expected feature dim "
                "(or add a projection)."
            )

        if self.concat_dim == 0:
            z = torch.cat([z_visual, act_base.unsqueeze(2)], dim=2)
        elif self.concat_dim == 1:
            act_tiled = repeat(act_base.unsqueeze(2), "b t 1 a -> b t p a", p=z_visual.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat([z_visual, act_repeated], dim=3)
        else:
            raise ValueError(f"Unsupported concat_dim: {self.concat_dim}")

        return {
            "z": z,
            "vq_outputs": vq_outputs,
            "latent_actions": latent_actions,
            "quantized_latent_actions": quantized_latent_actions,
            "visual_embs": z_visual,
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
        assert self.plan_action_type == "raw", f"forward() expects plan_action_type='raw' for training, got: {self.plan_action_type}"
        encode_output = self.encode(obs, act)  
        z = encode_output["z"]  # (b, num_frames, num_patches, emb_dim)
        vq_output = encode_output["vq_outputs"]
        if vq_output is not None and ("loss" in vq_output) and (vq_output["loss"] is not None):
            loss_components["idm_vq_loss"] = vq_output["loss"]
            loss = loss + loss_components["idm_vq_loss"]

        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
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

            # Predict visual latents only (actions are conditioning; do not include them in the target loss)
            if self.concat_dim == 0:
                # action is appended as an extra token (last token)
                pred_visual_latents = z_pred[:, :, :-1, :]               # (B, H, P, D)
                tgt_visual_latents  = z_tgt[:, :, :-1, :].detach()

            elif self.concat_dim == 1:
                # action is appended to feature dim (last self.action_dim channels)
                pred_visual_latents = z_pred[:, :, :, :-self.action_dim] # (B, H, P, D_vis)
                tgt_visual_latents  = z_tgt[:, :, :, :-self.action_dim].detach()

            else:
                raise ValueError(f"Unsupported concat_dim: {self.concat_dim}")

            visual_latent_pred_loss = self.emb_criterion(pred_visual_latents, tgt_visual_latents)

            loss = loss + visual_latent_pred_loss
            loss_components["visual_latent_pred_loss"] = visual_latent_pred_loss
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
        encode_output = self.encode(obs, act)

        # initial context
        z = encode_output["z"][:, :num_obs_init]

        # ------------------------------------------------------------
        # Select the action stream to condition rollout on (explicit)
        # ------------------------------------------------------------
        if self.plan_action_type == "raw":
            if self.use_action_encoder:
                actions_all = self.encode_act(act)  # (B,T,Afeat)
            elif self.use_lam:
                # In raw mode you allow inference (current behavior)
                if self.use_vq:
                    actions_all = encode_output["quantized_latent_actions"]
                    if actions_all is None:
                        raise RuntimeError("plan_action_type='raw': expected quantized_latent_actions but got None.")
                else:
                    actions_all = encode_output["latent_actions"]
                    if actions_all is None:
                        raise RuntimeError("plan_action_type='raw': expected latent_actions but got None.")
            else:
                raise RuntimeError("plan_action_type='raw' but neither use_action_encoder nor use_lam is enabled.")

        elif self.plan_action_type in ("latent", "discrete"):
            # Planner provides already-embedded action vectors
            actions_all = act  # (B,T,Afeat)

        else:
            raise ValueError(f"Unsupported plan_action_type: {self.plan_action_type}")

        # Validate dim early (prevents silent shape bugs)
        if actions_all.shape[-1] != self.act_feat_dim:
            raise ValueError(
                f"Action dim mismatch in rollout: got {actions_all.shape[-1]}, expected {self.act_feat_dim}. "
                f"plan_action_type={self.plan_action_type}"
            )

        # transitions for T frames
        actions_all = actions_all[:, :-1, :]
        action = actions_all[:, num_obs_init:, :]

        # ------------------------------------------------------------
        # Iterative rollout
        # ------------------------------------------------------------
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

        z_obses, _ = self.separate_emb(z)
        return z_obses, z



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pymotion.rotations.ortho6d_torch as ortho6d
from utils import network_timing
from networks.unified.components import (
    DisplacementPredictor,
    LatentGRU,
    RotsFromWaypoints,
    TagEncoder,
    TimingEncoder,
    resolve_vae_latent_dim,
)
from networks.unified.decoder import Decoder
from networks.unified.encoder import Encoder
from networks.unified.prior import PRIOR_EVAL_MODES, RecurrentPrior
from networks.unified.foot_contact import (
    apply_foot_keep_mask,
    foot_contact_channels,
    foot_count,
    sample_foot_keep_mask,
)


class Autoencoder(nn.Module):
    def __init__(self, param, parents, device, transform_net = False, is_vae = False, is_vq_vae = False, *, variant):
        super(Autoencoder, self).__init__()
        self.variant = variant
        self.is_vae = is_vae
        self.is_vq_vae = is_vq_vae
        self.vae_latent_dim = resolve_vae_latent_dim(param)
        self.encoder = Encoder(param, parents, device, is_vae = self.is_vae, is_vq_vae=self.is_vq_vae)
        self.decoder = Decoder(param, self.encoder, device, variant=variant)
        self.parents = parents
        self.timing_enc = TimingEncoder(input_dim=48,latent_dim=32,output_dim=5, hidden_dim=32)
        self.tag_enc = TagEncoder(input_dim=7, output_dim=64, hidden_dim=128 ,kernel_size=9)
        # keep historical attribute name for checkpoint compatibility
        self.latent_lstm = LatentGRU(tag_embed_dim=64, z_dim=5, hidden_dim=256, output_dim=self.vae_latent_dim)
        self.rots_waypoints = RotsFromWaypoints(waypoints_dim=64)
        
        # foot-contact conditioning (opt-in): the prior's velocity and acceleration controls become
        # per-foot contact and validity channels. the displacement predictor keeps every control.
        self.foot_contact = bool(param.get("foot_contact_conditioning", False))
        self.foot_contact_mask_prob = float(param.get("foot_contact_mask_probability", 0.25))
        self.foot_contact_feet = foot_count(param["feet_idxs"]) if self.foot_contact else 0
        replaced = ("ctrl_velocity", "ctrl_acceleration") if self.foot_contact else ()
        self._prior_base_columns = [i for i, name in enumerate(variant.control_tags) if name not in replaced]
        prior_ctrl_dim = len(self._prior_base_columns) + 2 * self.foot_contact_feet
        self.codebook_predictor2 = RecurrentPrior(ctrl_dim=prior_ctrl_dim, num_codebook_vectors=param["codebook_size"], num_levels=1, window_size=param["gru_window"],
                                                  num_layers=param["gru_layers"], hidden_dim=param["gru_hidden_dim"], vq_dim=param["gru_vq_dim"],
                                                  use_hidden_state=variant.hidden_state_enabled(param), variant=variant,
                                                  passthrough_channels=2 * self.foot_contact_feet,
                                                  scale_velocity_channels=not self.foot_contact,
                                                  # stage overrides from param; None keeps the variant's value
                                                  dropout=param.get("prior_dropout"),
                                                  teacher_forcing_prob=param.get("prior_teacher_forcing_prob"),
                                                  ctrl_channel_dropout=param.get("prior_ctrl_channel_dropout"),
                                                  gru_dropout=param.get("prior_gru_dropout"))
        self.disp_pred = DisplacementPredictor(ctrl_dim=len(variant.control_tags),embed_dim=self.vae_latent_dim)
        self.enc_output = None
        self.lstm_output = None
        self.param = param
        self.training_stage = str(param.get("training_stage", "prior")).lower()
        self.decoder_source = str(param.get("decoder_source", "stage")).lower()
        # runtime-only (not a parameter or buffer, so it is not in state_dict): True during the continuous AE warmup
        self.continuous_warmup = False

    def set_continuous_warmup(self, enabled):
        # warmup reconstructs from the encoder output, drops the VQ commitment term and freezes the EMA codebook
        self.continuous_warmup = bool(enabled)
        self.encoder.ema_enabled = not self.continuous_warmup

    def _get_training_stage(self):
        return str(self.param.get("training_stage", self.training_stage)).lower()

    def _is_autoencoder_stage(self):
        return self._get_training_stage() in {"autoencoder", "encoder", "vqvae", "vq_vae"}

    def _get_decoder_source(self):
        if self.continuous_warmup:
            return "encoder_output"
        decoder_source = str(self.param.get("decoder_source", self.decoder_source)).lower()
        if self.training:
            return "quantized" if self._is_autoencoder_stage() else "predicted"

        if decoder_source == "stage":
            return "quantized" if self._is_autoencoder_stage() else "predicted"

        if decoder_source in {"encoder", "encoder_output", "raw_encoder_output"}:
            return "encoder_output"
        if decoder_source == "autoencoder":
            return "quantized"
        if decoder_source in {"predicted", "predicted_codebook", "prior", "rnn"}:
            return "predicted"
        if decoder_source == "quantized":
            return "quantized"

        return "quantized" if self._is_autoencoder_stage() else "predicted"

    def _get_inference_temperature(self):
        return float(self.param.get("inference_temperature", 0.0))

    def _get_inference_top_k(self):
        top_k = self.param.get("inference_top_k", None)
        if top_k is None:
            return None
        top_k = int(top_k)
        return top_k if top_k > 0 else None
        
    @staticmethod
    def _control_tag(tags, name, multiplier):
        # ArchitectureSettings.control_multipliers: None keeps the tensor untouched; a number multiplies it.
        return tags[name] if multiplier is None else tags[name] * multiplier

    @staticmethod
    def prepare_encoder_input(input, tags):
        """Encoder input built from a sparse motion [batch, channels, frames]. Writes into input and returns it.

        The root channels take the control proxies (ctrl_velocity, ctrl_height, ctrl_yaw_rate), interpolated
        to the frame count, and the final four channels are zeroed. forward() and load_latents.py share this.
        """
        def tag_to_channels(tag):
            if tag.dim() == 2:
                tag = tag.unsqueeze(-1)
            tag = tag.permute(0, 2, 1).contiguous()
            if tag.size(-1) != input.size(-1):
                tag = F.interpolate(tag, size=input.size(-1), mode="linear", align_corners=False)
            return tag

        root_proxy = torch.zeros_like(input[:, :9])
        root_proxy[:, 0:1] = tag_to_channels(tags["ctrl_velocity"])
        root_proxy[:, 1:2] = tag_to_channels(tags["ctrl_height"])
        root_proxy[:, 2:3] = tag_to_channels(tags["ctrl_yaw_rate"])
        input[:, :9] = root_proxy
        input[:,-4:] = input[:,-4:]*0
        return input

    def get_latents(self):
        return self.enc_output,self.lstm_output

    def _prior_controls(self, tags, ctrl_vector):
        # prior input with foot-contact channels in place of velocity and acceleration.
        base = ctrl_vector[..., self._prior_base_columns]
        # the batch may add a trailing channel dimension to tags; foot contact is [batch, frames, feet]
        shape = (ctrl_vector.shape[0], ctrl_vector.shape[1], self.foot_contact_feet)
        contact = tags["foot_contact"].reshape(shape)
        valid = tags["foot_contact_valid"].reshape(shape)
        if self.training:
            # random masking: each foot is hidden for a whole training sequence with probability p
            keep = sample_foot_keep_mask(contact.shape[0], self.foot_contact_feet, self.foot_contact_mask_prob, contact.device)
            contact, valid = apply_foot_keep_mask(contact, valid, keep)
        return torch.cat([base, foot_contact_channels(contact, valid)], dim=-1)
    
    def forward(self, input, offset, mean_dqs, std_dqs, denorm_offsets, mean_root = None, std_root = None, mean_sin_cos=None, std_sin_cos=None,
                phase = None, phase_per_8_frames = None, velocity_per_8_frames = None, tags = None, real_dqs=None,
                prior_eval_mode="rollout"):
        noise_scale = 0.0
        # Stage 1 training reconstructs from the quantized codes, so the prior, timing, tag, latent and displacement
        # networks are skipped there. Evaluation and prior training keep the full forward.
        skip_unused = self.training and self._is_autoencoder_stage()

        dx_noisy = tags["dx"] + noise_scale * torch.randn_like(tags["dx"])
        dz_noisy = tags["dz"] + noise_scale * torch.randn_like(tags["dz"])
        velo_noisy = tags["velo"] + noise_scale * torch.randn_like(tags["velo"])

        
        recon_feet = None
        if skip_unused:
            tags_vector = ctrl_vector = yaw_vector = to_timing_enc = None
        else:
            tags_vector = torch.cat([velo_noisy,#[dx_noisy,dz_noisy,
                                     #velo_noisy,
                                    tags["head_height"],
                                    tags["height_full"]], dim=-1) 

            waypoints_vector = torch.cat([tags["amps"],
                                          tags["sin"],
                                          tags["cos"]],dim=-1)

            ctrl_vector = torch.cat(
                                    [self._control_tag(tags, name, multiplier)
                                     for name, multiplier in zip(self.variant.control_tags, self.variant.control_multipliers)],
                                    dim=-1)
        
            yaw_vector = tags["yaw_quats"].squeeze(-1).clone()
            tags_vector = torch.cat([yaw_vector,tags_vector],dim=-1)
            to_timing_enc = input[:,8:56,:].permute(0,2,1).clone()

        self.prepare_encoder_input(input, tags)

        quantized, encoding_indices, vq_loss, encoding_indices_all = self.encoder(input.clone())
        self.quantized_latent = quantized
        
       
        if (
            self.training
            and self.encoder.is_vq_vae
            and hasattr(self.encoder, "ema_cluster_size")
            and self._is_autoencoder_stage()
            and self.encoder.ema_enabled
        ):
            try:
                idxs = encoding_indices_all
                if idxs.dim() == 3:
                    # Update each level separately
                    for lvl in range(1):#range(idxs.size(2)):
                        idxs_lvl = idxs[..., lvl]  # [B,T]
                        
                        # For level 0, use encoder.grouped_latent
                        # For level 1, encoder.residual will be used inside ema_update
                        if hasattr(self.encoder, "grouped_latent") and int(self.encoder.ema_updates.item()) > 0:
                            self.encoder.ema_update(idxs_lvl, self.encoder.grouped_latent, lvl=lvl)
                            
                    # Reset unused vectors (only needed for level 0 typically)
                    if int(self.encoder.ema_updates.item()) % self.param["ema_updates"] == 0 and int(self.encoder.ema_updates.item()) > 0:
                        
                        for lvl in range(1):#range(idxs.size(2)):
                            self.encoder.reset_unused_codebook_vectors(
                                min_usage_frac=1e-5,
                                sample_z=self.encoder.grouped_latent if lvl == 0 else self.encoder.residual,
                                lvl=lvl
                            )
                else:
                    # Backward compatibility for single level
                    if int(self.encoder.ema_updates.item()) > 0:
                        self.encoder.ema_update(idxs, self.encoder.grouped_latent, lvl=0)
                    if int(self.encoder.ema_updates.item()) % self.param["ema_updates"] == 0 and int(self.encoder.ema_updates.item()) > 0:
                        self.encoder.reset_unused_codebook_vectors(
                            min_usage_frac=1e-5,
                            sample_z=self.encoder.grouped_latent,
                            lvl=0
                        )
            except Exception as e:
                print(f"EMA update error: {e}")
            
        
        if skip_unused:
            mu_timing = logvar_timing = displ = None
        else:
            z,mu_timing,logvar_timing = self.timing_enc(to_timing_enc.clone())
            encoded_tags = self.tag_enc(tags_vector.clone())

            batch, seq_len, _ = encoded_tags.shape
            seq_len = seq_len // self.encoder.group_size
            if(self.training):
                encoding_indices = encoding_indices.view(batch, seq_len, -1 )
            else:
                encoding_indices = encoding_indices.unsqueeze(0)
            
            
            latent, displ, rots = self.latent_lstm(encoded_tags.clone(), z.clone())
        
        
        if prior_eval_mode not in PRIOR_EVAL_MODES:
            raise ValueError(f"prior_eval_mode must be one of {PRIOR_EVAL_MODES}, got {prior_eval_mode!r}")
        if skip_unused:
            predicted_indices = primary_logits = embedding_logits = predicted_quantized = None
        else:
            placeholder_logits = torch.zeros(batch, seq_len, encoding_indices_all.shape[2], self.encoder.num_embeddings, device='cuda')
            predicted_indices = encoding_indices_all
            primary_logits = placeholder_logits
            embedding_logits = placeholder_logits
            codebooks = None # self.encoder.vq_codebooks
            yaw_sin_cos = torch.cat([tags["yaw_sin"],tags["yaw_cos"]],dim=-1)
        
            prior_ctrl_vector = self._prior_controls(tags, ctrl_vector) if self.foot_contact else ctrl_vector
            with network_timing.section("prior"):
                if prior_eval_mode == "teacher_forced":
                    # evaluation only: same targets and control inputs as training, ground-truth code feedback
                    predicted_indices, primary_logits = self.codebook_predictor2.forward_teacher_forced(
                        prior_ctrl_vector, encoding_indices_all, codebooks=codebooks, yaw_sin_cos=yaw_sin_cos)
                elif(self.training):
                    predicted_indices, primary_logits = self.codebook_predictor2(prior_ctrl_vector, target_indices=encoding_indices_all,
                                                                                       codebooks = codebooks, yaw_sin_cos=yaw_sin_cos)
                else:
                    predicted_indices, primary_logits = self.codebook_predictor2.sample(
                        prior_ctrl_vector,
                        temperature=self._get_inference_temperature(),
                        top_k=self._get_inference_top_k(),
                        codebooks=codebooks,
                        yaw_sin_cos=yaw_sin_cos,
                    )
            if predicted_indices is not None and predicted_indices.dim() >= 2:
                network_timing.record_codebook_steps(predicted_indices.shape[1])
        
            with network_timing.section("codebook_lookup"):
                predicted_quantized = self.encoder.indices_to_vector_(predicted_indices)
        
        
        decoder_source = self._get_decoder_source()
        if decoder_source == "predicted":
            to_decoder = predicted_quantized.permute(0,2,1).clone()
        elif decoder_source == "encoder_output":
            to_decoder = self.encoder.enc_out
            vq_loss *= 0
        elif decoder_source == "quantized":
            to_decoder = quantized
        else:
            to_decoder = quantized
     
        if not skip_unused:
            with network_timing.section("displacement_predictor"):
                displ = self.disp_pred(ctrl_vector,to_decoder.permute(0,2,1).clone().detach(),yaw_vector)
        B = tags["yaw_sin"].shape[0]
        device = tags["yaw_sin"].device
        yaw_augment_offset = torch.rand(B, device=device) * (4.0 * math.pi) - 2.0 * math.pi

        if(mean_sin_cos is not None and self.training and self._is_autoencoder_stage()):
            mean_sin = mean_sin_cos[0]
            mean_cos = mean_sin_cos[1]
            std_sin = std_sin_cos[0]
            std_cos = std_sin_cos[1]    

            sin = (tags["yaw_sin"] * std_sin) + mean_sin
            cos = (tags["yaw_cos"] * std_cos) + mean_cos

            sin_yaw_offset = torch.sin(yaw_augment_offset).view(B,1,1)
            cos_yaw_offset = torch.cos(yaw_augment_offset).view(B,1,1)
            
            sin_rot = sin*cos_yaw_offset + cos*sin_yaw_offset
            cos_rot = cos*cos_yaw_offset - sin*sin_yaw_offset
            
            tags["yaw_sin"] = (sin_rot - mean_sin) / std_sin
            tags["yaw_cos"] = (cos_rot - mean_cos) / std_cos
        else:
            yaw_augment_offset *= 0
            
        forward_dim = torch.cat([tags[name] for name in self.variant.heading_tags], dim=-1)
        
        with network_timing.section("decoder"):
            output = self.decoder(
                        to_decoder, offset, mean_dqs, std_dqs,
                        denorm_offsets, self.parents, smooth_root_pos=tags["smooth_root_pos"], heading=forward_dim)
        network_timing.record_generated_frames(output.shape[-1])
    
        mu_enc = mu_timing
        logvar_enc = logvar_timing
        mu_tt = mu_timing
        logvar_tt = logvar_timing

        
        
        root = output[:,:9].clone().permute(0,2,1)
        
        root = root * std_root + mean_root
        root = ortho6d.normalize(root)
        root = (root - mean_root) / std_root 
        
        
        return (output, vq_loss, mu_enc, logvar_enc,
                None, None, mu_timing, logvar_timing, primary_logits, 
                encoding_indices_all, None, None, embedding_logits, self.encoder.grouped_latent,
                self.encoder.grouped_latent, displ, root, yaw_augment_offset)

import importlib
import torch
import torch.nn as nn
from architectures import ARCHITECTURE_MODULES, DEFAULT_ARCHITECTURE
from training.losses import GeneratorLoss
from pymotion.ops.skeleton import (
    from_root_dual_quat_torch,
    translation_each_joint_torch,
    compute_global_pos_torch,
)
from pymotion.rotations.dual_quat_torch import to_rotation_translation
import pymotion.rotations.ortho6d_torch as ortho6d
from motion.signals import calc_feet_energy_torch


def resolve_architecture(architecture=None):
    """Validated architecture identifier. None selects DEFAULT_ARCHITECTURE."""
    name = DEFAULT_ARCHITECTURE if architecture is None else architecture
    if not isinstance(name, str) or name not in ARCHITECTURE_MODULES:
        raise ValueError(
            f"unsupported architecture {name!r}; supported: {', '.join(sorted(ARCHITECTURE_MODULES))}"
        )
    return name


def architecture_classes(architecture=None):
    """(Autoencoder, StaticEncoder) for an architecture. Each module is imported on demand and
    nothing is swapped in this module's namespace, so several architectures can coexist."""
    module = importlib.import_module(ARCHITECTURE_MODULES[resolve_architecture(architecture)])
    return module.Autoencoder, module.StaticEncoder


class Generator_Model(nn.Module):
    def __init__(
        self, device, param, parents, train_data, is_vae=False, is_vq_vae=False, architecture=None
    ) -> None:
        super().__init__()

        self.device = device
        self.param = param
        self.parents = parents
        self.data = train_data
        self.is_vae = is_vae
        self.is_vq_vae = is_vq_vae
        self.training_stage = str(param.get("training_stage", "prior")).lower()
        self.architecture = resolve_architecture(architecture)

        autoencoder_cls, static_encoder_cls = architecture_classes(self.architecture)
        self.static_encoder = static_encoder_cls(param, parents, device).to(device)
        self.autoencoder = autoencoder_cls(
            param, parents, device, is_vae=self.is_vae, is_vq_vae=self.is_vq_vae
        ).to(device)
        expected_sparse_channels = (len(param["sparse_joints"]) + 1) * 9
        encoder_channels = self.autoencoder.encoder.channel_list[0]
        if encoder_channels != expected_sparse_channels:
            raise ValueError(
                "sparse_joints/channel mismatch: encoder expects {} channels from the pooled skeleton, "
                "but sparse_joints plus displacement provide {} channels. Update sparse_joints to match "
                "the encoder pooling survivors.".format(
                    encoder_channels,
                    expected_sparse_channels,
                )
            )

        optimizer_params = list(self.static_encoder.parameters()) + list(
            self.autoencoder.parameters()
        )
        self.optimizer_params = optimizer_params
        self.named_params = list(self.static_encoder.named_parameters()) + list(
            self.autoencoder.named_parameters()
        )

        parameters_enc = list(self.autoencoder.encoder.named_parameters())
        parameters_root = [
            param
            for name, param in parameters_enc
            if "fc_mu_root" in name or "fc_logvar_root" in name
        ]
        parameters_discr = [
            param for name, param in self.named_params if "discriminator" in name
        ]

        # Print number parameters
        dec_params = 0
        for parameter in optimizer_params:
            dec_params += parameter.numel()
        print("# parameters generator:", dec_params)

        param["warmup_steps"] = 1000

        # The codebook embeddings are owned by the EMA update (encoder.ema_update), not by gradients. Left in AdamW,
        # decoupled weight decay shrinks them on every step, so they are excluded from the optimizer.
        codebook_ids = {id(p) for p in self.autoencoder.encoder.vq_codebooks.parameters()}
        self.optimizer = torch.optim.AdamW(
            [p for p in self.optimizer_params if id(p) not in codebook_ids], param["learning_rate"]
        )

        # param["warmup_steps"] above is written but never read: no warmup schedule is implemented.


        self.loss = GeneratorLoss(param, parents, device).to(device)
        train_data.losses.append(self.loss)
        self._last_debug_total_loss = None

    def _get_training_stage(self):
        return str(self.param.get("training_stage", self.training_stage)).lower()

    def _is_autoencoder_stage(self):
        return self._get_training_stage() in {"autoencoder", "encoder", "vqvae", "vq_vae"}

    def _set_trainable_parameters(self):
        train_prior_only = not self._is_autoencoder_stage()
        for name, param in self.named_params:
            param.requires_grad = ("codebook_predictor2" in name) if train_prior_only else True

    def configure_prior_train_mode(self):
        # Prior stage: only codebook_predictor2 trains. Every other child runs in eval mode, so its dropout is off and
        # its buffers (e.g. the encoder's ema_updates counter) are not written. autoencoder itself stays in training
        # mode, because its forward reads that flag to pick teacher-forced prior training. train() recurses into all
        # children, so this must be called after every generator_model.train().
        if self._is_autoencoder_stage():
            return
        self.static_encoder.eval()
        for name, module in self.autoencoder.named_children():
            if name != "codebook_predictor2":
                module.eval()

    def _debug_scalar(self, value):
        if value is None:
            return "None"
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return "empty"
            value = value.detach()
            if value.numel() == 1:
                return "{:.6g}".format(float(value.item()))
            value = value.float()
            return "mean={:.6g} max={:.6g}".format(
                float(value.mean().item()),
                float(value.abs().max().item()),
            )
        return "{:.6g}".format(float(value))

    def _debug_tensor_stats(self, name, tensor):
        if tensor is None:
            print(f"[loss-debug] {name}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            print(f"[loss-debug] {name}: {type(tensor).__name__}")
            return
        with torch.no_grad():
            data = tensor.detach()
            finite = torch.isfinite(data)
            finite_count = int(finite.sum().item())
            total_count = data.numel()
            if total_count == 0:
                print(f"[loss-debug] {name}: shape={tuple(data.shape)} empty")
                return
            if finite_count == 0:
                print(f"[loss-debug] {name}: shape={tuple(data.shape)} no finite values")
                return
            finite_data = data[finite].float()
            print(
                "[loss-debug] {}: shape={} finite={}/{} mean={:.6g} std={:.6g} min={:.6g} max={:.6g} absmax={:.6g}".format(
                    name,
                    tuple(data.shape),
                    finite_count,
                    total_count,
                    float(finite_data.mean().item()),
                    float(finite_data.std(unbiased=False).item()),
                    float(finite_data.min().item()),
                    float(finite_data.max().item()),
                    float(finite_data.abs().max().item()),
                )
            )

    def _top_gradient_norms(self, limit=8):
        rows = []
        for name, param in self.named_params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            finite = torch.isfinite(grad)
            if finite.any():
                grad_norm = float(torch.linalg.vector_norm(grad[finite].float()).item())
                grad_absmax = float(grad[finite].abs().max().item())
            else:
                grad_norm = float("nan")
                grad_absmax = float("nan")
            rows.append((grad_norm, grad_absmax, name))
        rows.sort(key=lambda row: row[0] if row[0] == row[0] else float("inf"), reverse=True)
        return rows[:limit]

    def _should_debug_loss(self, total_loss):
        if not bool(self.param.get("debug_loss_spikes", True)):
            return False
        if not torch.isfinite(total_loss.detach()).all():
            return True
        loss_value = float(total_loss.detach().item())
        threshold = float(self.param.get("debug_loss_threshold", 1000.0))
        spike_factor = float(self.param.get("debug_loss_spike_factor", 150.0))
        previous = self._last_debug_total_loss
        if loss_value >= threshold:
            return True
        if previous is not None and previous > 0.0 and loss_value >= previous * spike_factor:
            return True
        return False

    def _print_loss_debug(self, debug_context, loss, vq_loss, logits_loss, total_loss, grad_norm=None):
        context = debug_context or {}
        epoch = context.get("epoch", "?")
        step = context.get("step", "?")
        num_steps = context.get("num_steps", "?")
        stage = self._get_training_stage()
        lr_values = [group.get("lr", None) for group in self.optimizer.param_groups]

        print(
            "[loss-debug] spike epoch={} step={}/{} stage={} lr={} total={} train_loss={} vq={} logits={} grad_norm_preclip={}".format(
                epoch,
                step,
                num_steps,
                stage,
                lr_values,
                self._debug_scalar(total_loss),
                self._debug_scalar(loss),
                self._debug_scalar(vq_loss),
                self._debug_scalar(logits_loss),
                self._debug_scalar(grad_norm),
            )
        )

        components = getattr(self.loss, "debug_last_vq_vae_losses", {})
        if components:
            component_text = ", ".join(
                f"{name}={self._debug_scalar(value)}"
                for name, value in components.items()
            )
            print(f"[loss-debug] components: {component_text}")

        self._debug_tensor_stats("res_decoder", getattr(self, "res_decoder", None))
        self._debug_tensor_stats("target_motion", getattr(self.data, "motion", None))
        self._debug_tensor_stats("sparse_motion", getattr(self.data, "sparse_motion", None))
        self._debug_tensor_stats("disp_8_pred", getattr(self, "disp_8", None))
        self._debug_tensor_stats("disp_8_target", getattr(self.data, "disp_8", None))
        self._debug_tensor_stats("output_logits", getattr(self, "output_logits", None))
        self._debug_tensor_stats("target_indices", getattr(self, "target_indices", None))
        self._debug_tensor_stats("pred_latent", getattr(self, "pred_latent", None))
        self._debug_tensor_stats("true_latent", getattr(self, "true_latent", None))
        self._debug_tensor_stats("mean_dqs", getattr(self.data, "mean_dqs", None))
        self._debug_tensor_stats("std_dqs", getattr(self.data, "std_dqs", None))
        self._debug_tensor_stats("mean_root_rots", getattr(self.data, "mean_root", None))
        self._debug_tensor_stats("std_root_rots", getattr(self.data, "std_root", None))

        if hasattr(self.autoencoder, "encoder") and hasattr(self.autoencoder.encoder, "vq_codebooks"):
            for idx, codebook in enumerate(self.autoencoder.encoder.vq_codebooks):
                self._debug_tensor_stats(f"vq_codebook[{idx}]", codebook.weight)

        top_grads = self._top_gradient_norms()
        if top_grads:
            print("[loss-debug] top gradient norms:")
            for grad_norm_value, grad_absmax, name in top_grads:
                print(
                    "[loss-debug]   {}: norm={:.6g} absmax={:.6g}".format(
                        name,
                        grad_norm_value,
                        grad_absmax,
                    )
                )

    def forward(self, prior_eval_mode="rollout"):
        # prior_eval_mode is evaluation-only ("rollout" or "teacher_forced"); it is read by the VQ branch

        # Execute Static Encoder to obtain the offsets
        # input offsets has shape (1, n_joints, 3)

        self.ae_offsets = self.static_encoder(self.data.offsets)
        # Execute Autoencoder to obtain the motion
        if self.is_vae:
            (
                self.latent,
                self.og_latent,
                self.res_decoder,
                self.mu_z,
                self.logvar_z,
                self.mu_s,
                self.logvar_s,
                self.mu_shat,
                self.logvar_shat,
                self.mu_root,
                self.logvar_root,
                self.mu_root_z,
                self.logvar_root_z,
                self.mu_root_shat,
                self.logvar_root_shat,
                self.s_real,
                self.s_root,
                self.output_root_only,
                self.before_attn,
            ) = self.autoencoder(
                self.data.sparse_motion,
                self.ae_offsets,
                self.data.mean_dqs,
                self.data.std_dqs,
                self.data.denorm_offsets,
                phase=self.data.phase,
            )
            return self.res_decoder

        elif self.is_vq_vae:
            (
                self.res_decoder,
                self.vq_dict,
                self.mu,
                self.logvar,
                self.mu_s,
                self.logvar_s,
                self.timing_mu,
                self.timing_logvar,
                self.output_logits,
                self.target_indices,
                self.recon_feet,
                self.predicted_vectors,
                self.embedding_logits,
                self.pred_latent,
                self.true_latent,
                self.disp_8,
                self.decoder_root,
                self.phi,
            ) = self.autoencoder(
                self.data.sparse_motion,
                self.ae_offsets,
                self.data.mean_dqs,
                self.data.std_dqs,
                self.data.denorm_offsets,
                mean_root=self.data.mean_root,
                std_root=self.data.std_root,
                mean_sin_cos=[self.data.mean_sin, self.data.mean_cos],
                std_sin_cos=[self.data.std_sin, self.data.std_cos],
                phase_per_8_frames=self.data.phase_per_8_frames,
                velocity_per_8_frames=self.data.velocity_per_8_frames,
                tags=self.data.tags,
                real_dqs=self.data.motion,
                prior_eval_mode=prior_eval_mode,
            )

            self.enc_output, self.lstm_output = self.autoencoder.get_latents()

            dqs = self.res_decoder.permute(0, 2, 1)
            batch, frames, dims = dqs.shape
            dqs = dqs.flatten(0, 1)
            dqs = dqs * self.data.std_dqs + self.data.mean_dqs

            dqs = dqs.reshape(
                batch, frames, dims // 9, 9
            )  # reshape to (batch, frames, n_joints, 8)
            dqs = ortho6d.to_dual_quat(dqs)
            rots, self.pos = to_rotation_translation(dqs)

            glob_pos, glob_rots = compute_global_pos_torch(
                rots,
                self.data.global_pos,
                self.data.denorm_offsets,
                self.parents,
                end_sites=self.data.end_sites,
                end_sites_parents=self.data.end_sites_parents,
            )

            diffs = torch.diff(glob_pos[..., self.param["feet_idxs"], :], dim=1)
            norms = torch.norm(diffs, dim=-1)


            self.foot_speed = norms  # .detach()


        else:

            (
                self.latent,
                self.res_decoder,
                self.recon_feet,
                self.predicted_vectors,
                self.target_vectors,
                self.discr_fake,
                self.discr_real,
                self.discr_fake_det,
                self.discr,
                self.fake_latent,
                self.real_latent,
            ) = self.autoencoder(
                self.data.sparse_motion,
                self.ae_offsets,
                self.data.mean_dqs,
                self.data.std_dqs,
                self.data.denorm_offsets,
                tags=self.data.tags,
            )

            if self.data.global_pos.ndimension() == 4:
                self.data.global_pos = self.data.global_pos.squeeze(1)
            global_pos = self.data.global_pos.permute(0, 2, 1)
            num_frames = min(self.res_decoder.shape[-1], global_pos.shape[-2])
            dqs = self.res_decoder[..., :num_frames].permute(0, 2, 1)

            dqs = dqs * self.data.std_dqs + self.data.mean_dqs
            dqs = dqs.reshape(dqs.shape[0], dqs.shape[1], -1, 8)

            self.pos, self.rots = from_root_dual_quat_torch(dqs, self.parents)
            self.rots, self.pos = to_rotation_translation(dqs)
            diff = torch.diff(self.pos[:, [8, 12, 15, 18]], dim=0)
            foot_speed = torch.norm(diff, dim=-1)


            global_pos = global_pos[..., :num_frames, :]

            pos_each_joint = translation_each_joint_torch(
                self.rots.clone(),
                global_pos.clone(),
                torch.tensor(self.parents).clone(),
                self.data.denorm_offsets[0].clone(),
            )  # modify to work across batch!

            # fix order to be x y z. from previous func call its z x y
            # pos_z = pos_each_joint[...,1:,0].clone()
            # pos_x = pos_each_joint[...,1:,1].clone()
            # pos_y = pos_each_joint[...,1:,2].clone()
            # pos_xyz = torch.cat([pos_x.unsqueeze(-1), pos_y.unsqueeze(-1), pos_z.unsqueeze(-1)], dim=-1)
            # pos_each_joint = torch.cat([pos_each_joint[..., :1, :], pos_xyz], dim=-2)

            pos_each_joint = torch.cat(
                (pos_each_joint[..., :1, :], pos_each_joint[..., 1:, [1, 2, 0]]), dim=-2
            )
            self.energy_feet_predicted = calc_feet_energy_torch(
                pos_each_joint, self.param["feet_idx"]
            )
            self.energy_feet_target = self.data.energy


        return self.res_decoder, self.disp_8, self.decoder_root

    def optimize_parameters(self):

        torch.autograd.set_detect_anomaly(True)
        loss, perceptual_loss, discr_loss, gen_loss = self.loss.forward_generator(
            self.res_decoder,
            self.data.motion,
            self.recon_feet,
            self.data.sparse_motion[:, 2 * 8 : 6 * 8, :],
            self.predicted_vectors,
            self.target_vectors,
            self.discr_fake,
            self.discr_real,
            self.discr_fake_det,
            self.discr,
            self.fake_latent,
            self.real_latent,
        )
        total_loss = loss + perceptual_loss

        self.optimizer.zero_grad()

        total_loss.backward()
        self.optimizer.step()

        self.optimizer_d.zero_grad()
        discr_loss.backward()
        self.optimizer_d.step()

        return gen_loss, perceptual_loss, discr_loss

    def optimize_parameters_vq_vae(self, debug_context=None):
        self.optimizer.zero_grad()
        self._set_trainable_parameters()

        loss, vq_loss, temporal_loss, logits_loss, logits_loss_copy, velo_loss, decoder_root_mse = self._compute_vq_vae_losses()
        total_loss = (
            loss + 1.0 * vq_loss + logits_loss * 0
        )  # not in total_loss: temporal_loss, logits_loss_copy (second copy of logits_loss), predicted_timing_loss

        if not self._is_autoencoder_stage():
            loss = logits_loss
            total_loss = loss  # + decoder_root_mse

        should_debug = self._should_debug_loss(total_loss)
        if debug_context is None:
            debug_context = {}

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer_params, max_norm=1.0)
        if should_debug:
            self._print_loss_debug(
                debug_context,
                loss,
                vq_loss,
                logits_loss,
                total_loss,
                grad_norm=grad_norm,
            )
        self.optimizer.step()
        if torch.isfinite(total_loss.detach()).all():
            self._last_debug_total_loss = float(total_loss.detach().item())

        return loss, vq_loss, velo_loss, decoder_root_mse  # temporal_loss and logits_loss_copy are not returned

    def _compute_vq_vae_losses(self):
        codebook_weights = None
        if self.param.get("soft_codebook_targets", False):
            codebook_weights = [
                codebook.weight for codebook in self.autoencoder.encoder.vq_codebooks
            ]
        return self.loss.forward_generator_vq_vae(
            input=self.res_decoder,
            target=self.data.motion,
            vq_dict=self.vq_dict,
            mu=self.mu,
            logvar=self.logvar,
            timing_mu=self.timing_mu,
            timing_logvar=self.timing_logvar,
            tag_mu=self.mu_s,  # tag-encoder placeholder, always None
            tag_logvar=self.logvar_s,  # tag-encoder placeholder, always None
            output_logits=self.output_logits,  # for transformer
            target_indices=self.target_indices,  # for transformer
            recon_feet=self.recon_feet,
            target_sparse_pose=self.data.sparse_motion[:, 2 * 8 : 6 * 8, :],  # sparse channels 16:48, not feet
            predicted_vectors=self.predicted_vectors,
            embedding_logits=self.embedding_logits,
            pred_latent=self.pred_latent,
            true_latent=self.true_latent,
            disp_8_target=self.data.disp_8,
            disp_8=self.disp_8,
            foot_speed=self.foot_speed,
            target_velo=self.data.tags["velo_foot"],
            means=self.data.mean_dqs,
            stds=self.data.std_dqs,
            yaw_target=self.data.rots,
            yaw_pred=self.decoder_root,
            enc_output=self.enc_output,
            lstm_output=self.lstm_output,
            loss_weights=self.data.loss_weights,
            mean_root=self.data.mean_root,
            std_root=self.data.std_root,
            phi=self.phi,
            codebook_weights=codebook_weights,
            soft_codebook_temperature=self.param.get("soft_codebook_temperature", 0.1),
            soft_codebook_alpha=self.param.get("soft_codebook_alpha", 0.5),
        )

    def evaluate_prior_cross_entropy(self):
        _, _, _, logits_loss, _, _, _ = self._compute_vq_vae_losses()
        return logits_loss

    def optimize_parameters_vae(self, batch_size, epoch, root_loss_only):

        (
            recon_loss,
            KLD_loss_z,
            KLD_loss_s,
            KLD_loss_root,
            KLD_loss_root_z,
            mu_s,
            var_s,
            mu_shat,
            var_shat,
            mu_z,
            var_z,
            mu_root,
            var_root,
            mse_root,
            latent_loss,
        ) = self.loss.forward_generator_vae(
            self.res_decoder,
            self.data.motion,
            self.mu_z,
            self.logvar_z,
            self.mu_s,
            self.logvar_s,
            self.mu_shat,
            self.logvar_shat,
            self.mu_root,
            self.logvar_root,
            self.mu_root_z,
            self.logvar_root_z,
            self.mu_root_shat,
            self.logvar_root_shat,
            self.s_real,
            self.s_root,
            self.latent,
            self.og_latent,
            self.output_root_only,
            batch_size,
        )
        beta = 1.5  # legacy VAE path (optimize_parameters_vae): KL weight; max(0.0,1.5-recon_loss) was an earlier gating idea

        root_loss = latent_loss  # KLD_loss_root + latent_loss #* beta #(KLD_loss_root + mse_root*10)

        self.optimizer.zero_grad()
        loss = (
            recon_loss + KLD_loss_z * beta + KLD_loss_s * beta
        )  # + KLD_loss_root * beta )# + KLD_loss_root * beta * 2)#*0.002 + (root_loss) # use this if latent is s_root

        # if(root_loss_only and False): # maybe check avg root loss from train_vae instead?

        loss.backward()
        self.optimizer.step()

        return (
            loss,
            KLD_loss_z,
            KLD_loss_s,
            root_loss * 7,
            mu_s,
            var_s,
            mu_shat,
            var_shat,
            mu_z,
            var_z,
        )

    def toggle_model_except_root_layers(self, enable=False):
        for name, param in self.named_params:
            if "lstm" in name:
                param.requires_grad = True  # (not enable)
            else:
                param.requires_grad = enable

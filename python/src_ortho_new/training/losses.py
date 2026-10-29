import torch
import torch.nn as nn
from pymotion.ops.forward_kinematics_torch import fk
from pymotion.ops.skeleton_torch import from_root_dual_quat
import torch.nn.functional as F
import pymotion.rotations.ortho6d_torch as ortho6d



class GeneratorLoss(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.param = param
        self.parents = parents
        self.device = device

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs.unsqueeze(-1)

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs.unsqueeze(-1)

    def set_offsets(self, offsets):
        self.joint_distances = torch.norm(offsets, dim=1)


    def soft_codebook_cross_entropy(
        self,
        logits,
        target_indices,
        codebook_weights,
        temperature=0.1,
    ):
        """
        Give partial credit to codes whose embeddings are close to the target code.
        logits [B, T, L, K]; target_indices [B, T, L].
        """
        B, T, L, K = logits.shape
        losses = []
        temperature = max(float(temperature), 1e-8)

        for level in range(L):
            weights = codebook_weights[min(level, len(codebook_weights) - 1)].detach()
            logits_l = logits[:, :, level, :].reshape(-1, K)
            targets_l = target_indices[:, :, level].reshape(-1)
            target_vectors = weights[targets_l]

            distances = (
                torch.sum(target_vectors * target_vectors, dim=1, keepdim=True)
                + torch.sum(weights * weights, dim=1).unsqueeze(0)
                - 2.0 * (target_vectors @ weights.t())
            )
            soft_targets = F.softmax(-distances / temperature, dim=-1)
            log_probs = F.log_softmax(logits_l, dim=-1)
            losses.append(F.kl_div(log_probs, soft_targets, reduction="batchmean"))

        return torch.stack(losses).mean()
    
    def wasserstein_discriminator_loss(self, discr_real, discr_fake):
        # No sigmoid is applied at the output of the critic.
        # Loss: difference between the fake score and the real score.
        return torch.mean(discr_fake) - torch.mean(discr_real)

    def wasserstein_generator_loss(self, discr_fake):
        # Generator tries to minimize negative score, i.e. maximize discr_fake.
        return -torch.mean(discr_fake)
    
    def gradient_penalty(self, discr, real, fake, device):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real.size(0), 1, 1).to(device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates = discr(interpolates)
        fake = torch.ones(real.size(0), 1).to(device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    

    def forward_generator(self, input, target, recon_feet, target_feet, predicted_vectors, target_vectors, discr_fake=None, discr_real= None, discr_fake_det=None, discr = None, fake_latent=None, real_latent=None):
        dqs = input.clone()
        target_dqs = target.clone()
        
        displacement_error = self.mse(dqs[:,4:7,:], target_dqs[:,4:7,:])
        # Dual Quaternions MSE Loss
        loss_joints = self.mse(dqs[:, 8:, :], target_dqs[:, 8:, :])
        loss_root = self.mse(dqs[:, :8, :], target_dqs[:, :8, :])




        if discr_fake is not None and discr_real is not None:
            discr_loss = self.wasserstein_discriminator_loss(discr_real, discr_fake)
            with torch.backends.cudnn.flags(enabled=False):
                gp = self.gradient_penalty(discr, real_latent, fake_latent, self.device)
            discr_loss = discr_loss + 10 * gp
            generator_loss = self.wasserstein_generator_loss(discr_fake_det.clone())
        else:
            discr_loss = loss_root * 0
            generator_loss = 0
        
        if(predicted_vectors is not None):
            predicted_vectors_loss = self.mse(predicted_vectors, target_vectors)
            return (loss_root * self.param["lambda_root"] + loss_joints ), predicted_vectors_loss , discr_loss , generator_loss
        
        
        if(recon_feet is not None):
            recon_feet_loss = self.mse(recon_feet, target_feet)
        else:
            recon_feet_loss = loss_root * 0     

        return (loss_root * self.param["lambda_root"] + loss_joints), displacement_error, discr_loss
    
    def forward_generator_vae(self, input, target, mu_z, logvar_z, mu_s, logvar_s, mu_shat, logvar_shat, mu_root, logvar_root, mu_root_z, logvar_root_z, mu_root_shat, logvar_root_shat, s_real, s_root, latent, og_latent, output_root_only, batch_size):
        dqs = input.clone()
        target_dqs = target.clone()

        #dqs_root = output_root_only.clone()
        # Dual Quaternions MSE Loss
        # = self.mse(dqs[:, 8:24, :], target_dqs[:, 8:24, :])
        loss_joints = self.mse(dqs[:, 8:, :], target_dqs[:, 8:, :])
        loss_root = self.mse(dqs[:, :8, :], target_dqs[:, :8, :])
        recon_loss = loss_root * self.param["lambda_root"]+ loss_joints #+ loss_feet * 5

        if(mu_z is not None):
            KLD_loss_s = 0.5 * torch.sum(logvar_shat - logvar_s + (torch.exp(logvar_s) + (mu_s - mu_shat)**2) / torch.exp(logvar_shat) - 1)
            temporal_loss = self.mse(s_real[:,1:],s_real[:,:-1])#1/torch.mean(torch.abs(s_real[:,1:] - s_real[:,:-1])) # CONTINUITY IN THE LATENT SPACE
            KLD_loss_z = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            
            KLD_loss_root_z = -0.5 * torch.sum(1 + logvar_root_z - mu_root_z.pow(2) - logvar_root_z.exp())
            KLD_loss_root = 0.5 * torch.sum(logvar_root - logvar_s + (torch.exp(logvar_s) + (mu_s - mu_root)**2) / torch.exp(logvar_root) - 1)
            KLD_loss_z /= (batch_size*512)
            KLD_loss_s /= (batch_size*512)
            KLD_loss_root_z /= (batch_size*512)
            KLD_loss_root /= (batch_size*512)
            loss_root_to_real = self.mse(s_real, s_root)

            latent_loss = KLD_loss_root#self.mse(latent, og_latent)
            return recon_loss, KLD_loss_z, KLD_loss_s, KLD_loss_root, KLD_loss_root_z, mu_s, torch.exp(logvar_s), mu_shat, torch.exp(logvar_shat), mu_z, torch.exp(logvar_z), mu_root, torch.exp(logvar_root), loss_root_to_real, latent_loss   
    
        return recon_loss
    
    def forward_generator_vq_vae(
                self, input, target, vq_dict, mu, logvar, timing_mu,
                timing_logvar, tag_mu, tag_logvar, output_logits, target_indices, recon_feet,
                target_sparse_pose, predicted_vectors, embedding_logits=None, pred_latent=None, true_latent=None, disp_8 = None, 
                disp_8_target = None, foot_speed=None, target_velo=None, means=None, stds=None,
                yaw_target=None, yaw_pred=None, enc_output = None, lstm_output = None,
                loss_weights=None, mean_root=None, std_root=None, phi=None,
                codebook_weights=None, soft_codebook_temperature=0.1,
                soft_codebook_alpha=0.5
                ):       
        


        feet_start_idxs = [(8-1)*9,(12-1)*9,(15-1)*9,(18-1)*9]
        feet_loss = 0.0
        rest_loss = 0.0
        num_joints = input.shape[1] // 9
        feet_indices = []
        for start_idx in feet_start_idxs:
            feet_indices.extend(list(range(start_idx, start_idx + 9)))

        feet_indices = torch.tensor(feet_indices, device=input.device)
        
        all_indices = torch.arange(input.shape[1], device=input.device)
        
        non_feet_mask = torch.ones(input.shape[1], dtype=torch.bool, device=input.device)
        non_feet_mask[feet_indices] = False
        rest_indices = all_indices[non_feet_mask]
        
        input_feet = input[:, feet_indices]
        target_feet = target[:, feet_indices]
        input_rest = input[:, rest_indices]
        target_rest = target[:, rest_indices]
        
        feet_loss = self.mse(input_feet, target_feet)
        rest_loss = self.mse(input_rest, target_rest)
        
        if (phi is not None):
            means = means.unsqueeze(-1)
            stds= stds.unsqueeze(-1)
            input = (input*stds) + means
            target = (target*stds) + means
            
            new_root_in, new_root_tgt = self.rotate_root_channels(input[:,:9], target[:,:9], phi)
            target = torch.cat([new_root_tgt, target[:, 9:, :].clone()], dim=1)
            
            input = (input-means) / stds
            target = (target-means) / stds
        
        rots_weight = 10 #10
        joints_weight = 1#1
        disp_weight = 1#100
        rots_loss = self.mse(input[:,:6], target[:,:6]) * rots_weight
        joints_loss = self.mse(input[:,9:], target[:,9:]) * joints_weight
        disp_loss = self.mse(input[:,6:9],target[:,6:9]) * disp_weight #.mean() * 10
        cumulative_root_loss = input.new_tensor(0.0)
        if means is not None and stds is not None:
            mean_ch = means if means.dim() == 2 else means.unsqueeze(-1)
            std_ch = stds if stds.dim() == 2 else stds.unsqueeze(-1)
            input_denorm = input * std_ch + mean_ch
            target_denorm = target * std_ch + mean_ch
            pred_xz = torch.cumsum(input_denorm[:, [6, 8], :], dim=2)
            target_xz = torch.cumsum(target_denorm[:, [6, 8], :], dim=2)
            pred_xz = pred_xz - pred_xz[:, :, :1]
            target_xz = target_xz - target_xz[:, :, :1]
            cumulative_root_loss = self.mse(pred_xz, target_xz) * 10
    
        
        recon_loss = rots_loss + joints_loss
       
        

        decoder_root_mse = self.mse(yaw_pred[:,:,:],yaw_target[:,:,:]) * 10 #.mean() * 10 #+ position_loss
        

        
        recon_loss = recon_loss + disp_loss * 1 + cumulative_root_loss * 0 + decoder_root_mse * 0
        vq_loss = vq_dict * 1
        
        soft_probs = target_indices.clone()
        original_output_logits = output_logits
        original_target_indices = target_indices
        
        if output_logits is None:
            # Stage 1 training runs no prior, so there is no prior cross-entropy to report
            hard_logits_loss = logits_loss = input.new_zeros(())
        else:
            output_logits= output_logits.view(-1,output_logits.size(-1))
            target_indices = target_indices.view(-1)
            hard_logits_loss = self.cross_entropy_loss(output_logits, target_indices)
            logits_loss = hard_logits_loss
            if codebook_weights is not None:
                soft_logits_loss = self.soft_codebook_cross_entropy(
                    original_output_logits,
                    original_target_indices,
                    codebook_weights,
                    temperature=soft_codebook_temperature,
                )
                soft_codebook_alpha = float(soft_codebook_alpha)
                logits_loss = (
                    (1.0 - soft_codebook_alpha) * hard_logits_loss
                    + soft_codebook_alpha * soft_logits_loss
                )

        if(foot_speed is not None):
            target_velo = target_velo.squeeze(2).clone() # [batch, frames, 1, j] -> [batch, frames, j]
            velo_loss = self.mse(foot_speed[:,1:], target_velo[:,2:,:]) * 100#.mean() * 100
            recon_loss += velo_loss * 0
        else:
            velo_loss = None

        if mu is None:
            # Stage 1 training has no timing network, so the timing diagnostics are zero
            timing_kl = temporal_loss = input.new_zeros(())
        else:
            timing_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            z = mu
            diff = z[:, 1:] - z[:, :-1]
            temporal_loss = torch.mean(torch.abs(diff**2)) * 1000  # CONTINUITY IN THE LATENT SPACE
        KLD_timing = timing_kl
        KLD_tags = timing_kl
        '''
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #KLD_s = 0.5 * torch.sum(logvar_shat - logvar_s + (torch.exp(logvar_s) + (mu_s - mu_shat)**2) / torch.exp(logvar_shat) - 1)
        #KLD_c = 0.5 * torch.sum(logvar_c - logvar_s + (torch.exp(logvar_s) + (mu_s - mu_c)**2) / torch.exp(logvar_c) - 1)
        KLD_tags = -0.5 * torch.sum(1 + logvar_s - mu_s.pow(2) - logvar_s.exp())
        KLD_timing = -0.5 * torch.sum(1 + logvar_shat - mu_shat.pow(2) - logvar_shat.exp())
        KLD_tags_and_latent = 0.5 * torch.sum(logvar - logvar_s + (torch.exp(logvar_s) + (mu_s - mu)**2) / torch.exp(logvar) - 1)
        KLD  /= (128*512)
        KLD_tags_and_latent /= (128*512)
        #KLD_s /= (128*512)
        KLD_tags /= (128*512)
        KLD_timing /= 10000#(128*512)

        KLD_tags = KLD_tags_and_latent
        #vq_loss = q_latent_loss + commitment_cost * e_latent_loss
        '''
        predicted_timing_loss = None

        if(predicted_vectors is not None):
            predicted_vectors_loss = None#self.mse(predicted_vectors, target_vectors)
        else:
            predicted_vectors_loss = recon_loss * 0
        if(recon_feet is not None):
            predicted_timing_from_h_loss = self.mse(timing_mu, timing_mu) * 10
        else: 
            predicted_timing_from_h_loss = recon_loss*0
        total_loss = recon_loss #* 10 #+ KLD * 2 # + KLD_s #+ KLD_c * 1.5#+ recon_feet_loss * 0.3 #recon_loss + vq_loss# + KLD + KLD_s #+ vq_loss
    
        pred_latent_loss = self.mse(pred_latent, true_latent)#.mean()

        self.debug_last_vq_vae_losses = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "rots_loss": rots_loss.detach(),
            "joints_loss": joints_loss.detach(),
            "disp_loss": disp_loss.detach(),
            "cumulative_root_loss": cumulative_root_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "temporal_loss": temporal_loss.detach(),
            "hard_logits_loss": hard_logits_loss.detach(),
            "logits_loss": logits_loss.detach(),
            "timing_kl": timing_kl.detach(),
            "velo_loss": velo_loss.detach() if velo_loss is not None else None,
        }

        return total_loss, vq_loss, temporal_loss, logits_loss, logits_loss, velo_loss, decoder_root_mse #return total_loss, KLD, temporal_loss, KLD_tags, predicted_timing_loss, velo_loss

        
    
            
    def rotate_root_channels(self, inp: torch.Tensor, tgt: torch.Tensor, phi: torch.Tensor):
        """
        Rotate both 9-channel root representations (6D ortho6d + 3D position) by yaw phi (scalar or [B], radians).
        Returns rotated copies; the inputs are not modified.
        """
        if phi is None:
            return inp, tgt

        device = inp.device
        dtype = inp.dtype
        B = inp.shape[0]

        if isinstance(phi, (float, int)):
            phi_t = torch.full((B,), float(phi), device=device, dtype=dtype)
        else:
            phi_t = torch.as_tensor(phi, device=device, dtype=dtype)
            if phi_t.dim() == 0:
                phi_t = phi_t.view(1).expand(B)
            elif phi_t.numel() == 1:
                phi_t = phi_t.view(1).expand(B)
            elif phi_t.numel() == B and phi_t.dim() == 1:
                pass
            else:
                # try to squeeze/broadcast to (B,)
                try:
                    phi_t = phi_t.view(B)
                except Exception:
                    raise ValueError(f"phi must be scalar or length batch_size ({B}), got {phi_t.shape}")

        inp_c = inp.clone()
        tgt_c = tgt.clone()

        root_in = inp_c[:, :9, :].permute(0, 2, 1).contiguous()   # [B, T, 9]
        root_tgt = tgt_c[:, :9, :].permute(0, 2, 1).contiguous()  # [B, T, 9]
        B, T, _ = root_in.shape
        N = B * T

        flat_in = root_in.reshape(N, 9)
        flat_tgt = root_tgt.reshape(N, 9)

        rot6_in = flat_in[:, :6].reshape(N, 3, 2)
        pos_in = flat_in[:, 6:9]
        rot6_tgt = flat_tgt[:, :6].reshape(N, 3, 2)
        pos_tgt = flat_tgt[:, 6:9]

        mats_in = ortho6d.to_matrix(rot6_in)
        mats_tgt = ortho6d.to_matrix(rot6_tgt)

        # build per-batch yaw R [B,3,3]
        cos_phi = torch.cos(phi_t).view(B, 1)
        sin_phi = torch.sin(phi_t).view(B, 1)
        R_batch = torch.zeros((B, 3, 3), device=device, dtype=dtype)
        R_batch[:, 0, 0] = cos_phi.squeeze(-1)
        R_batch[:, 0, 2] = sin_phi.squeeze(-1)
        R_batch[:, 1, 1] = 1.0
        R_batch[:, 2, 0] = -sin_phi.squeeze(-1)
        R_batch[:, 2, 2] = cos_phi.squeeze(-1)

        # expand to per-frame [N,3,3]
        Rb = R_batch.view(B, 1, 3, 3).expand(-1, T, -1, -1).reshape(N, 3, 3)

        new_mats_in = torch.matmul(Rb, mats_in)
        new_mats_tgt = torch.matmul(Rb, mats_tgt)

        # back to ortho6d 6-vector and rotate positions
        new_rot6_in = ortho6d.from_matrix(new_mats_in).reshape(B, T, 6)
        new_rot6_tgt = ortho6d.from_matrix(new_mats_tgt).reshape(B, T, 6)

        pos_in_rot = torch.matmul(Rb, pos_in.unsqueeze(-1)).squeeze(-1).reshape(B, T, 3)
        pos_tgt_rot = torch.matmul(Rb, pos_tgt.unsqueeze(-1)).squeeze(-1).reshape(B, T, 3)

        # reassemble back to [B,9,T]
        new_root_in = torch.cat([new_rot6_in, pos_in_rot], dim=-1).permute(0, 2, 1).contiguous()
        new_root_tgt = torch.cat([new_rot6_tgt, pos_tgt_rot], dim=-1).permute(0, 2, 1).contiguous()


        return new_root_in, new_root_tgt

# backward compatibility
MSE_DQ = GeneratorLoss


class MSE_DQ_FK(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = torch.Tensor(parents).long().to(device)
        self.device = device

        # indices without sparse input
        self.indices_no_sparse = []
        for i in range(0, 22):
            if i not in self.param["sparse_joints"]:
                self.indices_no_sparse.append(i)
        self.indices_no_sparse = torch.tensor(self.indices_no_sparse).to(self.device)

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs.unsqueeze(-1)

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs.unsqueeze(-1)

    def set_offsets(self, offsets):
        self.offsets = offsets  # denormalized
        self.joint_distances = torch.norm(offsets, dim=1)

    def forward_ik(self, input, input_ik, input_decoder, target):
        dqs = input_ik.clone()
        dqs_decoder = input_decoder.clone()
        target_dqs = target.clone()
        # add fake root rotation to input_ik (it is to keep the code compatible with previous versions)
        # without using the root rotation predicted from the decoder
        dqs = torch.cat((torch.zeros_like(dqs[:, :8, :]), dqs), dim=1)
        # FK Loss
        # denormalize dual quaternions
        dqs = dqs * self.std_dqs + self.mean_dqs
        dqs[:, :8, :] = (
            torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(-1)
        )  # make root quaternion identity (so asserts will pass)
        dqs = dqs.reshape((dqs.shape[0], -1, 8, dqs.shape[-1])).permute(0, 3, 1, 2)
        dqs_decoder = dqs_decoder * self.std_dqs + self.mean_dqs
        dqs_decoder[:, :8, :] = (
            torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(-1)
        )  # make root quaternion identity (no root)
        dqs_decoder = dqs_decoder.reshape(
            (dqs_decoder.shape[0], -1, 8, dqs_decoder.shape[-1])
        ).permute(0, 3, 1, 2)
        target_dqs = target_dqs * self.std_dqs + self.mean_dqs
        target_dqs = target_dqs.reshape(
            (target_dqs.shape[0], -1, 8, target_dqs.shape[-1])
        ).permute(0, 3, 1, 2)
        # compute rotations and translations local to their parents
        _, local_rot = from_root_dual_quat(dqs, self.parents)
        local_rot[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        _, local_rot_decoder = from_root_dual_quat(dqs_decoder, self.parents)
        local_rot_decoder[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        _, target_local_rot = from_root_dual_quat(target_dqs, self.parents)
        target_local_rot[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        # compute final positions (root space) using standard FK
        joint_poses, joint_rot_mat = fk(
            local_rot,
            torch.zeros((local_rot.shape[0], local_rot.shape[1], 3)),
            self.offsets,
            self.parents,
        )
        target_joint_poses, target_joint_rot_mat = fk(
            target_local_rot,
            torch.zeros((target_local_rot.shape[0], target_local_rot.shape[1], 3)),
            self.offsets,
            self.parents,
        )
        decoder_joint_poses, decoder_rot_mat = fk(
            local_rot_decoder,
            torch.zeros((local_rot_decoder.shape[0], local_rot_decoder.shape[1], 3)),
            self.offsets,
            self.parents,
        )

        # positions
        loss_ee = self.mse(
            joint_poses[:, :, self.param["sparse_joints"][1:], :],
            target_joint_poses[:, :, self.param["sparse_joints"][1:], :],
        )
        # rotations
        loss_ee += self.mse(
            joint_rot_mat[:, :, self.param["sparse_joints"][1:], :, :],
            target_joint_rot_mat[:, :, self.param["sparse_joints"][1:], :, :],
        )

        # regularization
        loss_ee_reg = self.mse(
            decoder_joint_poses[:, :, self.indices_no_sparse, :],
            joint_poses[:, :, self.indices_no_sparse, :],
        )
        loss_ee_reg += self.mse(
            decoder_rot_mat[:, :, self.indices_no_sparse, :, :],
            joint_rot_mat[:, :, self.indices_no_sparse, :, :],
        )

        return (
            loss_ee * self.param["lambda_ee"]
            + loss_ee_reg * self.param["lambda_ee_reg"]
        )


class MSE_Latent(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = parents
        self.device = device

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs.unsqueeze(-1)

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs.unsqueeze(-1)

    def set_offsets(self, offsets):
        self.joint_distances = torch.norm(offsets, dim=1)

    def compute_latent_loss(self, input, target):
        input_latent = input.clone()
        target_latent = target.clone()
        # Dual Quaternions MSE Loss
        #len = int(input.shape[2] / 2)
        
        latent_loss = torch.sqrt(1+self.mse(input_latent,target_latent))
        return latent_loss
  

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.skeleton_layers import (
    SkeletonConv,
    create_pooling_list,
    find_neighbor,
)
from networks.unified.components import (
    dummy_joint,
    num_layers,
    resolve_vae_latent_dim,
)


class Encoder(nn.Module):
    def __init__(self, param, parents, device, is_vae = False, is_vq_vae = False):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        self.convs = []
        self.convs_2 = []
        self.parents = [parents]
        self.pooling_lists = []
        self.channel_list = []
        self.channel_list_2 = []
        self.is_vae = is_vae
        self.is_vq_vae = is_vq_vae
        self.device = device
        self.param = param
        self.vae_latent_dim = resolve_vae_latent_dim(param)

        kernel_size = 9 #param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2
        stride = param["stride_encoder_conv"]

        survivors_list = []
        number_layers = num_layers
        layer_parents = parents
        for l in range(number_layers):
            pooling_list, layer_parents = create_pooling_list(
                layer_parents, l == number_layers - 1
            )
                    
            # original joint indices that survived (not pooled away)
            survivors = [grp[0] for grp in pooling_list]
            survivors_list.append(survivors)
            # mapping from original joint -> pooled index
            # old_to_new = {}
            # for new_idx, grp in enumerate(pooling_list):
            #     for old_idx in grp:
            #         old_to_new[old_idx] = new_idx

            self.pooling_lists.append(pooling_list)
            self.parents.append(layer_parents)
            
        # sparse joints survivors #
        # idxs_level1 = [survivors_list[1][i] for i in survivors_list[2]]
        # result = [survivors_list[0][j] for j in idxs_level1]
        # print(result)

        default_channels_per_joints = 9
        self.channel_size = [default_channels_per_joints] + [
            (default_channels_per_joints) * (2**i)
            for i in range(1, number_layers + 1)
        ]  # first each joint has 8 (dual quaternion) channels, then we multiply by 2 every layer to increase the number of conv. filters
        low_res_parents = self.parents[-1]
        neighbor_list, _ = find_neighbor(
            low_res_parents, param["neighbor_distance"], add_displacement=True     #<---
        )
        self.num_joints = len(neighbor_list)  # it includes the displacement fake joint
        if(dummy_joint):
            self.num_joints+=1

        for l in range(number_layers):
            seq = []
            in_channels = self.channel_size[l] * self.num_joints
            out_channels = self.channel_size[l + 1] * self.num_joints
            if l == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            
            if l == number_layers-1:
                last_layer_scale = 1
            else:
                last_layer_scale = 1    
                
            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[l],
                    out_channels_per_joint=self.channel_size[l + 1] * last_layer_scale,
                    joint_num=self.num_joints,
                    in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                    padding=padding,
                    stride=stride * last_layer_scale,
                    device=device,
                    add_offset=False,
                )
            )
            
            self.convs.append(seq[-1])
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            #seq.append(nn.Tanh())
            #seq.append(nn.BatchNorm1d(out_channels))
            #seq.append(ConvLayerNorm(out_channels * last_layer_scale))
            self.layers.append(nn.Sequential(*seq))
      
        self.output_projection = nn.Sequential(
            nn.Linear(self.channel_list[-1],self.vae_latent_dim),
        )
        
        self.res_downsample = nn.Sequential(
            SkeletonConv(
                param=param,
                neighbor_list=neighbor_list,
                kernel_size=1,
                in_channels_per_joint=self.channel_size[1],
                out_channels_per_joint=self.channel_size[-1] * last_layer_scale,
                joint_num=self.num_joints,
                in_offset_channel=3*self.channel_size[-1] // self.channel_size[0],
                padding=0,
                stride=8,
                device=device,
                add_offset=False,
            ),
        )
        vq_dim = self.channel_list[-1]*last_layer_scale # self.channel_list[0] * 8
        
        self.group_size = 1
        self.grouped_vq_dim = vq_dim * self.group_size
        self.downsample = nn.Sequential(
            nn.Conv1d(self.channel_list[-1], self.channel_list[-1], kernel_size = kernel_size, padding = kernel_size // 2, stride = stride),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.channel_list[-1], self.channel_list[-1], kernel_size = kernel_size, padding = kernel_size // 2, stride = stride),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        base_ch = self.channel_list[0]  # vq_dim / input channels
        in_ch_per_joint = base_ch // max(1, self.num_joints)
        num_stages = 3 
        kernel_size = 3
        # Encoder: per-stage SkeletonConv with stride=2
        enc_layers = []
        for i in range(num_stages):
            in_c = in_ch_per_joint * (2 ** i)
            out_c = in_ch_per_joint * (2 ** (i + 1))
            enc_layers.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=in_c,
                    out_channels_per_joint=out_c,
                    joint_num=self.num_joints,
                    in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                    padding=kernel_size//2,
                    stride=2,
                    device=device,
                    add_offset=False,
                )
            )
            enc_layers.append(nn.LeakyReLU(0.2))
        self.sparse_enc = nn.Sequential(*enc_layers)
        # Decoder: reverse of encoder. Upsample (x2) then SkeletonConv (stride=1)
        kernel_size = 3
        dec_layers = []
        for i in range(num_stages):
            # stage index from top (deepest) to shallow
            in_c = in_ch_per_joint * (2 ** (num_stages - i))
            out_c = in_ch_per_joint * (2 ** (num_stages - i - 1))
            dec_layers.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            dec_layers.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=in_c,
                    out_channels_per_joint=out_c,
                    joint_num=self.num_joints,
                    in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                    padding=kernel_size//2,
                    stride=1,
                    device=device,
                    add_offset=False,
                )
            )
            dec_layers.append(nn.LeakyReLU(0.2))
        self.sparse_dec = nn.Sequential(*dec_layers)
        # Decoder: upsample then per-joint SkeletonConv (reverse channels)
     


        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode='linear', align_corners=False),
            nn.Conv1d(vq_dim, vq_dim, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Upsample(scale_factor=stride, mode='linear', align_corners=False),
            nn.Conv1d(vq_dim, vq_dim, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.projection_no_quats = nn.Linear(self.channel_list[0]-8, self.channel_list[0])
        
        self.vq_mlp = nn.Sequential(
            nn.Linear(self.channel_list[-1] * last_layer_scale, vq_dim),  # Project to latent space
        )
        self.vq_ln = nn.LayerNorm(vq_dim)
        
        self.vq_ffn = nn.Sequential(
            nn.Linear(vq_dim, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, vq_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.vq_fc = nn.Sequential(
            nn.Linear(vq_dim, vq_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.gs_tau_start = 1.0
        self.gs_tau_min = 0.1
        self.gs_tau_decay = 1e-4
        self.register_buffer("gs_tau", torch.tensor(self.gs_tau_start))

        # Projection layer for logits
        
        self.num_embeddings = self.param["codebook_size"]
        self.logits_proj = nn.Linear(vq_dim, self.num_embeddings)
        self.logits_proj_grouped = nn.Linear(self.grouped_vq_dim, self.num_embeddings)
    
        self.num_quantizers = 1
        self.vq_codebooks = nn.ModuleList(
            [nn.Embedding(self.num_embeddings, self.grouped_vq_dim) for _ in range(self.num_quantizers)]
        )
        for codebook in self.vq_codebooks:
            codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
            
        self.commitment_cost = 0.75

        self.use_codebook = False

         # ---------- EMA buffers for codebook maintenance ----------
        self.ema_decay = self.param["ema_decay"]
        self.ema_eps = 1e-5
        
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        # ema_w must match the codebook embedding dimensionality (grouped_vq_ddim)
        self.register_buffer("ema_w", torch.zeros(self.num_embeddings, self.grouped_vq_dim))
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        # runtime-only switch (not a buffer): False freezes ema_updates while the continuous AE warmup runs
        self.ema_enabled = True
        
        self.test_layer = nn.Sequential(
            nn.Conv1d(self.channel_list[-1],self.channel_list[-1],kernel_size=3, stride=1,padding=3//2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        
        if(self.use_codebook):
            self.load_codebook("cluster_centers_test.npy")
            print("codebook shape", self.codebook.shape)
            num_codebook_vectors = self.codebook.shape[0]

    def step_gs_tau(self):
        """Anneal Gumbel-Softmax temperature during training"""
        if self.training:
            new_tau = max(self.gs_tau_min, float(self.gs_tau) * (1.0 - self.gs_tau_decay))
            self.gs_tau.fill_(new_tau)

        with torch.no_grad():
            self.ema_updates += 1
    
    def incr_ema_updates(self):
        with torch.no_grad():
            self.ema_updates += 1

    # ================= EMA & Reset Utilities =================
        
    @torch.no_grad()
    def ema_update(self, indices: torch.Tensor, encoder_latent: torch.Tensor, lvl: int = 0):
        """
        Update codebook weights and EMA counts/sums from assigned latents for the selected level.
        indices [B, T]; encoder_latent [B, T, D] (the residual for lvl > 0).
        """
        if indices is None or encoder_latent is None:
            return
            
        # # Use residual for level 1 if available
        if lvl == 1 and hasattr(self, 'residual'):
            source_z = self.residual  # Use stored residual for level 1
        else:
            source_z = encoder_latent  # Use original latents for level 0
        #source_z = encoder_latent
        # Ensure we have appropriate EMA buffers for this level
        if not hasattr(self, f'ema_cluster_size_lvl{lvl}'):
            # Create level-specific EMA buffers if they don't exist
            if lvl == 0:
                # Level 0 uses the original buffers
                ema_cluster_size = self.ema_cluster_size
                ema_w = self.ema_w
            else:
                # For other levels, create new buffers
                self.register_buffer(f'ema_cluster_size_lvl{lvl}', 
                                    torch.zeros_like(self.ema_cluster_size))
                self.register_buffer(f'ema_w_lvl{lvl}', 
                                    torch.zeros_like(self.ema_w))
                ema_cluster_size = getattr(self, f'ema_cluster_size_lvl{lvl}')
                ema_w = getattr(self, f'ema_w_lvl{lvl}')
        else:
            # Use existing level-specific buffers
            ema_cluster_size = getattr(self, f'ema_cluster_size_lvl{lvl}')
            ema_w = getattr(self, f'ema_w_lvl{lvl}')
        
        flat_idx = indices.reshape(-1).clamp(0, self.num_embeddings - 1)          # [N]
        flat_z = source_z.reshape(-1, source_z.size(-1))                          # [N,D]
        
        one_hot = F.one_hot(flat_idx, num_classes=self.num_embeddings).float()  # [N,K]
        counts = one_hot.sum(0)                                                # [K]
        dw = one_hot.t() @ flat_z                                              # [K,D]
        
        ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
        
        n = ema_cluster_size.sum()
        if n.item() == 0:
            return
        cluster_size = (ema_cluster_size + self.ema_eps) / (n + self.num_embeddings * self.ema_eps) * n
        denom = cluster_size.unsqueeze(1).clamp_min(self.ema_eps)
        new_weight = ema_w / denom
        
        self.vq_codebooks[lvl].weight.data.copy_(new_weight)

        
    @torch.no_grad()   
    def reset_unused_codebook_vectors(self, min_usage_frac=1e-3, sample_z: torch.Tensor = None, lvl = 0):
        """
        Replace underused codes and reset their EMA statistics consistently.
        Replacements are sampled from sample_z [B, T, D] when given, otherwise random.
        """
        if lvl == 0:
            ema_cluster_size = self.ema_cluster_size
            ema_w = self.ema_w
        elif hasattr(self, f'ema_cluster_size_lvl{lvl}'):
            ema_cluster_size = getattr(self, f'ema_cluster_size_lvl{lvl}')
            ema_w = getattr(self, f'ema_w_lvl{lvl}')
        else:
            print(f"No EMA cluster size for level {lvl}, skipping reset")
            return
        
        total = ema_cluster_size.sum().clamp_min(1.0)
        usage = ema_cluster_size / total
        mask = usage < min_usage_frac
        
        # Skip if no vectors need resetting
        if not mask.any():
            return
        
        # Determine how many vectors to reset
        num_reset = mask.sum().item()
        D = self.vq_codebooks[lvl].weight.size(1)
        
        if sample_z is not None and sample_z.numel() > 0:
            flat = sample_z.reshape(-1, D)
            rand_idx = torch.randint(0, flat.size(0), (num_reset,), device=flat.device)
            new_vecs = flat[rand_idx]
        else:
            new_vecs = torch.randn(num_reset, D, device=self.vq_codebooks[lvl].weight.device) * 0.02
        
        self.vq_codebooks[lvl].weight.data[mask] = new_vecs
        # give them a small non-zero count to avoid immediate re-reset
        ema_cluster_size.data[mask] = min_usage_frac * total
        # the weighted sum must agree with the new vector and count, using the same
        # smoothing as ema_update. otherwise the next update rebuilds the old vector.
        n = ema_cluster_size.sum()
        smoothed = (ema_cluster_size + self.ema_eps) / (n + self.num_embeddings * self.ema_eps) * n
        ema_w.data[mask] = new_vecs * smoothed[mask].unsqueeze(1)
        

    @torch.no_grad()
    def init_codebook_from_centroids(self, centroids: torch.Tensor, counts: torch.Tensor):
        """
        Initialize level-0 codebook and EMA buffers from warmup centroids, once, when warmup hands over to VQ training.
        The next EMA update must reproduce those weights; counts holds per-centroid EMA mass in ema_update units.
        """
        K, D = self.num_embeddings, self.grouped_vq_dim
        if tuple(centroids.shape) != (K, D):
            raise ValueError(f"centroids must have shape {(K, D)}, got {tuple(centroids.shape)}")
        if tuple(counts.shape) != (K,):
            raise ValueError(f"counts must have shape {(K,)}, got {tuple(counts.shape)}")
        if not torch.isfinite(centroids).all():
            raise ValueError("codebook centroids contain non-finite values")

        centroids = centroids.to(self.ema_w.device, torch.float32)
        # a code with no mass is floored at ema_eps, so the smoothing below stays above its clamp and the centroid is kept
        ema_cluster_size = counts.to(self.ema_w.device, torch.float32).clamp_min(self.ema_eps)
        n = ema_cluster_size.sum()
        smoothed = (ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        self.vq_codebooks[0].weight.data.copy_(centroids)
        self.ema_cluster_size.copy_(ema_cluster_size)
        self.ema_w.copy_(centroids * smoothed.unsqueeze(1))
        self.ema_updates.zero_()

        # the weights ema_update would write from these buffers must reproduce the centroids
        cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + K * self.ema_eps) * n
        recomputed = self.ema_w / cluster_size.unsqueeze(1).clamp_min(self.ema_eps)
        if not torch.allclose(recomputed, centroids, rtol=1e-4, atol=1e-6):
            max_error = (recomputed - centroids).abs().max().item()
            raise RuntimeError(f"EMA buffers do not reproduce the centroids (max abs error {max_error:.3e})")

    # =============== Group Stuff ================# 
    def _group_latent(self, z: torch.Tensor):
        """
        Group adjacent latent timesteps, padding the final group if needed.
        z [B, T, D] -> z_group [B, G, group_size*D]; returns (z_group, T_orig, pad).
        """
        B, T, D = z.shape
        g = self.group_size
        pad = (g - (T % g)) % g
        if pad > 0:
            z = torch.cat([z, torch.zeros(B, pad, D, device=z.device, dtype=z.dtype)], dim=1)
        T_new = z.size(1)
        G = T_new // g
        z_group = z.view(B, G, g, D).reshape(B, G, g * D)
        return z_group, T, pad
        
    def split_quantized_groups(self, grouped: torch.Tensor, T_orig: int, pad: int):
        """Undo latent grouping and remove padding: [B, G, group_size*D] -> [B, T_orig, D]."""
        B, G, GD = grouped.shape
        g = self.group_size
        D = GD // g
        per_frame = grouped.view(B, G, g, D).reshape(B, G * g, D)
        if pad > 0:
            per_frame = per_frame[:, :T_orig]
        return per_frame


    def forward(self, input, phase=None, phase_per_8_frames = None, velocity_per_8_frames = None, get_tags = False, train_lstm = False, tags=None, predict_from_timing=False):
        # if using dual quaternions, all channels have 8 components but the last one (displacement)
        # has 3 channels. Pad the last one with zeros so that all have 8 components
        # input = torch.cat(
        #     (input, torch.zeros_like(input[:, [0, 1, 2, 3, 4], :])), dim=1
        # )
        input = torch.cat(
            (input, torch.zeros_like(input[:, [0, 1, 2, 3, 4], :])), dim=1
        )

    
        self.original_sparse = input.clone()
        
        original = input.clone()
        for i,layer in enumerate(self.layers):
            input = layer(input)
            if(i==0):
                intermediate = input.clone()
        self.enc_out = input.clone()

        use_gumbel = 0
        if (self.is_vq_vae and not use_gumbel):

            if self.training and self.ema_enabled:
                self.incr_ema_updates()
        
            z = input.permute(0,2,1).clone()
            B, T, D = z.shape

            # Group consecutive timesteps and concatenate features
            z_grouped, T_orig, pad = self._group_latent(z)  # [B, G, group_size*D] where G = T//group_size
            self.grouped_latent = z_grouped.clone()
            B_g, G, GD = z_grouped.shape
            flat_inputs = z_grouped.reshape(-1, GD)  # [B*G, group_size*vq_dim]

            # Residual vector quantization on grouped features
            device = flat_inputs.device
            residual = flat_inputs.clone()
            quantized_sum = torch.zeros_like(residual, device=device)
            all_indices = []
            vq_loss_total = torch.tensor(0.0, device=device)

            for level, codebook in enumerate(self.vq_codebooks):
                # distances: [N, K]
                
                distances = (
                    torch.sum(residual * residual, dim=1, keepdim=True)  # [N,1]
                    + torch.sum(codebook.weight * codebook.weight, dim=1).unsqueeze(0)  # [1,K]
                    - 2.0 * (residual @ codebook.weight.t())  # [N,K]
                )

                
                topk = torch.topk(distances, k=3, dim=1, largest=False)
                topk_indices = topk.indices  # [N, 3]
                idx0 = topk_indices[:, 0]    # always the closest
                N = distances.size(0)
                # 5% chance to pick randomly among top 3 (excluding argmin)
                rand = torch.rand(N, device=distances.device)
                choose_random = rand < 0.075
                idx_final = idx0.clone()
                if choose_random.any() and False:
                    # For those, randomly pick 1 or 2 (not 0)
                    alt_idx = torch.randint(1, 3, (choose_random.sum(),), device=distances.device)
                    idx_final[choose_random] = topk_indices[choose_random, alt_idx]
                encoding_indices = idx_final.unsqueeze(1)  # [N,1]
                
                all_indices.append(encoding_indices)
                
                # One-hot and lookup
                N = encoding_indices.size(0)
                encodings = torch.zeros(N, codebook.num_embeddings, device=device)
                encodings.scatter_(1, encoding_indices, 1.0)
                quantized = encodings @ codebook.weight  # [N, GD]

                # VQ losses (pull codebook to residual and encoder to codebook)
                codebook_loss = F.mse_loss(quantized, residual.detach())
                commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), residual)
                vq_loss = codebook_loss*0 + commitment_loss
                vq_loss_total = vq_loss_total + vq_loss

                # Straight-through estimator for this level
                quantized_st = residual + (quantized - residual).detach()
                # Accumulate quantized vectors and update residual
                quantized_sum = quantized_sum + quantized_st
                residual = residual - quantized_st
                if(level == 1):
                    self.residual = quantized.clone()
                

            encoding_indices = torch.cat(all_indices, dim=1)  # [N, L]
            encoding_indices = encoding_indices.view(B, G, -1)  # [B, G, L]
            quantized_grouped = quantized_sum.view(B, G, GD)  # [B, G, group_size*vq_dim]

            # Split back to per-frame for decoder (returns [B, T_orig, D_per_frame])
            quantized_per_frame = self.split_quantized_groups(quantized_grouped, T_orig, pad)
            

            return quantized_per_frame.permute(0,2,1), encoding_indices[:, :, 0].clone(), vq_loss_total, encoding_indices.clone()
            
            
            
        if (self.is_vq_vae and use_gumbel):
            # Gumbel-Softmax residual VQ over GROUPED latent (mirrors non-gumbel path)
            if self.training:
                self.step_gs_tau()

            # Project to latent and layer-norm
            z = self.vq_ln(self.vq_mlp(input.permute(0, 2, 1).clone()))  # [B, T, vq_dim]
            B, T, D = z.shape

            # Group consecutive timesteps
            z_grouped, T_orig, pad = self._group_latent(z)               # [B, G, group_size*D]
            self.grouped_latent = z_grouped.clone()
            B_g, G, GD = z_grouped.shape
            flat = z_grouped.reshape(-1, GD)                             # [B*G, GD]

            residual = flat.clone()
            quantized_accum = torch.zeros_like(residual)
            hard_indices_levels = []
            vq_loss_total = 0.0

            tau = float(self.gs_tau)

            for level, codebook in enumerate(self.vq_codebooks):
                # logits from grouped residual
                logits = self.logits_proj_grouped(residual)              # [N, K]
                if self.training:
                    probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # straight-through
                else:
                    idx = torch.argmax(logits, dim=-1)
                    probs = F.one_hot(idx, num_classes=self.num_embeddings).float()

                # Hard indices for logging
                hard_idx = torch.argmax(probs.detach(), dim=-1)          # [N]
                hard_indices_levels.append(hard_idx.unsqueeze(1))        # [N,1]

                # Soft quantized vector (grouped dimensionality)
                quantized_soft = probs @ codebook.weight                 # [N, GD]

                # VQ loss (codebook + commitment) with soft assignment (same style as non-gumbel path)
                codebook_loss = F.mse_loss(quantized_soft.detach(), residual)
                commitment_loss = self.commitment_cost * F.mse_loss(quantized_soft, residual.detach())
                vq_loss = codebook_loss + commitment_loss
                vq_loss_total = vq_loss_total + vq_loss

                # Straight-through
                quantized_st = quantized_soft
                quantized_accum = quantized_accum + quantized_st
                residual = residual - quantized_st

            # Collect indices
            hard_indices = torch.cat(hard_indices_levels, dim=1)         # [N, L]
            hard_indices = hard_indices.view(B, G, -1)                   # [B, G, L]

            # Ungroup back to per-frame
            quantized_grouped = quantized_accum.view(B, G, GD)           # [B, G, GD]
            quantized_per_frame = self.split_quantized_groups(quantized_grouped, T_orig, pad)  # [B, T_orig, D]
            quantized_per_frame = self.vq_fc(quantized_per_frame)


            return quantized_per_frame, hard_indices[:, :, 0].clone(), vq_loss_total, hard_indices.clone()


        return input
    
    
    def indices_to_vector_(self, indices):
        """Sum codebook embeddings for each index sequence and undo temporal grouping."""
        if indices.dim() == 2:  # Handle single-level indices for compatibility
            indices = indices.unsqueeze(-1)

        B, T, L = indices.shape
        vectors = torch.zeros_like(self.vq_codebooks[0].weight[0]).expand(B, T, -1).to(indices.device)
       
        for l in range(L):
            vectors = vectors + self.vq_codebooks[l](indices[..., l])  # Sum embeddings from all levels
        vectors = self.split_quantized_groups(vectors, T*self.group_size, pad=0)
        return vectors

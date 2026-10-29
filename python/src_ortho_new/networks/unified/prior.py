import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unified.components import ConvLayerNorm

# evaluation-only prior modes: "rollout" is free-running sampling (inference), "teacher_forced" feeds ground-truth codes
PRIOR_EVAL_MODES = ("rollout", "teacher_forced")


class RecurrentPrior(nn.Module):
    """Autoregressive GRU prior over VQ code indices, conditioned on motion controls."""
    def __init__(self,
                 ctrl_dim=6,
                 num_codebook_vectors=256,
                 num_levels=1,
                 vq_dim=256,
                 hidden_dim=512,
                 window_size=4,
                 stride=8,
                 kernel_size=7,
                 num_layers=2,
                 dropout=None,
                 use_hidden_state=None,
                 *,
                 variant,
                 passthrough_channels=0,
                 scale_velocity_channels=True,
                 teacher_forcing_prob=None,
                 ctrl_channel_dropout=None,
                 gru_dropout=None):
        super().__init__()
        self.variant = variant
        # trailing control channels that dropout never removes (foot-contact channels)
        self.passthrough_channels = int(passthrough_channels)
        # velocity scaling applies to channels 0 and 1, which are velocity only without foot contact
        self.scale_velocity_channels = bool(scale_velocity_channels)
        dropout = variant.prior_dropout if dropout is None else dropout
        # None keeps the variant's value (the GRU falls back to the general dropout, as before)
        gru_dropout = (variant.gru_dropout if variant.gru_dropout is not None else dropout) if gru_dropout is None else gru_dropout
        if use_hidden_state is None:
            use_hidden_state = variant.hidden_state == "carry"
        self.use_hidden_state = bool(use_hidden_state)
        self.ctrl_dim = ctrl_dim
        self.num_codebook_vectors = num_codebook_vectors
        self.num_levels = num_levels
        self.vq_dim = vq_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.stride = stride
        
        self.per_channel_ctrl_out = 8
        self.ctrl_out_dim = self.ctrl_dim * self.per_channel_ctrl_out

        # Control downsampler (8x reduction to match VQ rate)
        # self.ctrl_downsample = nn.Sequential(
        #     nn.Conv1d(ctrl_dim, hidden_dim, kernel_size=kernel_size, 
        #              stride=stride, padding=kernel_size//2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(dropout)
        # )
        
        self.ctrl_channel_dropout_prob = variant.ctrl_channel_dropout if ctrl_channel_dropout is None else ctrl_channel_dropout
        # If True, drop per-timestep instead of whole-channel; default False (whole-channel drop)
        self.ctrl_channel_dropout_per_timestep = True
        self.velocity_scale_range = variant.velocity_scale_range
        
        self.ctrl_downsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.per_channel_ctrl_out//4, kernel_size=kernel_size,
                stride= 2, padding=kernel_size//2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                
                nn.Conv1d(self.per_channel_ctrl_out//4, self.per_channel_ctrl_out//2, kernel_size=kernel_size,
                stride=2, padding=kernel_size//2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            
                nn.Conv1d(self.per_channel_ctrl_out//2, self.per_channel_ctrl_out, kernel_size=kernel_size,
                stride=2, padding=kernel_size//2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                
                ConvLayerNorm(self.per_channel_ctrl_out),
            ) for _ in range(self.ctrl_dim)
        ])


        self.ctrl_downsampler = nn.Sequential(
                nn.Conv1d(self.ctrl_dim, self.ctrl_out_dim, kernel_size=15, stride=8, padding=15//2),
                nn.LeakyReLU(0.2),
                ConvLayerNorm(self.ctrl_out_dim),
            )

        # Hybrid control encoder:
        # 1) light per-channel stems preserve signal identity
        # 2) shared mixer learns cross-channel interactions after downsampling
        # 3) raw residual branch keeps subtle control cues available
        self.hybrid_ctrl_stem_dim = self.per_channel_ctrl_out
        self.hybrid_ctrl_concat_dim = self.ctrl_dim * self.hybrid_ctrl_stem_dim
        self.hybrid_ctrl_raw_dim = self.ctrl_out_dim
        self.hybrid_ctrl_out_dim = self.ctrl_out_dim

        self.ctrl_hybrid_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.hybrid_ctrl_stem_dim // 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Conv1d(self.hybrid_ctrl_stem_dim // 2, self.hybrid_ctrl_stem_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Conv1d(self.hybrid_ctrl_stem_dim, self.hybrid_ctrl_stem_dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
                nn.LeakyReLU(0.2),
                ConvLayerNorm(self.hybrid_ctrl_stem_dim),
            ) for _ in range(self.ctrl_dim)
        ])

        self.ctrl_hybrid_raw_downsampler = nn.Sequential(
            nn.Conv1d(self.ctrl_dim, self.hybrid_ctrl_raw_dim, kernel_size=15, stride=8, padding=15 // 2),
            nn.LeakyReLU(0.2),
            ConvLayerNorm(self.hybrid_ctrl_raw_dim),
        )

        self.ctrl_hybrid_mixer = nn.Sequential(
            nn.Conv1d(self.hybrid_ctrl_concat_dim + self.hybrid_ctrl_raw_dim, self.hybrid_ctrl_out_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            ConvLayerNorm(self.hybrid_ctrl_out_dim),
            nn.Conv1d(self.hybrid_ctrl_out_dim, self.hybrid_ctrl_out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            ConvLayerNorm(self.hybrid_ctrl_out_dim),
        )
        
        # Embedding tables for each level (match VQ codebooks)
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(num_codebook_vectors, vq_dim) for _ in range(num_levels)
        ])
        for emb in self.level_embeddings:
            nn.init.normal_(emb.weight, 0.0, 0.02)

        # Project summed level embeddings to hidden space
        self.emb_proj = nn.Linear(vq_dim, hidden_dim)

        # Recurrent core with windowed context
        # rnn_input_size = hidden_dim * 2
        rnn_input_size = hidden_dim + self.ctrl_out_dim
        self.rnn = nn.GRU(
            input_size=rnn_input_size,  
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        
        self.teacher_forcing_prob = variant.teacher_forcing_prob if teacher_forcing_prob is None else teacher_forcing_prob
        self.teacher_forcing_min = variant.teacher_forcing_min
        self.state_reset_interval = 128 # 16

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_codebook_vectors)
            ) for _ in range(num_levels)
        ])
        
        self.rot_proj = nn.Sequential(
            nn.Linear(hidden_dim + 16 ,128),
            nn.LeakyReLU(0.2),
        )
        
        kernel_size = 5
        self.rot_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(128,64,kernel_size=kernel_size,stride=1,padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(64,32,kernel_size=kernel_size,stride=1,padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(32,9,kernel_size=kernel_size,stride=1,padding=kernel_size//2),
            nn.LeakyReLU(0.2), # <-------
        )

        # self.rot_cnn = nn.Sequential(
        #     nn.ConvTranspose1d(16, 64, kernel_size=6, stride=2, padding=2),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose1d(64, 128, kernel_size=6, stride=2, padding=2),
        #     nn.LeakyReLU(0.2),
        #     # nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
        #     # nn.Conv1d(128, 9, kernel_size=7, stride=1, padding= 7//2)
        #     nn.ConvTranspose1d(128, 9, kernel_size=6, stride=2, padding=2),
        # )
        
        self.rot_smoother = nn.Conv1d(9, 9, kernel_size=5, padding=5//2, groups=9, bias=False)
        # init as simple averaging kernel
        with torch.no_grad():
            avg_kernel = torch.ones(5, dtype=torch.float32) / 5.0
            k = avg_kernel.view(1, 1, -1)  # [out_ch_per_group, in_ch_per_group, K]
            # assign to each group's weights
            self.rot_smoother.weight.data.copy_(k.repeat(9, 1, 1))
            
        self.yaw_downsampler = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, stride=8, padding=7),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.rot_embed_proj = nn.Linear(vq_dim, 16)
        

        self.dropout = nn.Dropout(dropout)
    
    def anneal_teacher_forcing(self, factor: float):
        """Multiply teacher forcing prob by factor (call from training loop per epoch)."""
        self.teacher_forcing_prob = max(
            self.teacher_forcing_min,
            float(self.teacher_forcing_prob) * float(factor),
        )
        
    
    def _keep_passthrough(self, mask):
        # passthrough channels are never dropped: their missing state is already explicit
        if self.passthrough_channels:
            mask[..., -self.passthrough_channels:] = 1.0
        return mask

    def _encode_controls(self, ctrl_seq: torch.Tensor):
        B, T_full, C = ctrl_seq.shape
        device = ctrl_seq.device
        seq = ctrl_seq
        if self.scale_velocity_channels and self.velocity_scale_range is not None and self.training and C > 0:
            scale_min, scale_max = (float(v) for v in self.velocity_scale_range)
            velocity_scale = torch.empty(B, 1, 1, device=device).uniform_(scale_min, scale_max)
            seq = seq.clone()
            seq[:, :, 0:1] = seq[:, :, 0:1] * velocity_scale
            seq[:, :, 1:2] = seq[:, :, 1:2] * velocity_scale
        p = float(self.ctrl_channel_dropout_prob)
        # channel dropout (optionally per-timestep). keep similar corruption semantics.
        if self.training and p > 0.0:
            if not self.ctrl_channel_dropout_per_timestep:
                # zero entire channels with probability p per sample
                mask = (torch.rand(B, C, device=device) > p).float().unsqueeze(1)  # [B,1,C]
                mask = self._keep_passthrough(mask)
                seq = seq * mask  # broadcast over time
            else:
                # per-timestep channel dropout
                mask = (torch.rand(B, T_full, C, device=device) > p).float()
                mask = self._keep_passthrough(mask)
                seq = seq * mask
        x = seq.permute(0, 2, 1).contiguous()
        ds = self.ctrl_downsampler(x)
        return ds.permute(0, 2, 1).contiguous()


    
    def forward(self,
                ctrl_seq: torch.Tensor,
                target_indices: torch.Tensor = None,
                temperature: float = 1.0,
                max_length: int = None,
                top_k: int = None,
                codebooks: torch.Tensor = None,
                yaw_sin_cos: torch.Tensor = None):
        """
        Predict code indices and logits from controls [B, T_full, ctrl_dim].
        In training, target codes are fed back with probability teacher_forcing_prob; otherwise inference runs autoregressively.
        Returns indices [B, T_out, L] and logits [B, T_out, L, K].
        """
        ctrl_down = self._encode_controls(ctrl_seq)
        B, T_ds, H = ctrl_down.shape
        device = ctrl_down.device

        if self.training and target_indices is not None:
            return self._forward_train(ctrl_down, target_indices, codebooks=codebooks, yaw_sin_cos=yaw_sin_cos)
        else:
            seq_len = max_length if max_length is not None else T_ds
            return self._forward_inference(
                ctrl_down,
                seq_len,
                temperature,
                top_k=top_k,
                codebooks=codebooks,
                yaw_sin_cos=yaw_sin_cos,
            )


    
    def _forward_train(self, ctrl_down: torch.Tensor, target_indices: torch.Tensor, codebooks = None, yaw_sin_cos: torch.Tensor = None):
        """Training with teacher forcing 40% of the time"""
        B, T_ds, H = ctrl_down.shape
        seq_len = min(T_ds, target_indices.size(1))
        device = ctrl_down.device

        all_logits = []
        all_indices = []
        rnn_outs = []
        all_embs = []
        
        emb_history = torch.zeros(B, self.window_size, self.vq_dim, device=device)
        hidden_state = None
        
        # Track predictions vs ground truth usage
        predicted_embs = torch.zeros(B, seq_len, self.vq_dim, device=device)
        
        # First pass: collect GT embeddings for fallback
        gt_level_embs = [self.level_embeddings[l](target_indices[..., l]) for l in range(self.num_levels)]
        gt_sum_emb = torch.stack(gt_level_embs, dim=2).sum(dim=2)   # [B, T_ds, vq_dim]

        for t in range(seq_len):
            if t > 0 and self.state_reset_interval > 0 and t % self.state_reset_interval == 0:
                hidden_state = None
            
            w = min(self.window_size, t+1)
            emb_window = emb_history[:, -w: ,:]
            emb_proj_seq = self.emb_proj(emb_window)
            
            start = max(0, t-w+1)
            ctrl_window = ctrl_down[:, start:t+1]
            rnn_input = torch.cat([ctrl_window,emb_proj_seq],dim=-1)
            
            # Current embedding context (windowed history)
            # emb_t = emb_history[:, -1:, :]  # [B, 1, vq_dim]
            # emb_proj = self.emb_proj(emb_t)  # [B, 1, H]
            
            # Combine control + embedding
            # rnn_input = torch.cat([ctrl_t, emb_proj], dim=-1)  # [B, 1, 2H]
            
            rnn_out, next_hidden_state = self.rnn(
                rnn_input, hidden_state if self.use_hidden_state else None
            )
            if self.use_hidden_state:
                hidden_state = next_hidden_state
            rnn_out = rnn_out[:,-1:]
            rnn_outs.append(rnn_out)
            feat = self.dropout(rnn_out.squeeze(1))  # [B, H]
            
            # Predict each level
            level_logits = []
            level_indices = []
            level_embs = []
            
            for l in range(self.num_levels):
                logits_l = self.output_heads[l](feat)  # [B, K]
                level_logits.append(logits_l)
                
                # Teacher forcing decision (40% of time)
                use_teacher_forcing = torch.rand(B, device=device) < self.teacher_forcing_prob
                
                if self.training:
                    # Sample from predicted logits
                    # pred_idx = torch.argmax(logits_l, dim=-1)
                    # Mix teacher forcing with predictions
                    soft_sample = F.gumbel_softmax(logits_l, tau=0.5, hard=True)  # [B, K]
                    # Teacher forcing decision
                    use_teacher_forcing = torch.rand(B, device=device) < self.teacher_forcing_prob

                    gt_onehot = F.one_hot(target_indices[:, t, l], num_classes=logits_l.size(-1)).float()

                    # Mix teacher forcing and soft sample
                    
                    mixed_emb_input = torch.where(
                        use_teacher_forcing.unsqueeze(-1),  # [B,1] broadcast
                        gt_onehot,
                        soft_sample
                    )  # [B, K]
                    

                    final_idx = torch.argmax(logits_l, dim=-1)
                else:
                    # Inference: always use predictions
                    final_idx = torch.argmax(logits_l, dim=-1)
                
                level_indices.append(final_idx)
                
                if(codebooks == None):
                    emb_l = mixed_emb_input @ self.level_embeddings[l].weight
                else:
                    emb_l = codebooks[l](final_idx)
                level_embs.append(emb_l)
                all_embs.append(emb_l.unsqueeze(1))
            
            current_emb = torch.stack(level_embs, dim=1).sum(dim=1)  # [B, vq_dim]
            emb_history = torch.cat([
                emb_history[:, 1:, :],
                current_emb.unsqueeze(1)
            ], dim=1)
            
            all_logits.append(torch.stack(level_logits, dim=1))  # [B, L, K]
            all_indices.append(torch.stack(level_indices, dim=1))  # [B, L]
            
        embeddings = torch.cat(all_embs, dim=1)
        rnn_outs = torch.cat(rnn_outs, dim=1)  # [B, T, hidden_dim]
        indices = torch.stack(all_indices, dim=1)  # [B, T, L]
        logits = torch.stack(all_logits, dim=1)   # [B, T, L, K]
        
        return indices, logits#, rots


    def _sample_index_from_logits(self, logits_l: torch.Tensor, temperature: float, top_k: int = None) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits_l, dim=-1)

        sample_logits = logits_l / temperature
        if top_k is not None:
            k = min(int(top_k), sample_logits.size(-1))
            if k > 0 and k < sample_logits.size(-1):
                vals, _ = torch.topk(sample_logits, k, dim=-1)
                threshold = vals[:, -1].unsqueeze(-1)
                sample_logits = torch.where(
                    sample_logits >= threshold,
                    sample_logits,
                    torch.full_like(sample_logits, float("-inf")),
                )

        probs = F.softmax(sample_logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


    def _forward_inference(self, ctrl_down: torch.Tensor, seq_len: int, temperature: float, top_k: int = None, codebooks=None, yaw_sin_cos: torch.Tensor = None):
        """Autoregressive inference with windowed context"""
        B, T_ds, H = ctrl_down.shape
        seq_len = min(seq_len, T_ds)
        device = ctrl_down.device

        all_logits = []
        all_indices = []
        rnn_outs = []
        all_embs = []
        emb_history = torch.zeros(B, self.window_size, self.vq_dim, device=device)
        hidden_state = None
        
        for t in range(seq_len):
            if t > 0 and self.state_reset_interval > 0 and t % self.state_reset_interval == 0:
                hidden_state = None
            
            w = min(self.window_size, t+1)
            emb_window = emb_history[:, -w: ,:]
            emb_proj_seq = self.emb_proj(emb_window)
            
            start = max(0, t-w+1)
            ctrl_window = ctrl_down[:, start:t+1]
            rnn_input = torch.cat([ctrl_window,emb_proj_seq],dim=-1)
            
            
            
            rnn_out, next_hidden_state = self.rnn(
                rnn_input, hidden_state if self.use_hidden_state else None
            )
            if self.use_hidden_state:
                hidden_state = next_hidden_state
            rnn_out = rnn_out[:,-1:]
            rnn_outs.append(rnn_out)
            feat = rnn_out.squeeze(1)
            
            # Sample each level
            level_logits = []
            level_indices = []
            level_embs = []
            
            for l in range(self.num_levels):
                logits_l = self.output_heads[l](feat)
                level_logits.append(logits_l)
                
                idx_l = self._sample_index_from_logits(logits_l, temperature=temperature, top_k=top_k)
                
                level_indices.append(idx_l)
                if(codebooks == None):
                    emb_l = self.level_embeddings[l](idx_l)
                else:
                    emb_l = codebooks[l](idx_l)
           
                level_embs.append(emb_l)
                all_embs.append(emb_l.unsqueeze(1))

            
            current_emb = torch.stack(level_embs, dim=1).sum(dim=1)
            emb_history = torch.cat([
                emb_history[:, 1:, :],
                current_emb.unsqueeze(1)
            ], dim=1)
            
            all_logits.append(torch.stack(level_logits, dim=1))
            all_indices.append(torch.stack(level_indices, dim=1))

        embeddings = torch.cat(all_embs, dim=1)
        indices = torch.stack(all_indices, dim=1)
        logits = torch.stack(all_logits, dim=1)
        rnn_outs = torch.cat(rnn_outs, dim=1)

        return indices, logits#, rots


    def forward_teacher_forced(self, ctrl_seq: torch.Tensor, target_indices: torch.Tensor,
                               codebooks: torch.Tensor = None, yaw_sin_cos: torch.Tensor = None):
        """
        Evaluate next-code logits using preceding ground-truth codes, never the current code.
        Requires eval mode and uses inference windowing and hidden-state resets; no dropout, sampling or control augmentation, so the result is deterministic.
        Returns indices [B, T, L] and logits [B, T, L, K].
        """
        if self.training:
            raise RuntimeError("teacher-forced prior evaluation requires eval mode")
        ctrl_down = self._encode_controls(ctrl_seq)
        B, T_ds, _ = ctrl_down.shape
        seq_len = min(T_ds, target_indices.size(1))
        device = ctrl_down.device

        all_logits = []
        all_indices = []
        emb_history = torch.zeros(B, self.window_size, self.vq_dim, device=device)
        hidden_state = None

        for t in range(seq_len):
            if t > 0 and self.state_reset_interval > 0 and t % self.state_reset_interval == 0:
                hidden_state = None

            w = min(self.window_size, t + 1)
            emb_proj_seq = self.emb_proj(emb_history[:, -w:, :])
            start = max(0, t - w + 1)
            rnn_input = torch.cat([ctrl_down[:, start:t + 1], emb_proj_seq], dim=-1)

            rnn_out, next_hidden_state = self.rnn(
                rnn_input, hidden_state if self.use_hidden_state else None
            )
            if self.use_hidden_state:
                hidden_state = next_hidden_state
            feat = rnn_out[:, -1]

            level_logits = []
            level_embs = []
            for l in range(self.num_levels):
                logits_l = self.output_heads[l](feat)
                level_logits.append(logits_l)
                table = self.level_embeddings[l] if codebooks is None else codebooks[l]
                level_embs.append(table(target_indices[:, t, l]))

            all_logits.append(torch.stack(level_logits, dim=1))  # [B, L, K]
            all_indices.append(torch.stack([logits.argmax(dim=-1) for logits in level_logits], dim=1))  # [B, L]

            current_emb = torch.stack(level_embs, dim=1).sum(dim=1)
            emb_history = torch.cat([emb_history[:, 1:, :], current_emb.unsqueeze(1)], dim=1)

        return torch.stack(all_indices, dim=1), torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def sample(self, ctrl_seq: torch.Tensor, temperature: float = 1.0, top_k: int = None,
               codebooks: torch.Tensor = None, yaw_sin_cos: torch.Tensor = None):
        """Convenience sampling method"""
        self.eval()
        indices, logits = self.forward(
            ctrl_seq,
            temperature=temperature,
            top_k=top_k,
            codebooks=codebooks,
            yaw_sin_cos=yaw_sin_cos,
        )
        
        if top_k is not None:
            B, T, L, K = logits.shape
            for b in range(B):
                for t in range(T):
                    for l in range(L):
                        vals, _ = torch.topk(logits[b, t, l], min(top_k, K))
                        threshold = vals[-1]
                        logits[b, t, l] = torch.where(
                            logits[b, t, l] >= threshold,
                            logits[b, t, l],
                            torch.full_like(logits[b, t, l], float('-inf'))
                        )
        
        return indices, logits

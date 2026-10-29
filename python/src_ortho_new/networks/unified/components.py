import torch
import torch.nn as nn
import torch.nn.functional as F


num_layers = 3


dummy_joint = False


attn_dim = 448


plot = 0


def resolve_vae_latent_dim(param):
    return int(param.get("vae_latent_dim", (len(param["sparse_joints"]) + 1) * 9 * 8))


class ConvLayerNorm(nn.Module):
    """
    Apply LayerNorm to Conv1d outputs (input [B, C, T]).
    Behavior: normalize across channel dim per time step (permute -> LayerNorm -> permute back).
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C] -> LayerNorm over channels -> [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        return x.permute(0, 2, 1)


class HeadingConcatAdapter(nn.Module):
    """
    Downsample heading features, concatenate them with decoder latents,
    and project back to the decoder channel size.
    """
    def __init__(self, heading_dim: int, latent_channels: int, stride: int = 8, kernel_size: int = 15, hidden_dim: int = 128):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv1d(heading_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            ConvLayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.project = nn.Sequential(
            nn.Linear(latent_channels + hidden_dim, latent_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(latent_channels, latent_channels),
        )

    def forward(self, latent: torch.Tensor, heading: torch.Tensor = None) -> torch.Tensor:
        if heading is None:
            return latent
        if heading.dim() == 2:
            heading = heading.unsqueeze(-1)
        heading = heading.permute(0, 2, 1).contiguous()
        heading_features = self.downsample(heading)
        if heading_features.size(-1) != latent.size(-1):
            heading_features = F.interpolate(
                heading_features, size=latent.size(-1), mode="linear", align_corners=False
            )
        latent_time = latent.permute(0, 2, 1)
        heading_time = heading_features.permute(0, 2, 1)
        fused = torch.cat([latent_time, heading_time], dim=-1)
        fused = self.project(fused)
        return fused.permute(0, 2, 1)


class TimingEncoder(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=32, latent_dim=32, output_dim=8, num_layers=2, max_len=50000):
        """Encode temporal features into timing latents."""
        super(TimingEncoder, self).__init__()
        self.fc_bottleneck = nn.Sequential(nn.Linear(input_dim, 8),
                                             nn.LeakyReLU(negative_slope=0.2),
                                             nn.Linear(8, hidden_dim))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_layer = nn.Linear(hidden_dim,output_dim)
        self.sigmoid = nn.Sigmoid()
        
        kernel_size = 3
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(0.2),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels=8, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),#nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),#nn.LeakyReLU(0.2),
        )            

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Return (pos_enc, mu, logvar) from an input sequence [B, T, C]."""
        batch_size, seq_len, _ = x.size()
        # We only need the hidden state from the final time step.
        # x = self.fc_bottleneck(x)  # Apply bottleneck to input
        z_seq = self.enc(x.permute(0, 2, 1)).permute(0,2,1)
        out, (h,_) = self.lstm(z_seq[:,:64,:])
        z_seq = h[-1]
        z_seq = self.sigmoid(self.fc_layer(z_seq)) 
        mu = self.fc_mu(out)  # shape: [batch, seq_len, latent_dim]
        logvar = self.fc_logvar(out)   
        z = self.reparameterize(mu, logvar) 
        
        
        

        # Repeat z along the sequence dimension.
        #z = torch.randn(z.shape, device=z.device)
        #z_seq = z.unsqueeze(1).repeat(1, seq_len//8, 1)  # shape: [batch, seq_len, latent_dim]
        #print(z_seq)
        #####z_seq = torch.randn(z_seq.shape, device=z.device)
        #print(z_seq)
        # Pass through positional encoding.
        #pos_enc = self.pos_encoder(z_seq)              # [batch, seq_len, latent_dim]
        #print(mu,torch.exp(0.5*logvar))
        return z_seq, mu, logvar


class TagEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=8, hidden_dim=32, 
                 window_size=64,
                 kernel_size=3,
                 dropout=0.0,
                 sub_window_size=8,
                 waypoints_dim=3):   
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.sub_window_size = sub_window_size
 
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            #nn.Dropout(dropout),
            # Global pooling to collapse window_size to 1
            #nn.AdaptiveAvgPool1d(1),  # [B, hidden_dim, 1]
        )

        self.proj = nn.Linear(hidden_dim, output_dim)

        self.waypoints_net = nn.Sequential(
            nn.Linear(waypoints_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

        self.final_combiner = nn.Sequential(
            nn.Linear(output_dim + output_dim, 8),  # coarse + waypoints output
            nn.LeakyReLU(0.2),
            nn.Linear(8, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x: torch.Tensor, waypoints: torch.Tensor = None) -> torch.Tensor:
        """Encode frame-level tags, optionally combining windowed waypoints."""
        B, T, D = x.shape

        out = self.conv(x.permute(0,2,1))                
        out = out.squeeze(-1)                       
        coarse_out = self.proj(out.permute(0,2,1))            
        
        if waypoints is not None:
         
            waypoints_out = self.waypoints_net(waypoints)  # [B, num_windows, output_dim]
            #print(coarse_out.shape, waypoints_out.shape)
            combined = torch.cat([coarse_out, waypoints_out], dim=-1)  # [B, num_windows, output_dim*2]
            #print(combined.shape)
            # Pass through final combiner network
            coarse_out = self.final_combiner(combined)  # [B, num_windows, output_dim]
        
        return coarse_out    


class LatentGRU(nn.Module):
    def __init__(self, tag_embed_dim, z_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(LatentGRU, self).__init__()
        
        self.tag_embed_dim = tag_embed_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define feedback dimension
        self.feedback_dim = 20
        
        # Combined input dimension including feedback
        self.combined_dim = tag_embed_dim + z_dim + self.feedback_dim
        
        # forward() uses the GRU; the LSTM and MLP submodules are instantiated but unused.
        self.lstm = nn.LSTM(
            input_size=self.combined_dim - z_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.gru = nn.GRU(
        input_size=self.combined_dim - z_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim-z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
 
        # Projection for output
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Projection for feedback (generates the 10-dim vector to feed into next step)
        self.feedback_proj = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(output_dim, self.feedback_dim),
            nn.LeakyReLU(0.2),
        )
        # Displacement prediction network
        self.displacement_net = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3)  # x, y, z displacement
        )

        self.rots_proj = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(2,2)
        )

        
    def forward(self, encoded_tags, z):
        """
        Autoregressively predict (motion_embeddings, displacements, rots) from tags and timing latents.
        A rank-2 z is broadcast over time; feedback starts at zero.
        """
        batch_size, seq_len, _ = encoded_tags.shape
        device = encoded_tags.device
        
        
        if z.dim() == 2:  # shape [batch, z_dim]
            z = z.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, z_dim]

        hidden = None
        
        motion_embeddings = []
        displacements = []
        rots = []
        
        feedback = torch.zeros(batch_size, self.feedback_dim, device=device)
        
        
        for t in range(seq_len):
            current_input = torch.cat([
                encoded_tags[:, t, :],  # Current encoded tags
                feedback                # Feedback from previous step (or zeros for t=0)
            ], dim=-1)                  # [batch, combined_dim]
            
            current_input = current_input.unsqueeze(1)  # [batch, 1, combined_dim]

            output, hidden = self.gru(current_input, hidden)  # output: [batch, 1, hidden_dim]
            output_flat = output.squeeze(1)  # [batch, hidden_dim]
            #output_flat = self.mlp(current_input) 
            motion_embed = self.fc(output_flat)  # [batch, output_dim]
            motion_embeddings.append(motion_embed.unsqueeze(1))
            
            # Predict displacement
            disp = self.displacement_net(motion_embed)  # [batch, 3]
            displacements.append(disp.unsqueeze(1))
            rot_diff = self.rots_proj(motion_embed[:, -2:])  # Assuming last 2 dims are rotations
            rots.append(rot_diff.unsqueeze(1))
            # Generate feedback for next timestep (10-dim vector)
            feedback = self.feedback_proj(motion_embed)  # [batch, feedback_dim]
        
        motion_embeddings = torch.cat(motion_embeddings, dim=1)  # [batch, seq_len, output_dim]
        displacements = torch.cat(displacements, dim=1)  # [batch, seq_len, 3]
        rots = torch.cat(rots,dim=1)
        
        """# Concatenate encoded tags and z
        lstm_input = torch.cat([encoded_tags, z], dim=-1)  # [batch, seq_len, combined_dim]
        
        # Process the entire sequence through the LSTM
        lstm_output, _ = self.lstm(lstm_input)  # lstm_output: [batch, seq_len, hidden_dim]
        
        # Project LSTM output to motion embeddings
        motion_embeddings = self.fc(lstm_output)  # [batch, seq_len, output_dim]
        
        # Predict displacement
        displacements = self.displacement_net(motion_embeddings)  # [batch, seq_len, 3]
        
        # Predict rotations
        rots = self.rots_proj(motion_embeddings[:, :, -2:])  # Assuming last 2 dims are rotations"""
        
        
        return motion_embeddings, displacements, rots


# backward compatibility
LatentLSTM = LatentGRU


class RotsFromWaypoints(nn.Module):
    """Upsample sparse waypoint features to per-frame sine/cosine rotation features."""
    def __init__(self, waypoints_dim=64, output_dim=2, hidden_dim=64, scale_factor=8, kernel_size=3):
        super(RotsFromWaypoints, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(waypoints_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.net_output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initial encoding of waypoints
        self.encoder = nn.Sequential(
            nn.Linear(waypoints_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU()
        )
        
        # Progressive upsampling network
        self.upsampler = nn.ModuleList()
        
        # We need log_2(scale_factor) steps to achieve the full upsampling
        steps = 0
        factor = scale_factor
        while factor > 1:
            factor //= 2
            steps += 1
        
        curr_dim = hidden_dim
        for i in range(steps):
            self.upsampler.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                nn.Conv1d(curr_dim, curr_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.InstanceNorm1d(curr_dim),
                nn.LeakyReLU()
            ))
        
        # Final projection to output dimension
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
        self.scale_factor = scale_factor
        self.output_dim = output_dim
        
    def forward(self, waypoints):
      
        batch_size, num_waypoints, _ = waypoints.shape
        
        encoded = self.encoder(waypoints)  # [batch, num_waypoints, hidden_dim]
        
        x = encoded.permute(0, 2, 1)  # [batch, hidden_dim, num_waypoints]
 
        # Progressive upsampling
        for layer in self.upsampler:
            x = layer(x)
            
        # Final projection to output dimension
        x = self.output_proj(x)  # [batch, output_dim, sequence_length]
        
        # Ensure we have the exact output size in case of rounding errors
        if x.shape[2] != num_waypoints * self.scale_factor:
            x = nn.functional.interpolate(
                x, size=num_waypoints * self.scale_factor, mode='linear', align_corners=False
            )
        
        # Permute back to desired output shape
        x = x.permute(0, 2, 1)  # [batch, sequence_length, output_dim]
 
        
        return x


class DisplacementPredictor(nn.Module):
    """
    Predict per-frame 3D displacement from controls, latent features, and yaw.
    ctrl [B, T, ctrl_dim], emb [B, T_emb, embed_dim], yaw [B, T, yaw_dim] -> disp [B, T, 3].
    """
    def __init__(self,
                 ctrl_dim: int = 6,
                 embed_dim: int = 64,
                 yaw_dim: int = 4,
                 hidden_dim: int = 128,
                 conv_hidden: int = 128,
                 dropout: float = 0.1,
                 kernel_size: int = 7):
        super().__init__()
        # Per-stream projections
        self.ctrl_cnn = nn.Sequential(
            nn.Conv1d(ctrl_dim, hidden_dim//2, stride=8, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
        )
        self.yaw_cnn = nn.Sequential(
            nn.Conv1d(yaw_dim, hidden_dim//2, stride=8, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
        )

        # Temporal conv stack operates on concatenated features
        # pad = (conv_kernel - 1) // 2
        # self.temporal = nn.Sequential(
        #     nn.Conv1d(hidden_dim * 3, conv_hidden, kernel_size=conv_kernel, padding=pad),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(conv_hidden, conv_hidden, kernel_size=conv_kernel, padding=pad),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(conv_hidden, hidden_dim, kernel_size=1),
        #     nn.GELU(),
        # )

        # Head to output displacement
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )

    def forward(self, ctrl: torch.Tensor, emb: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        B, T, _ = ctrl.shape
        device = ctrl.device

        p_ctrl = self.ctrl_cnn(ctrl.permute(0,2,1)).permute(0,2,1)         # [B, T, H]
        p_emb  = self.embed_proj(emb) # [B, T, H]
        p_yaw  = self.yaw_cnn(yaw.permute(0,2,1)).permute(0,2,1)   # [B, T, H]
       
        x = torch.cat([p_ctrl, p_emb, p_yaw], dim=-1)   # [B, T, 3H]
        # Predict displacement
        disp = self.head(x)                             # [B, T, 3]

        return disp

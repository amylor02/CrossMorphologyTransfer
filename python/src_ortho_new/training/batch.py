import torch


class Train_Data:
    def __init__(self, device, param):
        super().__init__()
        self.device = device
        self.param = param
        self.losses = []

    def set_offsets(self, offsets, denorm_offsets):
        self.offsets = offsets  
        self.denorm_offsets = denorm_offsets
        for loss in self.losses:
            loss.set_offsets(denorm_offsets[0])

    def set_rot_order(self, rot_order):
        self.rot_order = rot_order

    def set_motions(self, dqs, displacement, disp_8, sin_diff=None, cos_diff=None, loss_weights=None): 

        DL = displacement[0,1].shape[0] # displacement length (3 or 4)
        downsampling_factor = self.param["stride_encoder_conv"] ** 3

        self.motion = torch.cat([dqs, displacement], dim=2)
        # swap second and third dimensions for convolutions (last row should be time)
        self.motion = self.motion.permute(0, 2, 1)
        self.disp_8 = disp_8

        if(loss_weights is not None):
            self.loss_weights = loss_weights
            
        if(sin_diff is not None):
            self.yaw_diff = torch.cat([sin_diff.unsqueeze(-1), cos_diff.unsqueeze(-1)],dim=-1)

        # self.motion is tensor of shape (batch_size, n_joints*8 + 3, frames)
        # if time dimension is not multiple of 8... make it multiple of 8
        # so that convolutions always return an even number (3 time downsamplings)
        if self.motion.shape[2] % downsampling_factor != 0:
            self.motion = self.motion[
                :, :, : self.motion.shape[2] - self.motion.shape[2] % downsampling_factor
            ]
        # sparse_dqs (batch_size, n_sparse_joints, 8, frames)
        sparse_dqs = (
            self.motion[:, :-DL, :]
            .reshape(self.motion.shape[0], -1, 9, self.motion.shape[-1])
            .clone()
        )
    
        sparse_dqs = sparse_dqs[:, self.param["sparse_joints"], ...]
        sparse_dqs = sparse_dqs.flatten(start_dim=1, end_dim=2)
        
        
        sparse_displacement = self.motion[:, -DL:, :].clone()
        # self.sparse_motion (batch_size, n_sparse_joints * 8 + 3, frames)
        self.sparse_motion = torch.cat([sparse_dqs, sparse_displacement], dim=1)
        expected_sparse_channels = len(self.param["sparse_joints"]) * 9 + DL
        if self.sparse_motion.shape[1] != expected_sparse_channels:
            raise ValueError(
                "sparse_motion has {} channels, expected {} for {} sparse joints plus {} displacement channels".format(
                    self.sparse_motion.shape[1],
                    expected_sparse_channels,
                    len(self.param["sparse_joints"]),
                    DL,
                )
            )
    
        # remove displacement from self.motion (we use it only as input)
        # self.motion is tensor of shape (batch_size, n_joints*8, frames)
        self.displacement = self.motion[:, -DL:, :]
        self.motion = self.motion[:, :-DL, :]
        # adding it back for now actually
        #self.motion[:,4:7] = self.displacement[:,0:3]

    def set_sparse_motion(self, sparse_motion):
        self.sparse_motion = sparse_motion

    def set_means(self, mean_dqs):
        self.mean_dqs = mean_dqs
        for loss in self.losses:
            loss.set_mean(mean_dqs)

    def set_stds(self, std_dqs):
        self.std_dqs = std_dqs
        for loss in self.losses:
            loss.set_std(std_dqs)
            
    def set_root_means_stds(self, mean_root, std_root):
        self.mean_root = mean_root
        self.std_root = std_root
        
    def set_sin_cos_means_stds(self, mean_sin, mean_cos, std_sin, std_cos):
        self.mean_sin = mean_sin
        self.mean_cos = mean_cos
        self.std_sin = std_sin
        self.std_cos = std_cos

    def set_phase(self, phase):
        self.phase = phase

    def set_phase_per_8_frames(self, phase_per_8_frames):
        self.phase_per_8_frames = phase_per_8_frames
        
    def set_velocity_per_8_frames(self, velocity_per_8_frames):
        self.velocity_per_8_frames = velocity_per_8_frames

    def set_tags(self, tags):
        self.tags = tags
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, value in self.tags.items():
            if not isinstance(value, torch.Tensor):
                self.tags[key] = torch.tensor(value,dtype=torch.float32, requires_grad=True).to(device)

            elif(self.tags["velo"].device != device):
                self.tags[key] = value.to(device, dtype=torch.float32).requires_grad_(True)
            
            if(key!="velo_foot"):
                self.tags[key] = self.tags[key].unsqueeze(-1)
        
        
    def set_global_pos(self, global_pos):
        if not isinstance(global_pos, torch.Tensor):
            self.global_pos = torch.tensor(global_pos,requires_grad=True).to(self.device)    
        else:
            self.global_pos = global_pos.to(self.device, dtype=torch.float32)
            
    def set_rots(self, rots):
        if not isinstance(rots, torch.Tensor):
            self.rots = torch.tensor(rots, requires_grad=True).to(self.device)
        else:
            self.rots = rots.to(self.device, dtype=torch.float32)
    
    def set_end_sites(self, end_sites, end_sites_parents):
        if not isinstance(end_sites, torch.Tensor):
            self.end_sites = torch.tensor(end_sites, dtype=torch.float32).to(self.device)
            self.end_sites_parents = torch.as_tensor(end_sites_parents, dtype=torch.long, device=self.device)
        else:
            self.end_sites = end_sites.to(self.device, dtype=torch.float32)
            self.end_sites_parents = end_sites_parents.to(self.device, dtype=torch.long)
    def set_energy(self, energy):
        self.energy = energy

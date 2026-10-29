import numpy as np
import torch
from torch.utils.data import Dataset
import pymotion.rotations.dual_quat as dquat
from pymotion.ops.skeleton import to_root_dual_quat
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d_torch as ortho6d
from utils import network_timing
from motion.signals import USE_GLOBAL_HEIGHT, DOWNSAMPLE_FACTOR, WINDOW_STEP_SIZE, compute_tags
from motion.phase import compute_phase


class TrainMotionData(Dataset):
    def __init__(self, param, scale, device, include_final_window=True):
        # a window that ends exactly at the sequence end is kept unless this is False.
        self.include_final_window = include_final_window
        self.motions = []
        self.norm_motions = []
        self.means_motions = []
        self.var_motions = []
        self.param = param
        self.scale = scale
        self.device = device
        self.rots = []
        self.global_pos = []
        self.downsample_factor = DOWNSAMPLE_FACTOR

    def add_motion(self, offsets, global_pos, rotations, parents, pos=None, og_rots = None, end_sites=None, end_sites_parents=None):
        """Convert one motion into training windows and append them to this dataset."""
        frames = rotations.shape[0]
        real_frames = frames  # frames before padding; padded frames repeat the last real frame
        pad_size = 0
        if frames < self.param["window_size"]:
            pad_size = self.param["window_size"] - frames
            last_global_pos = global_pos[-1]
            last_rotation = rotations[-1]
            last_pos = pos[-1]
            global_pos = np.concatenate([global_pos] + [last_global_pos[np.newaxis, :]] * pad_size, axis=0)
            rotations = np.concatenate([rotations] + [last_rotation[np.newaxis, :, :]] * pad_size, axis=0)
            if(og_rots is not None):
                og_rots = np.concatenate([og_rots] + [og_rots[-1][np.newaxis, :, :]] * pad_size, axis=0)
            pos = np.concatenate([pos] + [last_pos[np.newaxis, :, :]] * pad_size, axis=0)
            frames = self.param["window_size"]

        elif frames % self.param["window_size"] != 0:
            pad_size = self.param["window_size"] - frames % self.param["window_size"]
            last_global_pos = global_pos[-1]
            last_rotation = rotations[-1]
            last_pos = pos[-1]
            global_pos = np.concatenate([global_pos] + [last_global_pos[np.newaxis, :]] * pad_size, axis=0)
            rotations = np.concatenate([rotations] + [last_rotation[np.newaxis, :, :]] * pad_size, axis=0)
            if(og_rots is not None):
                og_rots = np.concatenate([og_rots] + [og_rots[-1][np.newaxis, :, :]] * pad_size, axis=0)
            pos = np.concatenate([pos] + [last_pos[np.newaxis, :, :]] * pad_size, axis=0)
            frames += pad_size

        assert frames >= self.param["window_size"]
        
        #because human has tpose at initial frame
        rotations[0] = rotations[1]
        global_pos[0] = global_pos[1]
        pos[0] = pos[1]

        self.rots = rotations.copy()
        
        
        
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = ortho6d.from_dual_quat(dqs) # <-------------------------------------------------------------
        dqs = torch.flatten(dqs, 1, 2)
        # offsets
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        self.global_pos = global_pos.clone()
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )

        if(USE_GLOBAL_HEIGHT):
            new_displacement = torch.zeros((displacement.shape[0], 4), device='cuda:0')
            for frame in range(displacement.shape[0]):
                new_displacement[frame] = torch.cat((displacement[frame], global_pos[frame, 1].unsqueeze(0)))
            displacement = new_displacement

        # displacement[torch.abs(displacement) < 0.01] = 0 <-----
        #displacement[:,1]=displacement[:,3]
        #dqs[:,4:7] = displacement[:,0:3].clone()
        # dqs[:,6:9] = global_pos.clone()
        dqs[:,6:9] = torch.cat([displacement[:,0:1], global_pos[:,1:2], displacement[:,2:3]], dim=1) # displacement[:,0:3].clone()

  
        if(og_rots is not None):
            rotations = og_rots
        
        # displacement[:,1] = global_pos[:,1]
        rots = torch.from_numpy(rotations).type(torch.float32).to(self.device)
        self.rots = ortho6d.from_quat(rots[:,0]).reshape(-1, 6)
        displacement[:,1] = displacement[:,3] # use y pos instead of y displacement
        self.rots = torch.cat([self.rots, global_pos], dim=-1)
        rots = rots.flatten(start_dim=1, end_dim=2).permute(1, 0).unsqueeze(0)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        phase, velocity_xz = compute_phase(pos, joint_id=[18])
        rot_order = np.tile(['y','x','z'], (rotations.shape[0], rotations.shape[1], 1))

        rots_euler = quat.to_euler(rotations, rot_order)

        tags = compute_tags(pos, downsample_factor=self.downsample_factor,rots=rots_euler[:,0], is_deg = False,
                            quats=rotations, padded_frames = pad_size, skeleton_height=self.param["skeleton_height"]
                            , head_height=self.param["head_height"], head_idx=self.param["head_idx"], feet_idx=self.param["feet_idxs"]
                            , not_dog=self.param["not_dog"], feet_contact_threshold=self.param["feet_contact_threshold"]
                            , yaw_error=self.param.get("yaw_error", 0.0))

        for key in tags:
            if not isinstance(tags[key], torch.Tensor):
                tags[key] = torch.tensor(tags[key]).float().to(self.device)

        slope = np.diff(phase, axis=1)  # Difference between consecutive points
        slope = np.pad(slope, ((0, 0), (0, 1)), mode='edge')  # Pad to maintain the same shape
        slope = np.sign(slope)
        phase = phase * slope

        # get avg phase and sum velocity (per 8 frames)
        num_frames = phase.shape[1]
        num_groups = num_frames // 8

        # Reshape phase to [num_joints, num_groups, 8] and compute the average
        phase_reshaped = phase[:, :num_groups * 8].reshape(phase.shape[0], num_groups, 8)
        phase_avg = np.mean(phase_reshaped, axis=2) # take the avg in the 8 frame windows

        # but if window contains 1 (peak) or -1 (valley) set the avg to 1 or -1
        set_next_to_peak = False
        set_next_to_valley = False
        # there has to be a more elegant way to do this <-----------------------------------
        for i in range(phase_reshaped.shape[0]):  # Iterate over joints
            for j in range(phase_reshaped.shape[1]):  # Iterate over groups of 8 frames
                if np.any(phase_reshaped[i, j, :] > 0.96) or set_next_to_peak:
                    set_next_to_peak = False # reset flag
                    phase_avg[i, j] = 1
                    if np.any(phase_reshaped[i, j, :] < -0.96): # sometimes peak and valley are in the same group
                        set_next_to_valley = True

                elif np.any(phase_reshaped[i, j, :] < -0.96) or set_next_to_valley:
                    set_next_to_valley = False
                    phase_avg[i, j] = -1
                    if np.any(phase_reshaped[i, j, :] > 0.96):
                        set_next_to_peak = True

        possible_values = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

        distances = np.abs(phase_avg[..., np.newaxis] - possible_values)

        # Find the index of the closest value
        closest_idx = np.argmin(distances, axis=-1)

        # Map to the closest value
        discrete_phase_every_8_frames = possible_values[closest_idx]
        velocity_xz_reshaped = velocity_xz[:num_groups * 8].reshape(num_groups, 8)
        velocity_every_8_frames = np.sum(velocity_xz_reshaped, axis=1)
        for start in range(0, frames, self.param["window_step"]):
            end = start + self.param["window_size"]

            tags_step = WINDOW_STEP_SIZE
            waypoints_step = 8

            fits = end <= frames if self.include_final_window else end < frames
            if fits:
                motion = {
                    "offsets": offsets,
                    "dqs": dqs[start:end],
                    "displacement": displacement[start:end],
                    "disp_8": tags["disp_8"][start//DOWNSAMPLE_FACTOR:end//DOWNSAMPLE_FACTOR],
                    "phase": phase[:,start:end],
                    "phase_per_8_frames": discrete_phase_every_8_frames[:, start//8 : end//8],
                    "velocity_per_8_frames": velocity_every_8_frames[start//8 : end//8],
                    "loss_weights": tags["loss_weights"][start:end],
                    # real (unpadded) frames in this window; used to leave padding out of the codebook initialisation
                    "valid_frames": int(min(max(real_frames - start, 0), self.param["window_size"])),
                    "end_sites": end_sites,
                    "end_sites_parents": np.array(end_sites_parents),
                    "tags": {
                        "velocity": tags["velocity"][start//tags_step : end//tags_step],
                        "acceleration": tags["acceleration"][start//tags_step : end//tags_step],
                        "ang_velocity": tags["ang_velocity"][start//tags_step : end//tags_step],
                        "height": tags["height"][start//tags_step : end//tags_step],
                        "velocity_full": tags["velocity_full"][start:end],
                        "velo": tags["velo"][start:end],
                        "accel": tags["accel"][start:end],
                        "dx": tags["dx"][start:end],
                        "dz": tags["dz"][start:end],
                        "height_full": tags["height_full"][start:end],
                        "alignment": tags["alignment"][start:end],
                        "root_rots": tags["root_rots"][start:end],
                        "direction": tags["direction"][start:end],
                        "velo_foot": tags["velo_foot"][start:end],
                        "head_height": tags["head_height"][start:end],
                        "amps": tags["amps"][start//waypoints_step:end//waypoints_step], # waypoints
                        "sin": tags["sin"][start//waypoints_step:end//waypoints_step],
                        "cos": tags["cos"][start//waypoints_step:end//waypoints_step],
                        "foot_labels": tags["foot_labels"][start:end],
                        "yaw_sin": tags["yaw_sin"][start:end],
                        "yaw_cos": tags["yaw_cos"][start:end],
                        "yaw_quats": tags["yaw_quats"][start:end],
                        "sin_diff": tags["sin_diff"][start//8:end//8],
                        "cos_diff": tags["cos_diff"][start//8:end//8],
                        "ctrl_velocity": tags["ctrl_velocity"][start:end],
                        "ctrl_acceleration": tags["ctrl_acceleration"][start:end],
                        "ctrl_yaw_rate": tags["ctrl_yaw_rate"][start:end],
                        "ctrl_yaw_accel": tags["ctrl_yaw_accel"][start:end],
                        "ctrl_height": tags["ctrl_height"][start:end],
                        "ctrl_head_height": tags["ctrl_head_height"][start:end],
                        "ctrl_roll": tags["ctrl_roll"][start:end],
                        "ctrl_foot_labels": tags["ctrl_foot_labels"][start:end],
                        "foot_contact": tags["foot_contact"][start:end],
                        "foot_contact_valid": tags["foot_contact_valid"][start:end],
                        "ctrl_vertical_velocity": tags["ctrl_vertical_velocity"][start:end],
                        "smooth_root_pos": tags["smooth_root_pos"][start:end]
                    },
                    "energy_feet": tags["energy_feet"][start//tags_step : end//tags_step],
                    "global_pos": self.global_pos[start:end],
                    "rots": self.rots[start:end],
                    
                }
                self.motions.append(motion)
            

        velo = tags["velo"].clone()
        dx = tags["dx"].clone()
        dz = tags["dz"].clone()
        height_full = tags["height_full"].clone()
        alignment = tags["alignment"].clone()
        root_rots = tags["root_rots"].clone()
        disp_8 = tags["disp_8"].clone()
        direction = tags["direction"].clone()
        
        # Means
        motion_mean = {
            "offsets": torch.mean(offsets, dim=0).to(self.device),
            "dqs": torch.mean(dqs, dim=0).to(self.device),
            "displacement": torch.mean(displacement, dim=0).to(self.device),
            "disp_8": torch.mean(disp_8, dim=0).to(self.device),
            "velo": torch.mean(velo, dim=0).to(self.device),
            "dx": torch.mean(dx, dim=0).to(self.device),
            "dz": torch.mean(dz, dim=0).to(self.device),
            "height_full": torch.mean(height_full, dim=0).to(self.device),
            "alignment": torch.mean(alignment, dim=0).to(self.device),
            "root_rots": torch.mean(root_rots, dim=0).to(self.device),
            "direction": torch.mean(direction, dim=0).to(self.device),
            "head_height": torch.mean(tags["head_height"], dim=0).to(self.device),
            "amps": torch.mean(tags["amps"], dim=0).to(self.device),
            "sin": torch.mean(tags["sin"], dim=0).to(self.device),
            "cos": torch.mean(tags["cos"], dim=0).to(self.device),
            "foot_labels": torch.mean(tags["foot_labels"], dim=0).to(self.device),
            "yaw_sin": torch.mean(tags["yaw_sin"], dim=0).to(self.device),
            "yaw_cos": torch.mean(tags["yaw_cos"], dim=0).to(self.device),
            "yaw_quats": torch.mean(tags["yaw_quats"], dim=0).to(self.device),
            "sin_diff": torch.mean(tags["sin_diff"], dim=0).to(self.device),
            "cos_diff": torch.mean(tags["cos_diff"], dim=0).to(self.device),
            "ctrl_velocity": torch.mean(tags["ctrl_velocity"], dim=0).to(self.device),
            "ctrl_acceleration": torch.mean(tags["ctrl_acceleration"], dim=0).to(self.device),
            "ctrl_yaw_rate": torch.mean(tags["ctrl_yaw_rate"], dim=0).to(self.device),
            "ctrl_yaw_accel": torch.mean(tags["ctrl_yaw_accel"], dim=0).to(self.device),
            "ctrl_height": torch.mean(tags["ctrl_height"], dim=0).to(self.device),
            "ctrl_head_height": torch.mean(tags["ctrl_head_height"], dim=0).to(self.device),
            "ctrl_roll": torch.mean(tags["ctrl_roll"], dim=0).to(self.device),
            "ctrl_vertical_velocity": torch.mean(tags["ctrl_vertical_velocity"], dim=0).to(self.device),
            "rots": torch.mean(self.rots, dim=0).to(self.device),
            "smooth_root_pos": torch.mean(tags["smooth_root_pos"], dim=0).to(self.device),
        }
        self.means_motions.append(motion_mean)
        # Stds
        motion_var = {
            "offsets": torch.var(offsets, dim=0).to(self.device),
            "dqs": torch.var(dqs, dim=0).to(self.device),
            "displacement": torch.var(displacement, dim=0).to(self.device),
            "disp_8": torch.var(disp_8, dim=0).to(self.device),
            "velo": torch.var(velo, dim=0).to(self.device),
            "dx": torch.var(dx, dim=0).to(self.device),
            "dz": torch.var(dz, dim=0).to(self.device),
            "height_full": torch.var(height_full, dim=0).to(self.device),
            "alignment": torch.var(alignment, dim=0).to(self.device),
            "root_rots": torch.var(root_rots, dim=0).to(self.device),
            "direction": torch.var(direction, dim=0).to(self.device),
            "head_height": torch.var(tags["head_height"], dim=0).to(self.device),
            "amps": torch.var(tags["amps"], dim=0).to(self.device),
            "sin": torch.var(tags["sin"], dim=0).to(self.device),
            "cos": torch.var(tags["cos"], dim=0).to(self.device),
            "foot_labels": torch.var(tags["foot_labels"], dim=0).to(self.device),
            "yaw_sin": torch.var(tags["yaw_sin"], dim=0).to(self.device),
            "yaw_cos": torch.var(tags["yaw_cos"], dim=0).to(self.device),
            "yaw_quats": torch.var(tags["yaw_quats"], dim=0).to(self.device),
            "sin_diff": torch.var(tags["sin_diff"], dim=0).to(self.device),
            "cos_diff": torch.var(tags["cos_diff"], dim=0).to(self.device),
            "ctrl_velocity": torch.var(tags["ctrl_velocity"], dim=0).to(self.device),
            "ctrl_acceleration": torch.var(tags["ctrl_acceleration"], dim=0).to(self.device),
            "ctrl_yaw_rate": torch.var(tags["ctrl_yaw_rate"], dim=0).to(self.device),
            "ctrl_yaw_accel": torch.var(tags["ctrl_yaw_accel"], dim=0).to(self.device),
            "ctrl_height": torch.var(tags["ctrl_height"], dim=0).to(self.device),
            "ctrl_head_height": torch.var(tags["ctrl_head_height"], dim=0).to(self.device),
            "ctrl_roll": torch.var(tags["ctrl_roll"], dim=0).to(self.device),
            "ctrl_vertical_velocity": torch.var(tags["ctrl_vertical_velocity"], dim=0).to(self.device),
            "rots": torch.var(self.rots, dim=0).to(self.device),
            "smooth_root_pos": torch.var(tags["smooth_root_pos"], dim=0).to(self.device),
        }
        self.var_motions.append(motion_var)

    def normalize(self):
        """Compute normalization statistics and populate the normalized motion windows."""
        offsets_means = torch.stack([m["offsets"] for m in self.means_motions], dim=0)
        offsets_vars = torch.stack([s["offsets"] for s in self.var_motions], dim=0)
        dqs_means = torch.stack([m["dqs"] for m in self.means_motions], dim=0)
        dqs_vars = torch.stack([s["dqs"] for s in self.var_motions], dim=0)

        velo_means = torch.stack([m["velo"] for m in self.means_motions], dim=0)
        dx_means = torch.stack([m["dx"] for m in self.means_motions], dim=0)
        dz_means = torch.stack([m["dz"] for m in self.means_motions], dim=0)
        height_full_means = torch.stack([m["height_full"] for m in self.means_motions], dim=0)
        alignment_means = torch.stack([m["alignment"] for m in self.means_motions], dim=0)
        root_rots_means = torch.stack([m["root_rots"] for m in self.means_motions], dim=0)
        direction_means = torch.stack([m["direction"] for m in self.means_motions], dim=0)
        head_height_means = torch.stack([m["head_height"] for m in self.means_motions], dim=0)
        foot_labels_means = torch.stack([m["foot_labels"] for m in self.means_motions], dim=0)

        velo_vars = torch.stack([s["velo"] for s in self.var_motions], dim=0)
        dx_vars = torch.stack([s["dx"] for s in self.var_motions], dim=0)
        dz_vars = torch.stack([s["dz"] for s in self.var_motions], dim=0)
        height_full_vars = torch.stack([s["height_full"] for s in self.var_motions], dim=0)
        alignment_vars = torch.stack([s["alignment"] for s in self.var_motions], dim=0)
        root_rots_vars = torch.stack([s["root_rots"] for s in self.var_motions], dim=0)
        direction_vars = torch.stack([s["direction"] for s in self.var_motions], dim=0)
        head_height_vars = torch.stack([s["head_height"] for s in self.var_motions], dim=0)
        foot_labels_vars = torch.stack([s["foot_labels"] for s in self.var_motions], dim=0)
        
        rots_means = torch.stack([m["rots"] for m in self.means_motions], dim=0)
        rots_vars = torch.stack([s["rots"] for s in self.var_motions], dim=0)


        displacement_means = torch.stack(
            [m["displacement"] for m in self.means_motions], dim=0
        )
        displacement_vars = torch.stack(
            [s["displacement"] for s in self.var_motions], dim=0
        )

        disp_8_means = torch.stack(
            [m["disp_8"] for m in self.means_motions], dim=0
        )
        disp_8_vars = torch.stack(
            [s["disp_8"] for s in self.var_motions], dim=0
        )

        self.means = {
            "offsets": torch.mean(offsets_means, dim=0).to(self.device),
            "dqs": torch.mean(dqs_means, dim=0).to(self.device),
            "displacement": torch.mean(displacement_means, dim=0).to(self.device),
            "disp_8": torch.mean(disp_8_means, dim=0).to(self.device),
            "velo": torch.mean(velo_means, dim=0).to(self.device),
            "dx": torch.mean(dx_means, dim=0).to(self.device),
            "dz": torch.mean(dz_means, dim=0).to(self.device),
            "height_full": torch.mean(height_full_means, dim=0).to(self.device),
            "alignment": torch.mean(alignment_means, dim=0).to(self.device),
            "root_rots": torch.mean(root_rots_means, dim=0).to(self.device),
            "direction": torch.mean(direction_means, dim=0).to(self.device),
            "head_height": torch.mean(head_height_means, dim=0).to(self.device),
            "amps": torch.mean(torch.stack([m["amps"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "sin": torch.mean(torch.stack([m["sin"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "cos": torch.mean(torch.stack([m["cos"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "foot_labels": torch.mean(foot_labels_means, dim=0).to(self.device),
            "yaw_sin": torch.mean(torch.stack([m["yaw_sin"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "yaw_cos": torch.mean(torch.stack([m["yaw_cos"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "yaw_quats": torch.mean(torch.stack([m["yaw_quats"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "sin_diff": torch.mean(torch.stack([m["sin_diff"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "cos_diff": torch.mean(torch.stack([m["cos_diff"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_velocity": torch.mean(torch.stack([m["ctrl_velocity"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_acceleration": torch.mean(torch.stack([m["ctrl_acceleration"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_yaw_rate": torch.mean(torch.stack([m["ctrl_yaw_rate"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_yaw_accel": torch.mean(torch.stack([m["ctrl_yaw_accel"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_height": torch.mean(torch.stack([m["ctrl_height"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_head_height": torch.mean(torch.stack([m["ctrl_head_height"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_roll": torch.mean(torch.stack([m["ctrl_roll"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "ctrl_vertical_velocity": torch.mean(torch.stack([m["ctrl_vertical_velocity"] for m in self.means_motions], dim=0), dim=0).to(self.device),
            "rots": torch.mean(rots_means, dim=0).to(self.device),
            "smooth_root_pos": torch.mean(torch.stack([m["smooth_root_pos"] for m in self.means_motions], dim=0), dim=0).to(self.device),
        }
        

        # Source: https://stats.stackexchange.com/a/26647
        self.stds = {
            "offsets": torch.sqrt(torch.mean(offsets_vars, dim=0)).to(self.device),
            "dqs": torch.sqrt(torch.mean(dqs_vars, dim=0)).to(self.device),
            "displacement": torch.sqrt(torch.mean(displacement_vars, dim=0)).to(
                self.device
            ),
            "disp_8": torch.sqrt(torch.mean(disp_8_vars, dim=0)).to(self.device),
            "velo": torch.sqrt(torch.mean(velo_vars, dim=0)).to(self.device),
            "dx": torch.sqrt(torch.mean(dx_vars, dim=0)).to(self.device),
            "dz": torch.sqrt(torch.mean(dz_vars, dim=0)).to(self.device),
            "height_full": torch.sqrt(torch.mean(height_full_vars, dim=0)).to(self.device),
            "alignment": torch.sqrt(torch.mean(alignment_vars, dim=0)).to(self.device),
            "root_rots": torch.sqrt(torch.mean(root_rots_vars, dim=0)).to(self.device),
            "direction": torch.sqrt(torch.mean(direction_vars, dim=0)).to(self.device),
            "head_height": torch.sqrt(torch.mean(head_height_vars, dim=0)).to(self.device),
            "amps": torch.sqrt(torch.mean(torch.stack([s["amps"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "sin": torch.sqrt(torch.mean(torch.stack([s["sin"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "cos": torch.sqrt(torch.mean(torch.stack([s["cos"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "foot_labels": torch.sqrt(torch.mean(foot_labels_vars, dim=0)).to(self.device),
            "yaw_sin": torch.sqrt(torch.mean(torch.stack([s["yaw_sin"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "yaw_cos": torch.sqrt(torch.mean(torch.stack([s["yaw_cos"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "yaw_quats": torch.sqrt(torch.mean(torch.stack([s["yaw_quats"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "sin_diff": torch.sqrt(torch.mean(torch.stack([s["sin_diff"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "cos_diff": torch.sqrt(torch.mean(torch.stack([s["cos_diff"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_velocity": torch.sqrt(torch.mean(torch.stack([s["ctrl_velocity"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_acceleration": torch.sqrt(torch.mean(torch.stack([s["ctrl_acceleration"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_yaw_rate": torch.sqrt(torch.mean(torch.stack([s["ctrl_yaw_rate"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_yaw_accel": torch.sqrt(torch.mean(torch.stack([s["ctrl_yaw_accel"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_height": torch.sqrt(torch.mean(torch.stack([s["ctrl_height"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_head_height": torch.sqrt(torch.mean(torch.stack([s["ctrl_head_height"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_roll": torch.sqrt(torch.mean(torch.stack([s["ctrl_roll"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "ctrl_vertical_velocity": torch.sqrt(torch.mean(torch.stack([s["ctrl_vertical_velocity"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
            "rots": torch.sqrt(torch.mean(rots_vars, dim=0)).to(self.device),
            "smooth_root_pos": torch.sqrt(torch.mean(torch.stack([s["smooth_root_pos"] for s in self.var_motions], dim=0), dim=0)).to(self.device),
        }
        
        if self.param.get("floor_root_rotation_std", False):
            root_rotation_std_floor = float(self.param.get("root_rotation_std_floor", 0.1))
            self.stds["dqs"][:6].clamp_(min=root_rotation_std_floor)
            self.stds["rots"][:6].clamp_(min=root_rotation_std_floor)


        for key in self.stds:
            if torch.count_nonzero(self.stds[key]) != torch.numel(self.stds[key]) or (self.stds[key] < 1e-6).any():
                print(f"WARNING: {key} stds are zero")
                self.stds[key][self.stds[key] < 1e-6] = 1
            


        # Normalized
        for motion in self.motions:
            norm_motion = {
                "offsets": (motion["offsets"] - self.means["offsets"])
                / self.stds["offsets"],
                "dqs": (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"],
                "displacement": (motion["displacement"] - self.means["displacement"])
                / self.stds["displacement"],
                "disp_8": (motion["disp_8"] - self.means["disp_8"]) / self.stds["disp_8"],
                "phase": motion["phase"],
                "loss_weights": motion["loss_weights"],
                "tags": {
                    "velo": (motion["tags"]["velo"] - self.means["velo"]) / self.stds["velo"],
                    "dx": (motion["tags"]["dx"] - self.means["dx"]) / self.stds["dx"],
                    "dz": (motion["tags"]["dz"] - self.means["dz"]) / self.stds["dz"],
                    "height_full": (motion["tags"]["height_full"] - self.means["height_full"]) / self.stds["height_full"],
                    "alignment": (motion["tags"]["alignment"] - self.means["alignment"]) / self.stds["alignment"],
                    "root_rots": (motion["tags"]["root_rots"] - self.means["root_rots"]) / self.stds["root_rots"],
                    "direction": (motion["tags"]["direction"] - self.means["direction"]) / self.stds["direction"],
                    "velo_foot": motion["tags"]["velo_foot"],
                    "head_height": (motion["tags"]["head_height"] - self.means["head_height"]) / self.stds["head_height"],
                    "amps": (motion["tags"]["amps"] - self.means["amps"]) / self.stds["amps"],
                    "sin": (motion["tags"]["sin"] - self.means["sin"]) / self.stds["sin"],
                    "cos": (motion["tags"]["cos"] - self.means["cos"]) / self.stds["cos"],
                    "foot_labels": (motion["tags"]["foot_labels"] - self.means["foot_labels"]) / self.stds["foot_labels"],
                    "yaw_sin": (motion["tags"]["yaw_sin"] - self.means["yaw_sin"]) / self.stds["yaw_sin"],
                    "yaw_cos": (motion["tags"]["yaw_cos"] - self.means["yaw_cos"]) / self.stds["yaw_cos"],
                    "yaw_quats": (motion["tags"]["yaw_quats"] - self.means["yaw_quats"]) / self.stds["yaw_quats"],
                    "sin_diff": (motion["tags"]["sin_diff"] - self.means["sin_diff"]) / self.stds["sin_diff"],
                    "cos_diff": (motion["tags"]["cos_diff"] - self.means["cos_diff"]) / self.stds["cos_diff"],
                    "ctrl_velocity": (motion["tags"]["ctrl_velocity"]- self.means["ctrl_velocity"]) / self.stds["ctrl_velocity"],
                    "ctrl_acceleration": (motion["tags"]["ctrl_acceleration"]- self.means["ctrl_acceleration"]) / self.stds["ctrl_acceleration"],
                    "ctrl_yaw_accel": (motion["tags"]["ctrl_yaw_accel"]- self.means["ctrl_yaw_accel"]) / self.stds["ctrl_yaw_accel"],
                    "ctrl_yaw_rate": (motion["tags"]["ctrl_yaw_rate"]- self.means["ctrl_yaw_rate"]) / self.stds["ctrl_yaw_rate"],
                    "ctrl_height": (motion["tags"]["ctrl_height"] - self.means["ctrl_height"]) / self.stds["ctrl_height"],
                    "ctrl_head_height": (motion["tags"]["ctrl_head_height"] - self.means["ctrl_head_height"]) / self.stds["ctrl_head_height"],
                    "ctrl_roll": (motion["tags"]["ctrl_roll"] - self.means["ctrl_roll"]) / self.stds["ctrl_roll"],
                    "ctrl_foot_labels": (motion["tags"]["ctrl_foot_labels"]),
                    "foot_contact": motion["tags"]["foot_contact"],
                    "foot_contact_valid": motion["tags"]["foot_contact_valid"],
                    "ctrl_vertical_velocity": (motion["tags"]["ctrl_vertical_velocity"] - self.means["ctrl_vertical_velocity"]) / self.stds["ctrl_vertical_velocity"],
                    "smooth_root_pos": (motion["tags"]["smooth_root_pos"] - self.means["smooth_root_pos"]) / self.stds["smooth_root_pos"],

                },
                "rots": (motion["rots"]-self.means["rots"]) / self.stds["rots"],
            }
            self.norm_motions.append(norm_motion)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        return (self.motions[index], self.norm_motions[index])


class TestMotionData:
    def __init__(self, param, scale, device):
        self.norm_motions = []
        self.bvhs = []
        self.filenames = []
        self.param = param
        self.scale = scale
        self.device = device
        self.downsample_factor = DOWNSAMPLE_FACTOR

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def add_motion(self, offsets, global_pos, rotations, parents, bvh, filename, pos=None, og_rots = None, end_sites=None, end_sites_parents=None):
        """Append a BVH motion to the evaluation dataset, padding it to a multiple of window_size by repeating the last frame."""
        timing_start = network_timing.start_data_timer()
        frames = rotations.shape[0]
        pad_size = 0
        if frames % self.param["window_size"] != 0:
            pad_size = self.param["window_size"] - frames % self.param["window_size"]
            last_global_pos = global_pos[-1]
            last_rotation = rotations[-1]
            last_pos = pos[-1]
            global_pos = np.concatenate([global_pos] + [last_global_pos[np.newaxis, :]] * pad_size, axis=0)
            rotations = np.concatenate([rotations] + [last_rotation[np.newaxis, :, :]] * pad_size, axis=0)
            if(og_rots is not None):
                og_rots = np.concatenate([og_rots] + [og_rots[-1][np.newaxis, :, :]] * pad_size, axis=0)
            pos = np.concatenate([pos] + [last_pos[np.newaxis, :, :]] * pad_size, axis=0)
            frames += pad_size
            
        #because human has tpose at initial frame
        rotations[0] = rotations[1]
        global_pos[0] = global_pos[1]
        pos[0] = pos[1]

        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = ortho6d.from_dual_quat(dqs)
        dqs = torch.flatten(dqs, 1, 2)
        # offsets
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        self.global_pos = global_pos
        self.pos_all_joints = pos
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        
        if(USE_GLOBAL_HEIGHT):
            new_displacement = torch.zeros((displacement.shape[0], 4), device='cuda:0')
            for frame in range(displacement.shape[0]):
                new_displacement[frame] = torch.cat((displacement[frame], global_pos[frame, 1].unsqueeze(0)))
            displacement = new_displacement



        if(og_rots is not None):
            rotations = og_rots
        
        dqs[:,6:9] = torch.cat([displacement[:,0:1], global_pos[:,1:2], displacement[:,2:3]], dim=1) #displacement[:,0:3].clone()
        #displacement = displacement * 100
        rots = torch.from_numpy(rotations).type(torch.float32).to(self.device)
        self.rots = ortho6d.from_quat(rots[:,0]).reshape(-1, 6)
        displacement[:,1] = displacement[:,3] # use y pos instead of y displacement

        self.rots = torch.cat([self.rots, global_pos], dim=-1)

        rots = rots.flatten(start_dim=1, end_dim=2).permute(1, 0).unsqueeze(0)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        phase, velocity_xz = compute_phase(pos, joint_id=[18])
        rot_order = np.tile(['y','x','z'], (rotations.shape[0], rotations.shape[1], 1))
        rots_euler = quat.to_euler(rotations, rot_order)
        tags = compute_tags(pos, downsample_factor=self.downsample_factor, rots= rots_euler[:,0], is_deg = False,
                            quats=rotations, padded_frames = pad_size,skeleton_height=self.param["skeleton_height"],
                            head_height=self.param["head_height"], head_idx=self.param["head_idx"], feet_idx=self.param["feet_idxs"],
                            not_dog=self.param["not_dog"],feet_contact_threshold=self.param["feet_contact_threshold"],
                            yaw_error=self.param.get("yaw_error", 0.0))

        for key in tags:
            if not isinstance(tags[key], torch.Tensor):
                tags[key] = torch.tensor(tags[key]).float().to(self.device)

        slope = np.diff(phase, axis=1)  # Difference between consecutive points
        slope = np.pad(slope, ((0, 0), (0, 1)), mode='edge')  # Pad to maintain the same shape
        slope = np.sign(slope)
        phase = phase * slope

        # get avg phase and sum velocity (per 8 frames)
        num_frames = phase.shape[1]
        num_groups = num_frames // 8

        # Reshape phase to [num_joints, num_groups, 8] and compute the average
        phase_reshaped = phase[:, :num_groups * 8].reshape(phase.shape[0], num_groups, 8)
        phase_avg = np.mean(phase_reshaped, axis=2) # take the avg in the 8 frame windows

        # but if window contains 1 (peak) or -1 (valley) set the avg to 1 or -1
        set_next_to_peak = False
        set_next_to_valley = False
        # there has to be a more elegant way to do this <-----------------------------------
        for i in range(phase_reshaped.shape[0]):  # Iterate over joints
            for j in range(phase_reshaped.shape[1]):  # Iterate over groups of 8 frames
                if np.any(phase_reshaped[i, j, :] > 0.96) or set_next_to_peak:
                    set_next_to_peak = False # reset flag
                    phase_avg[i, j] = 1
                    if np.any(phase_reshaped[i, j, :] < -0.96): # sometimes peak and valley are in the same group
                        set_next_to_valley = True

                elif np.any(phase_reshaped[i, j, :] < -0.96) or set_next_to_valley:
                    set_next_to_valley = False
                    phase_avg[i, j] = -1
                    if np.any(phase_reshaped[i, j, :] > 0.96):
                        set_next_to_peak = True

        possible_values = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

        distances = np.abs(phase_avg[..., np.newaxis] - possible_values)

        # Find the index of the closest value
        closest_idx = np.argmin(distances, axis=-1)

        # Map to the closest value
        discrete_phase_every_8_frames = possible_values[closest_idx]

        velocity_xz_reshaped = velocity_xz[:num_groups * 8].reshape(num_groups, 8)
        velocity_every_8_frames = np.sum(velocity_xz_reshaped, axis=1)
        motion = {
            "offsets": offsets,
            "denorm_offsets": offsets.clone(),
            "dqs": dqs,
            "displacement": displacement,
            "disp_8": tags["disp_8"],
            "phase" : phase,
            "phase_per_8_frames": discrete_phase_every_8_frames,
            "velocity_per_8_frames": velocity_every_8_frames,
            "loss_weights": tags["loss_weights"],
            "end_sites": end_sites,
            "end_sites_parents": np.array(end_sites_parents),
            "tags": {
                        "velocity": tags["velocity"],
                        "acceleration": tags["acceleration"],
                        "ang_velocity": tags["ang_velocity"],
                        "height": tags["height"],
                        "velocity_full": tags["velocity_full"],  
                        "velo": tags["velo"],
                        "accel": tags["accel"],
                        "dx": tags["dx"],
                        "dz": tags["dz"],    
                        "height_full": tags["height_full"],    
                        "alignment":tags["alignment"],
                        "root_rots": tags["root_rots"],     
                        "direction": tags["direction"],   
                        "velo_foot": tags["velo_foot"],
                        "head_height": tags["head_height"],
                        "amps": tags["amps"],
                        "sin": tags["sin"],
                        "cos": tags["cos"],
                        "foot_labels": tags["foot_labels"],
                        "yaw_sin": tags["yaw_sin"],
                        "yaw_cos": tags["yaw_cos"],
                        "yaw_quats": tags["yaw_quats"],
                        "sin_diff": tags["sin_diff"],
                        "cos_diff": tags["cos_diff"],
                        "ctrl_velocity": tags["ctrl_velocity"],
                        "ctrl_acceleration": tags["ctrl_acceleration"],
                        "ctrl_yaw_rate": tags["ctrl_yaw_rate"],
                        "ctrl_yaw_accel": tags["ctrl_yaw_accel"],
                        "ctrl_height": tags["ctrl_height"],
                        "ctrl_head_height": tags["ctrl_head_height"],
                        "ctrl_roll": tags["ctrl_roll"],
                        "ctrl_foot_labels": tags["ctrl_foot_labels"],
                        "foot_contact": tags["foot_contact"],
                        "foot_contact_valid": tags["foot_contact_valid"],
                        "ctrl_vertical_velocity": tags["ctrl_vertical_velocity"],
                        "smooth_root_pos": tags["smooth_root_pos"],
                },
            "energy_feet": tags["energy_feet"],
            "global_pos": global_pos,
            "rots": self.rots
        }
 
        self.norm_motions.append(motion)
        self.bvhs.append(bvh)
        self.filenames.append(filename)
        network_timing.end_data_timer("input_processing_motion_data_add_motion", timing_start)

    def normalize(self):
        timing_start = network_timing.start_data_timer()
        assert self.means is not None
        assert self.stds is not None

        def normalize_entry(container, key):
            if key in container and key in self.means and key in self.stds:
                if container[key].shape[1:] != self.means[key].shape:
                    raise ValueError(
                        "Cannot normalize tag '{}': value shape {} does not match stats shape {}".format(
                            key,
                            tuple(container[key].shape),
                            tuple(self.means[key].shape),
                        )
                    )
                container[key] = (container[key] - self.means[key]) / self.stds[key]
              
        for motion in self.norm_motions:
            motion["offsets"] = (motion["offsets"] - self.means["offsets"]) / self.stds[
                "offsets"
            ]
            motion["dqs"] = (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
            motion["displacement"] = (
                motion["displacement"] - self.means["displacement"]
            ) / self.stds["displacement"]
            motion["disp_8"] = (motion["disp_8"] - self.means["disp_8"]) / self.stds["disp_8"]
            motion["rots"] = (motion["rots"] - self.means["rots"]) / self.stds["rots"]
            if "tags" in motion:
                tags = motion["tags"]
                for key in (
                    "velo",
                    "dx",
                    "dz",
                    "height_full",
                    "alignment",
                    "root_rots",
                    "direction",
                    "head_height",
                    "amps",
                    "sin",
                    "cos",
                    "foot_labels",
                    "yaw_sin",
                    "yaw_cos",
                    "yaw_quats",
                    "sin_diff",
                    "cos_diff",
                    "ctrl_velocity",
                    "ctrl_acceleration",
                    "ctrl_yaw_rate",
                    "ctrl_yaw_accel",
                    "ctrl_height",
                    "ctrl_head_height",
                    "ctrl_roll",
                    "ctrl_foot_labels",
                    "ctrl_vertical_velocity",
                    "smooth_root_pos",
                ):
                    normalize_entry(tags, key)
         
        network_timing.end_data_timer("input_processing_motion_data_normalize", timing_start)

    def set_tags(self,tags):
        self.norm_motions[0]["tags"] = tags

    def get_bvh(self, index):
        return self.bvhs[index], self.filenames[index]

    def get_len(self):
        return len(self.norm_motions)

    def get_item(self, index):
        return self.norm_motions[index]
    
    def get_phase(self):
        return self.norm_motions[0]["phase"]

    def get_pos(self):
        return self.pos_all_joints


class RunMotionData:
    def __init__(self, param, device):
        self.param = param
        self.device = device

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def set_offsets(self, offsets):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        """
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        self.motion = {
            "offsets": offsets,
            "denorm_offsets": offsets.clone(),
        }

    def set_motion_from_bvh(self, offsets, global_pos, rotations, parents):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, np.array(parents), offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        if(USE_GLOBAL_HEIGHT):
            new_displacement = torch.zeros((displacement.shape[0], 4), device='cuda:0')
            for frame in range(displacement.shape[0]):
                new_displacement[frame] = torch.cat((displacement[frame], global_pos[frame, 1].unsqueeze(0)))
            displacement = new_displacement
        self.motion["dqs"] = dqs
        self.motion["displacement"] = displacement

    def set_motion(self, positions, rotations):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        global_pos = positions[:, 0, :]
        dqs = dquat.from_rotation_translation(rotations, positions)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        if(USE_GLOBAL_HEIGHT):
            new_displacement = torch.zeros((displacement.shape[0], 4), device='cuda:0')
            for frame in range(displacement.shape[0]):
                new_displacement[frame] = torch.cat((displacement[frame], global_pos[frame, 1].unsqueeze(0)))
            displacement = new_displacement
        self.motion["dqs"] = dqs
        self.motion["displacement"] = displacement

    def normalize_offsets(self):
        assert self.means is not None
        assert self.stds is not None
        self.motion["offsets"] = (
            self.motion["offsets"] - self.means["offsets"]
        ) / self.stds["offsets"]

    def normalize_motion(self):
        assert self.means is not None
        assert self.stds is not None
        self.motion["dqs"] = (self.motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
        self.motion["displacement"] = (
            self.motion["displacement"] - self.means["displacement"]
        ) / self.stds["displacement"]

    def get_item(self):
        return self.motion

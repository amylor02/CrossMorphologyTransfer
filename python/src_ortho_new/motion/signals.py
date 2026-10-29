import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import pymotion.rotations.quat as quat


USE_GLOBAL_HEIGHT = True # to include global height in displacement


DOWNSAMPLE_FACTOR = 8


WINDOW_STEP_SIZE = 8


def calc_velocity_from_pos(pos, downsample_factor, num_groups):
    x_values = pos[:,0,0].copy()  # ROOT X-axis movement data 
    z_values = pos[:,0,2].copy()  # ROOT Z-axis movement data
    step_size = WINDOW_STEP_SIZE  # Calculate every 8 frames
    window_size = downsample_factor  # Look ahead this many frames
    
    num_windows = num_groups
    
    vx_per_window = np.zeros(num_windows)
    vz_per_window = np.zeros(num_windows)

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, pos.shape[0]-1)
        
        vx_per_window[i] = x_values[end_idx] - x_values[start_idx]
        vz_per_window[i] = z_values[end_idx] - z_values[start_idx]
    
    velocity_per_window = np.sqrt(vx_per_window**2 + vz_per_window**2 + 1e-8) * 2
    return velocity_per_window


def calc_feet_energy(pos, feet_idx = [8,12,15,18]):
    step_size = WINDOW_STEP_SIZE  # Calculate every 8 frames
    window_size = DOWNSAMPLE_FACTOR  # Look ahead this many frames
    num_windows = (pos.shape[0] // step_size)
    energy_feet = np.zeros(num_windows)
    
    for i in feet_idx:
        feet_pos = pos[:, i:i+1, :]  # [frames, 1, 3]
        x_values = feet_pos[:,0,0].copy()
        z_values = feet_pos[:,0,2].copy()
        
        vx_per_window = np.zeros(num_windows)
        vz_per_window = np.zeros(num_windows)
        
        for w in range(num_windows):
            start_idx = w * step_size
            end_idx = min(start_idx + window_size,pos.shape[0]-1)
            
            vx_per_window[w] = x_values[end_idx] - x_values[start_idx]
            vz_per_window[w] = z_values[end_idx] - z_values[start_idx]
        
        feet_velocity = np.sqrt(vx_per_window**2 + vz_per_window**2 + 1e-8) * 2
        energy_feet = energy_feet + 0.5 * (feet_velocity**2)
        
    return energy_feet


def calc_feet_energy_torch(pos, feet_idx=[8, 12, 15, 18]):
    batch_size, num_frames, num_joints, _ = pos.shape
    step_size = WINDOW_STEP_SIZE  # Calculate every 8 frames
    window_size = DOWNSAMPLE_FACTOR  # Look ahead this many frames
    num_windows = num_frames // step_size
    energy_feet = torch.zeros((batch_size, num_windows), device=pos.device)
    
    for i in feet_idx:
        feet_pos = pos[..., i:i+1, :]  # [batch, frames, 1, 3]
        x_values = feet_pos[..., 0, 0]
        z_values = feet_pos[..., 0, 2]
        
        indices = torch.arange(num_windows, device=pos.device) * step_size
        
        start_indices = indices
        end_indices = torch.min(indices + window_size, torch.tensor(num_frames -1 , device=pos.device))
        
        vx_per_window = x_values[..., end_indices] - x_values[..., start_indices]
        vz_per_window = z_values[..., end_indices] - z_values[..., start_indices]
        
        feet_velocity = torch.sqrt(vx_per_window**2 + vz_per_window**2 + 1e-8) * 2
        energy_feet = energy_feet + 0.5 * (feet_velocity**2)
    
    return energy_feet


def compute_foot_labels(displacements, idxes = [8, 12, 15, 18], threshold=0.05):
    """
    Compute binary foot labels based on foot positions and velocities.
    If the foot is moving, label it as 1, otherwise 0.
    """

    num_frames = displacements.shape[0]
    num_joints = len(idxes)
    labels = np.zeros((num_frames, num_joints))
    disp_magnitudes = np.zeros((num_frames, num_joints))

    for i, j in enumerate(idxes):
        # Compute global position for the foot joint: add root's position
        diff = np.diff(displacements[:, j, :],axis=0)
        # Use only the x and z components to compute the displacement magnitude
        #joint_disp_xz = global_foot[:, [0, 2]]
        disp_magnitudes[1:,i] = np.linalg.norm(diff, axis=1)
        # Threshold may be adjusted based on your data scale
        labels[1:, i] = disp_magnitudes[1:,i] > threshold

    return labels, disp_magnitudes


def compute_foot_labels2(displacements, idxes=[8, 12, 15, 18], threshold=0.05,
                        temp=0.02, smooth_sigma=1, adapt_mode="none", adapt_k=1.5):
    """
    Compute per-foot contact probabilities from joint positions.

    Args:
        displacements: np.ndarray shape (F, J, 3) (joint world positions per frame)
        idxes: list of joint indices to consider as feet
        threshold: legacy fixed threshold (used only if adapt_mode == 'fixed')
        temp: temperature for sigmoid mapping (smaller -> sharper probability)
        smooth_sigma: gaussian smoothing sigma applied to per-joint magnitudes
        adapt_mode: 'mad' | 'percentile' | 'fixed' - how to compute adaptive threshold
        adapt_k: multiplier (for MAD) or percentile (0..1) when using percentile mode

    Returns:
        probs: np.ndarray shape (F, n_feet) of contact probabilities in [0,1]
        disp_magnitudes: np.ndarray shape (F, n_feet) of displacement magnitudes (smoothed)
    """
    num_frames = displacements.shape[0]
    num_joints = len(idxes)
    disp_magnitudes = np.zeros((num_frames, num_joints), dtype=np.float32)

    # compute per-frame displacement magnitudes (use x,z components)
    for i, j in enumerate(idxes):
        # differences between consecutive frames for joint j
        disp_between = displacements[1:, j, :] - displacements[:-1, j, :]
        mags = np.linalg.norm(disp_between, axis=1)
        disp_magnitudes[1:, i] = mags

    # smooth magnitudes to reduce noise
    if smooth_sigma is not None and smooth_sigma > 0:
        for i in range(num_joints):
            disp_magnitudes[:, i] = gaussian_filter1d(disp_magnitudes[:, i], sigma=smooth_sigma)

    # compute adaptive threshold per-joint
    if adapt_mode == "mad":
        med = np.median(disp_magnitudes, axis=0)
        mad = np.median(np.abs(disp_magnitudes - med[None, :]), axis=0)
        thr = med + adapt_k * mad
    elif adapt_mode == "percentile":
        p = float(adapt_k) if adapt_k <= 1.0 else adapt_k / 100.0
        thr = np.percentile(disp_magnitudes, p * 100.0, axis=0)
    else:  # fixed threshold
        thr = np.full((num_joints,), float(threshold), dtype=disp_magnitudes.dtype)

    # prevent tiny thresholds
    thr = np.maximum(thr, 1e-6)

    # map magnitude -> contact probability: contact when mag < thr
    # logits = (thr - mag) / temp  -> p = sigmoid(logits)
    denom = float(temp) if temp > 0.0 else 1e-6
    logits = (thr[None, :] - disp_magnitudes) / denom
    probs = 1.0 / (1.0 + np.exp(-logits))

    # optional: remove isolated spikes (if a one-frame positive surrounded by negatives)
    # convert to binary mask for isolation detection then restore probabilities for non-isolated frames
    binary = (probs > 0.5).astype(np.float32)
    for i in range(num_joints):
        for t in range(1, num_frames - 1):
            if binary[t, i] == 1 and binary[t - 1, i] == 0 and binary[t + 1, i] == 0:
                probs[t, i] = 0.0

    return probs, disp_magnitudes


def compute_tags(pos, head_idx = 2, head_height=1.0, downsample_factor = 8, is_human = False, skeleton_height = 0.90, rots = None,
                is_deg = True, scalar = 1.0, feet_idx = [8,12,15,18], quats=None, padded_frames = 0, shoulder_idx=[5,9], not_dog=True, feet_contact_threshold=0.02,
                yaw_error=0.0):

    num_frames = pos.shape[0]
    num_groups = num_frames // WINDOW_STEP_SIZE # each latent vector (encoder embedding) corresponds to 8 frames
    remainder = num_frames % downsample_factor
    

    #DX DZ AND VELOCITY
    #pos[:,1:,:] = pos[:,1:,[1,2,0]]
    
    # replace initial frames to suppress initialization artifacts
    if (is_human):
        pos[0:5] = np.copy(pos[5:10]) 
        
    pos_og = np.copy(pos)

    root_pos = np.copy(pos[:, 0, :])
    x = np.copy(root_pos[:, 0])
    z = np.copy(root_pos[:, 2])

    sample_rate_waypoints = 8
    sample_indices = np.append(np.arange(0, num_frames, sample_rate_waypoints), num_frames-1)
    samples_x = x[sample_indices] -x[0]
    samples_z = z[sample_indices] -z[0]
     

    dx = np.diff(samples_x)
    dz = np.diff(samples_z)

    amplitude = np.sqrt(dx**2 + dz**2)

    direction = np.arctan2(dz, dx)

    # Combine amplitude and direction into waypoints
    waypoints = np.stack([amplitude, direction], axis=1)  # Shape: (num_samples, 2)

    amplitudes_waypoints = waypoints[:, 0]
    direction_waypoints = waypoints[:, 1]

    sin_waypoints = np.sin(direction)
    cos_waypoints = np.cos(direction)

    x_splined = gaussian_filter1d(x, sigma=15)
    z_splined = gaussian_filter1d(z, sigma=15)
    
    direction_window = 1  # e.g., look 16 frames ahead
# Calculate the direction angles over the larger window, but only when movement exceeds threshold
    moving_direction_angles = np.zeros(num_frames)
    last_ref_pos_x = x_splined[0]  # Initialize reference position
    last_ref_pos_z = z_splined[0]
    last_direction = 0.0  # Initial direction value

    for i in range(num_frames):
        # Check if we've moved enough from the last reference position
        current_x = x_splined[i]
        current_z = z_splined[i]
        distance_moved = np.sqrt((current_x - last_ref_pos_x)**2 + (current_z - last_ref_pos_z)**2)
        
        if distance_moved > 0.01:  # Movement threshold
            # Determine the future frame to look at
            future_idx = min(i + direction_window, num_frames - 1)
            
            dx_window = x_splined[future_idx] - x_splined[i]
            dz_window = z_splined[future_idx] - z_splined[i]
            last_direction = np.arctan2(dz_window, dx_window)
            last_ref_pos_x = current_x
            last_ref_pos_z = current_z
        
        # Always assign the current best direction estimate
        moving_direction_angles[i] = last_direction

    # Unwrap the angles to avoid discontinuities at ±π
    #moving_direction_angles = rots[:, 0]
    moving_direction_angles = np.unwrap(moving_direction_angles)
    angles_splined = gaussian_filter1d(moving_direction_angles, sigma=20)
    sin_theta = np.sin(angles_splined)
    cos_theta = np.cos(angles_splined)
  
    #gaussian smoothing instead of spline
    #x = gaussian_filter1d(x, sigma=40)
    #z = gaussian_filter1d(z, sigma=40)
    x = x_splined
    z = z_splined

    root_pos[:,0] = x
    root_pos[:,2] = z 

    direction = np.arctan2(root_pos[:, 2], root_pos[:, 0])  # atan2(z, x)

    if(padded_frames > 0):
        root_pos[-padded_frames:] = root_pos[-padded_frames]

    smooth_pos = np.copy(pos[:, 0, :])

    step = DOWNSAMPLE_FACTOR
    pos_8 = np.copy(smooth_pos[::step])
    disp_8 = pos_8[1:] - pos_8[:-1]
    disp_8 = np.vstack((np.zeros((1, 3)), disp_8))  # Pad with zeros for the first 8 frames
    for i in range(3):
        disp_8[:,i] = gaussian_filter1d(disp_8[:, i], sigma=3)
        
    displacement = root_pos[1:, [0, 2]] - root_pos[:-1, [0, 2]]
    displacement = np.vstack((np.zeros((1, 2)), displacement))  # Pad with zeros for the first 8 frames
    time_step = 1 / 30.0  # Time from one frame to the next (8 frames)
    velocity = displacement / time_step
    velocity_xz = np.linalg.norm(velocity, axis=1)
    velocity_xz[:1] = velocity_xz[1]
    acceleration_xz = np.diff(velocity_xz,append=velocity_xz[-1]) / time_step
    acceleration_xz[:1] = acceleration_xz[1]

    dx = np.diff(x, prepend=x[0])  # change in x
    dz = np.diff(z, prepend=z[0])  # change in z
    dx[0] = dx[1]
    dz[0] = dz[1]

    # round to nearest 0.1
    #dx = np.round(dx, decimals=2)
    #dz = np.round(dz, decimals=2) 

    future_window = 8
    num_frames = len(x)
    future_displacement_mag = np.zeros(num_frames)  # displacement magnitude over the next future_window frames
    displacement_angle = np.zeros(num_frames)  # Angle of displacement

    for i in range(num_frames):
        if i + future_window < num_frames:
            # Vector from current position to position 16 frames ahead
            dx_future = x[i + future_window] - x[i]
            dz_future = z[i + future_window] - z[i]

        else:
            # For the last 16 frames where we can't look ahead fully,
            # either repeat the last valid value or set to zero
            if i > 0:
                dx_future = x[-1] - x[i]
                dz_future = z[-1] - z[i]

        future_displacement_mag[i] = np.sqrt(dx_future**2 + dz_future**2)
        displacement_angle[i] = np.arctan2(dz_future, dx_future)

 
    # Unwrap angles to avoid discontinuities
    disp_ang = np.unwrap(displacement_angle)
    past_window = 1
    num_frames = len(disp_ang)
    angle_delta = np.zeros(num_frames)

    for i in range(num_frames):
        if i >= past_window:
            delta = disp_ang[i] - disp_ang[i - past_window]
        else:
            delta = disp_ang[i] - disp_ang[0]
        # normalize to [-π, π]
        angle_delta[i] = (delta + np.pi) % (2 * np.pi) - np.pi

    # now use angle_delta for plotting
    displacement_angle = angle_delta

    frames = np.arange(num_frames)
    future_displacement_mag = np.round(future_displacement_mag*8)/8
    displacement_angle = np.round(displacement_angle*8)/8 
   
   
    # ALIGNMENT
    root_y_rot = np.unwrap(rots[:, 0])
    root_y_rot = gaussian_filter1d(root_y_rot, sigma=30)
    root_dirs = np.stack([np.cos(root_y_rot), np.sin(root_y_rot)], axis=1)

    displacement = np.diff(root_pos, axis=0)  # shape: (num_frames-1, 3)
    disp_xz = displacement[:, [0, 2]]  # shape: (num_frames-1, 2)
    norm_disp = np.linalg.norm(disp_xz, axis=1, keepdims=True)
    # Avoid dividing by zero
    norm_disp[norm_disp == 0] = 1.0
    moving_direction = disp_xz #/ norm_disp  # shape: (num_frames-1, 2)
    # To match the original sequence length, we prepend the first moving direction.
    moving_direction = np.vstack((moving_direction[0:1, :], moving_direction))
    moving_direction_yaw = np.arctan2(moving_direction[:, 0], moving_direction[:, 1])
    moving_direction_yaw = np.unwrap(moving_direction_yaw)
        
    num_frames = len(root_y_rot)

    alignment = np.zeros(num_frames)
    for i in range(num_frames):
        # ax.cla()  # clear axis
        # ax.set_xlim(-1.2, 1.2)
        # ax.set_ylim(-1.2, 1.2)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Z")
        # ax.set_title("Forward Facing vs. Moving Direction (Frame {})".format(i))
        # ax.grid(True)
        # Compute the unit vector from the root's yaw
        root_vect = np.array([np.cos(root_y_rot[i]), np.sin(root_y_rot[i])])
        
        # Compute the unit vector from the moving direction yaw
        moving_vect = np.array([np.cos(moving_direction_yaw[i]), np.sin(moving_direction_yaw[i])])
        moving_vect = moving_vect * velocity_xz[i] # scale by velocity
        mag = np.linalg.norm(moving_vect)
        if mag > 1:
            moving_vect = moving_vect / mag

        alignment[i] = np.dot(root_vect, moving_vect)
    
        
    #     # Draw the root yaw vector in blue
    #     ax.arrow(0, 0, root_vect[0], root_vect[1],
    #             head_width=0.05, head_length=0.05,
    #             fc='blue', ec='blue', label="Root Yaw")
        
    #     # Draw the moving direction vector in orange
    #     ax.arrow(0, 0, moving_vect[0], moving_vect[1],
    #             head_width=0.05, head_length=0.05,
    #             fc='orange', ec='orange', label="Moving Direction Yaw")

        
    #     # Optionally add a legend (note: legend call can be expensive in an animation loop)
    #     ax.legend(["Origin", "Root Yaw", "Moving Direction", "Sum"], loc="upper right")
        


    # SMOOTHED ROOT QUATS
    if(quats is not None):
        quat_joint0 = quats[:, 0, :]  # shape: (num_frames, 4)
        
        smoothed_quat_joint0 = quat_joint0.copy()
        sigma = 50
 
        # Smooth each component for joint 0 and joint 4
        for comp in range(4):
            smoothed_quat_joint0[:, comp] = gaussian_filter1d(quat_joint0[:, comp], sigma)


    pos_foot = np.copy(pos[:,[feet_idx]]) # + pos[:,0]
     

    diff= np.diff(pos_foot, axis=0)
    velo_foot_magn = np.linalg.norm(diff[:], axis=-1)
    

    velocity_per_8_frames = calc_velocity_from_pos(pos[:,0:1,:], downsample_factor, num_groups)
    acceleration_per_8_frames = np.diff(velocity_per_8_frames, prepend=velocity_per_8_frames[0])

    # angular velocity (how fast looking direction changes)
    if(is_human or True):
        rotation_y = rots[:,0]
        if(is_deg):
            rotation_y = np.deg2rad(rotation_y)
        rotation_y = rotation_y / 3.14159
        head_pos = pos[:, 4, :] 
        head_pos = head_pos - root_pos # root centered
        rotation_y = rotation_y + np.arctan2(head_pos[:, 2], head_pos[:, 0])  # atan2(z, x)
    else:
        head_pos = pos[:, 4, :] # z x y
        head_pos = head_pos[:, [1, 2, 0]]  # Change from z x y to x y z
        head_pos = head_pos - root_pos # root centered
        rotation_y = np.arctan2(head_pos[:, 2], head_pos[:, 0])  # atan2(z, x)

    rotation_y = np.unwrap(rotation_y)  # Unwrap to avoid jumps
    angular_velocities = []
    for i in range(num_groups):
        start_idx = i * WINDOW_STEP_SIZE
        end_idx = min(start_idx + downsample_factor, num_frames-1)
        # Calculate angular displacement over window
        angular_displacement = rotation_y[end_idx-1] - rotation_y[start_idx]
        angular_velocities.append(angular_displacement / downsample_factor)
    angular_velocity_per_8_frames = np.array(angular_velocities)

    # height per 8 frames
    # HUMAN HEIGHT = 0.96, DOG HEIGHT = 0.90
    height = root_pos[:, 1] / skeleton_height  # y values
    heights = []
    for i in range(num_groups):
        start_idx = i * WINDOW_STEP_SIZE
        end_idx = min(start_idx + downsample_factor, num_frames-1)
        # Calculate mean height over window
        window_mean = np.mean(height[start_idx:end_idx])
        heights.append(window_mean)
    height_per_8_frames = np.array(heights)

    energy_feet = calc_feet_energy(pos, feet_idx)

    velocity_per_8_frames = np.round(velocity_per_8_frames, decimals=2)
    acceleration_per_8_frames = np.round(acceleration_per_8_frames, decimals=2)
    angular_velocity_per_8_frames = np.round(angular_velocity_per_8_frames, decimals=2)
    height_per_8_frames = np.round(height_per_8_frames, decimals=2)
    energy_feet = np.round(energy_feet, decimals=3)

    # to nearest 0.25
    velocity_xz = np.round(velocity_xz*4)/4
    height = np.round(height*5)/5
    alignment = np.round(alignment*4)/4

    # consecutive-frame root displacement magnitude, computed before dx and dz are rebound below
    root_displacement_mag = np.sqrt(dx**2 + dz**2)

    dsin = np.diff(sin_theta, prepend=sin_theta[0])
    dcos = np.diff(cos_theta, prepend=cos_theta[0])

    dx = dsin
    dz = dcos

    # Gaussian smoothing of velo_foot_magn is disabled (commented out). The velo_foot tag uses the unsmoothed foot_disp_raw.
    #velo_foot_magn = gaussian_filter1d(velo_foot_magn, sigma=3, axis=0)

         
    # combined_velo_sum = np.sum(np.abs(velo_foot_magn[:,0,:]), axis=1)
    # combined_velo_rms = np.sqrt(np.mean(velo_foot_magn[:,0,:]**2, axis=1))
    # combined_velo = gaussian_filter1d(combined_velo_sum, sigma=10)
    # actually using spine for dog
    # The head_height argument is overwritten by the next line and is not read.
    # The result is a binary flag, not a height: 1.0 where the head joint y-coordinate is above lying_down_threshold, else 0.0. Tag names are persisted.
    lying_down_threshold = 0.35
    head_height = pos[:,head_idx,1].copy() #- pos[:,0,1] #- head_height
    head_height = np.where(head_height > lying_down_threshold, 1.0, 0.0)

    # compute_foot_labels2 returns (contact probabilities, smoothed displacement magnitudes).
    # compute_foot_labels returns (binary labels, unsmoothed displacement magnitudes).
    foot_contact_prob,foot_disp_smooth = compute_foot_labels2(pos, feet_idx, threshold=feet_contact_threshold)
    foot_contact_binary, foot_disp_raw = compute_foot_labels(pos, feet_idx, threshold = feet_contact_threshold)
    # binary contact per actual foot. duplicated feet_idx entries count once, at their first position.
    foot_positions = [list(feet_idx).index(j) for j in dict.fromkeys(feet_idx)]
    foot_contact = (foot_contact_prob[:, foot_positions] > 0.5).astype(np.float32)
    

    root_yaw = np.unwrap(rots[:, 0])  # Unwrap to avoid jumps
     
    root_yaw_quaternions = quat.from_angle_axis(np.expand_dims(root_yaw,axis=-1),
                            axis=np.tile(np.array([0, 1, 0]),(len(root_yaw),1)))  # Rotation around the vertical axis (Y-axis)
    root_yaw_quaternions = quat.normalize(quat.unroll(root_yaw_quaternions,axis=0))
    root_yaw_quaternions = gaussian_filter1d(root_yaw_quaternions, sigma=30, axis=0)
     

    yaw_sin = np.sin(root_yaw)
    yaw_cos = np.cos(root_yaw)
    yaw = np.arctan2(yaw_sin, yaw_cos)  # atan2(sin, cos)
    yaw_sin_diff = gaussian_filter1d(np.diff(yaw_sin, prepend=yaw_sin[0]),sigma=25)
    yaw_cos_diff = gaussian_filter1d(np.diff(yaw_cos, prepend=yaw_cos[0]),sigma=25)
    yaw_sin_recon = np.cumsum(yaw_sin_diff, axis=0) + yaw_sin[0]
    yaw_cos_recon = np.cumsum(yaw_cos_diff, axis=0) + yaw_cos[0]
    yaw_recon = np.arctan2(yaw_sin_recon, yaw_cos_recon)  # atan2(sin, cos)

    sample_indices = np.arange(0, len(root_yaw), 8)
    yaw_samples = root_yaw[sample_indices]

    sin_samples = np.sin(yaw_samples)
    cos_samples = np.cos(yaw_samples)

    yaw_sin_diff = np.diff(sin_samples, prepend=sin_samples[0])
    yaw_cos_diff = np.diff(cos_samples, prepend=cos_samples[0])

    pos_l_shoulder = pos[:, 5]
    pos_r_shoulder = pos[:, 9]
    
    l_shoulder_x = pos_l_shoulder[:, 0]
    l_shoulder_z = pos_l_shoulder[:, 2]
    r_shoulder_x = pos_r_shoulder[:, 0]
    r_shoulder_z = pos_r_shoulder[:, 2]

    dx = r_shoulder_x - l_shoulder_x
    dz = r_shoulder_z - l_shoulder_z
    

    perp_dx = -dz
    perp_dz = dx

    #####
    root_x = pos[:, 0, 0]
    root_z = pos[:, 0, 2]
    mid_shoulder_x = (l_shoulder_x + r_shoulder_x) / 2
    mid_shoulder_z = (l_shoulder_z + r_shoulder_z) / 2

    vec_x = mid_shoulder_x - root_x
    vec_z = mid_shoulder_z - root_z
    
    # Optionally normalize the vector
    norm = np.sqrt(vec_x**2 + vec_z**2)
    vec_x_norm = vec_x / (norm + 1e-8)
    vec_z_norm = vec_z / (norm + 1e-8)
    #####

    # Normalize perpendicular vectors for consistent arrow length
    norm = np.sqrt(perp_dx**2 + perp_dz**2)
    perp_dx_norm = perp_dx / (norm + 1e-8)
    perp_dz_norm = perp_dz / (norm + 1e-8)

    perp_angles = np.arctan2(vec_x_norm, vec_z_norm) # Angle in radians

    # Optionally unwrap to avoid discontinuities
    perp_angles_unwrapped = np.unwrap(perp_angles)
    sin_perp_angles = np.sin(perp_angles_unwrapped)
    cos_perp_angles = np.cos(perp_angles_unwrapped)
    sin_perp_angles = gaussian_filter1d(sin_perp_angles, sigma=20)
    cos_perp_angles = gaussian_filter1d(cos_perp_angles, sigma=20)


    sin_perp_angles_samples = sin_perp_angles[::8]
    cos_perp_angles_samples = cos_perp_angles[::8]

    shoulders_yaw_quaternions = quat.from_angle_axis(np.expand_dims(perp_angles_unwrapped, axis=-1),
                            axis=np.tile(np.array([0, 1, 0]), (len(perp_angles_unwrapped), 1)))  # Rotation around the vertical axis (Y-axis)
    shoulders_yaw_quaternions = quat.normalize(quat.unroll(shoulders_yaw_quaternions, axis=0))
    shoulders_yaw_quaternions = gaussian_filter1d(shoulders_yaw_quaternions, sigma=30, axis=0)
     
    #plt.plot(quats[:, 0, :], label="Root Quaternion")
    #plt.show()
    ##Plot the angle over frames
    # plt.figure(figsize=(10, 4))
    # #plt.plot(np.sin(perp_angles_unwrapped), label="Perpendicular Vector Angle (radians)")
    # #plt.plot(np.cos(perp_angles_unwrapped), label="Perpendicular Vector Angle (radians)") 
    # plt.plot(perp_angles_unwrapped, label="Perpendicular Vector Angle (radians)")
    # plt.plot(np.unwrap(rots[:,0]))
    # plt.xlabel("Frame")
    # plt.ylabel("Angle (radians)")
    # plt.title("Angle of Perpendicular Vectors in XZ Plane Over Time")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.plot(shoulders_yaw_quaternions)
    # plt.show()
    weight_weight = 0.5 # weight for the weights. seems like window size hyperparam combined with weights 
                        # affects how the model learns
    loss_weights = np.ones_like(root_displacement_mag)
    loss_weights[root_displacement_mag * scalar > 0.050] = 1#3 * weight_weight
    loss_weights[root_displacement_mag * scalar > 0.075] = 1#5 * weight_weight
    loss_weights[root_displacement_mag * scalar > 0.100] = 1#7 * weight_weight
    loss_weights[root_displacement_mag * scalar > 0.125] = 1#10 * weight_weight
    loss_weights[root_displacement_mag * scalar > 0.150] = 1#15 * weight_weight


 


    
    # # plt.subplot(4, 1, 4)
    # # plt.plot(amplitudes_waypoints, label="Amplitude")
    # # plt.title("Waypoints")
    # # plt.legend()
    # plt.subplot(4, 1, 4)
    # plt.plot(shoulders_yaw_quaternions[:,0], label="yaw")
    # plt.plot(shoulders_yaw_quaternions[:,2], label="yaw2")
    # #plt.plot(loss_weights)
    # plt.title("yaw")
    # plt.legend()



    extra = 0 #5 
    extra_sincos = 0
    if(not_dog):
        perp_angles_unwrapped = root_yaw #
    perp_angles_unwrapped = perp_angles_unwrapped + yaw_error

    sin_perp_angles = np.sin( gaussian_filter1d(perp_angles_unwrapped, sigma=25+extra_sincos))
    cos_perp_angles = np.cos( gaussian_filter1d(perp_angles_unwrapped, sigma=25+extra_sincos))
    
    
    root_displacements = np.diff(pos[:,0,[0,2]],axis=0, prepend=pos[0:1,0,[0,2]])
    root_displacements[:,0] = gaussian_filter1d(root_displacements[:,0],sigma = 10)
    root_displacements[:,1] = gaussian_filter1d(root_displacements[:,1], sigma = 10)
    root_displacements = root_displacements * scalar

    ctrl_velocity = np.linalg.norm(root_displacements, axis=1) / (1/30.0) #* scalar
    

    ctrl_acceleration = np.diff(ctrl_velocity[1:], prepend=ctrl_velocity[0]) / time_step
    ctrl_acceleration = np.insert(ctrl_acceleration,0,ctrl_acceleration[0])

    ctrl_yaw_rate = np.diff(gaussian_filter1d(perp_angles_unwrapped, sigma=30 + extra), prepend=perp_angles_unwrapped[0]) / time_step
    ctrl_yaw_rate[0] = ctrl_yaw_rate[1]

    ctrl_yaw_accel = np.diff(ctrl_yaw_rate, prepend=ctrl_yaw_rate[0]) / time_step

    ctrl_height = root_pos[:, 1] / skeleton_height 
    ctrl_height = gaussian_filter1d(ctrl_height, sigma=8)
    
    ctrl_vertical_velocity = np.diff(ctrl_height, prepend=ctrl_height[0]) / time_step
    ctrl_vertical_velocity[0] = ctrl_vertical_velocity[1]  # Fix first frame
    ctrl_height[ctrl_height <= 0.5] = 0 
    ctrl_height[ctrl_height >= 1.05] = 1.4
    ctrl_vertical_velocity[ctrl_height <= 1.1 ] = 0
    ctrl_vertical_velocity[ctrl_vertical_velocity > 0.2 ] = 1
    ctrl_vertical_velocity[ctrl_vertical_velocity < -0.2 ] = -1
    


    # 0 mean unit variance
    velocity_magn_normalized = (ctrl_velocity - np.mean(ctrl_velocity)) / np.std(ctrl_velocity)
    acceleration_magn_normalized = (ctrl_acceleration - np.mean(ctrl_acceleration)) / np.std(ctrl_acceleration)
    
    left_shoulder = pos[:, shoulder_idx[0], :]  # [frames, 3]
    right_shoulder = pos[:, shoulder_idx[1], :]  # [frames, 3]
    shoulder_vector = left_shoulder - right_shoulder  # [frames, 3]
    # Compute the roll angle (angle between shoulder vector and horizontal plane)
    roll_angles = np.arctan2(shoulder_vector[:, 1], np.linalg.norm(shoulder_vector[:, [0, 2]], axis=1))  # [frames]
    roll_angles = gaussian_filter1d(roll_angles, sigma=30)
    ctrl_roll_angles = np.round(roll_angles)
    
    
    smooth_root_pos = gaussian_filter1d(root_pos, sigma=1,axis=0)
    smooth_root_pos[:,0] = smooth_root_pos[:,0] * scalar
    smooth_root_pos[:,1] /= skeleton_height
    smooth_root_pos[:,2] = smooth_root_pos[:,2] * scalar


    # plt.plot(pos[:,0])
    # plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.plot(ctrl_velocity)
    # plt.plot(ctrl_velocity_)
    # plt.plot(ctrl_velocity__)
    # plt.plot(ctrl_acceleration)
    # plt.plot(ctrl_height)
    # plt.plot(ctrl_height**1.45)
    # plt.plot(ctrl_vertical_velocity)
    # plt.plot(ctrl_yaw_rate)
    # plt.plot(gaussian_filter1d(root_yaw,sigma=30))
    # plt.plot(root_yaw)
    # plt.plot(ctrl_yaw_accel)
    # plt.plot(head_height)
    # plt.plot(pos[:,head_idx,1])
    # plt.plot(roll_angles)
    # plt.grid(True)
    # plt.plot(sin_perp_angles)
    # plt.plot(cos_perp_angles)
    # plt.plot(foot_disp_smooth)
    # plt.plot(foot_disp_raw)
    # plt.plot(smooth_root_pos)
    # plt.plot(foot_contact_prob[:,-2:])
    # plt.plot(pos[:,8])
    # plt.plot(ctrl_velocity)
    # plt.plot(root_pos)
    # plt.plot(perp_angles_unwrapped)
    #######################
    # ax=plt.gca()
    # ax.plot(ctrl_height, color="C0", linewidth=1.25)
    # X axis: frames
    # ax.set_xlim(0, num_frames - 1)
    # Y axis: data-driven with small padding to avoid clipping
    # ymin = 0
    # ymax = 3
    # pad = (ymax - ymin) * 0.1
    # if pad == 0:
    #     pad = 0.5
    # ax.set_ylim(ymin - pad, ymax + pad)
    #######################
    

    #####################################################################################
    #####################################################################################
    tags = {
        "velocity": velocity_per_8_frames * scalar ,
        "acceleration": acceleration_per_8_frames * scalar ,
        "ang_velocity": angular_velocity_per_8_frames * scalar ,
        "height": height_per_8_frames,
        "energy_feet": energy_feet,
        "velocity_full": velocity,
        "velo": root_displacement_mag * scalar, #!
        "accel": acceleration_xz,
        "dx" : dx, #* scalar, #!
        "dz" : dz, #* scalar, #!
        "height_full": height, #!
        "alignment": alignment,
        "root_rots" : smoothed_quat_joint0,
        "disp_8": disp_8,
        "direction": displacement_angle,
        "velo_foot": np.expand_dims(foot_disp_raw, axis=1),  # unsmoothed foot displacement magnitudes
        "head_height": head_height,
        "amps": amplitudes_waypoints * scalar, #!
        "sin": sin_waypoints, #! 
        "cos": cos_waypoints, #!
        "foot_labels": foot_disp_smooth,  # NOT contact labels: smoothed foot displacement magnitudes (persisted key)
        "yaw_sin": sin_perp_angles,
        "yaw_cos": cos_perp_angles,
        "yaw_quats": shoulders_yaw_quaternions,
        "sin_diff": sin_perp_angles_samples,
        "cos_diff": cos_perp_angles_samples,
        "loss_weights": loss_weights,
        "ctrl_velocity": ctrl_velocity,
        "ctrl_acceleration": ctrl_acceleration,
        "ctrl_yaw_rate": ctrl_yaw_rate,
        "ctrl_yaw_accel": ctrl_yaw_accel,
        "ctrl_height": ctrl_height,
        "ctrl_head_height": head_height,
        "ctrl_roll": ctrl_roll_angles,
        "ctrl_foot_labels": foot_contact_prob[:,2:],  # contact PROBABILITIES for feet 2 onward (persisted key)
        "ctrl_vertical_velocity": ctrl_vertical_velocity,
        "smooth_root_pos": smooth_root_pos,
        "binary_foot_labels": foot_contact_binary,  # 1 where the foot moves (displacement above threshold), not contact
        "foot_contact": foot_contact,  # 1 in contact, one column per actual foot
        "foot_contact_valid": np.ones_like(foot_contact),  # computed tags have every foot available
    }
  
    return tags

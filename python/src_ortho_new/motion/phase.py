import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def compute_phase(pos, joint_id = [15,18], indices_only = False):

    # Assuming x_values, z_values, and y_values are 1D arrays representing the movement over time
    
    # which joint to compute the phase from
    x_values = pos[:,0,0]  # ROOT X-axis movement data 
    z_values = pos[:,0,2]  # ROOT Z-axis movement data

    ### foot velocity
    num_joints = len(joint_id)
    num_frames = pos.shape[0]
    velocities = np.zeros((num_joints, num_frames))
    smoothed_velocities = np.zeros((num_joints, num_frames))

    for i, j in enumerate(joint_id):
        joint_pos = pos[:, j, :]  # [frames, 3]
        joint_velocity = np.sqrt(np.sum(np.diff(joint_pos, axis=0, prepend=joint_pos[0:1])**2, axis=1))
        velocities[i] = joint_velocity
        smoothed_velocities[i] = gaussian_filter1d(joint_velocity, sigma=12)  # Apply smoothing
    
    y_values = np.zeros((pos.shape[0], len(joint_id)))
    for i, j in enumerate(joint_id):
        y_values[:,i] = velocities[i, :]

    epsilon = 0.01  # Stationary threshold

    vx = np.diff(x_values, prepend=x_values[0])
    vz = np.diff(z_values, prepend=z_values[0])
    velocity_xz = np.sqrt(vx**2 + vz**2)
    acceleration_xz = np.diff(velocity_xz, prepend=velocity_xz[0])


    num_groups = num_frames // 8
    vx_per_8_frames = np.array([x_values[(i+1)*8-1] - x_values[i*8] for i in range(num_groups)])
    vz_per_8_frames = np.array([z_values[(i+1)*8-1] - z_values[i*8] for i in range(num_groups)])
    velocity_per_8_frames = np.sqrt(vx_per_8_frames**2 + vz_per_8_frames**2)
    acceleration_per_8_frames = np.diff(velocity_per_8_frames, prepend=velocity_per_8_frames[0])

    head_pos = pos[:, 4, :] # z x y
    head_pos = head_pos[:, [1, 2, 0]]  # Change from z x y to x y z
    root_pos = pos[:, 0, :] 

   
    head_pos = head_pos - root_pos # root centered
    rotation_y = np.arctan2( head_pos[:, 0], head_pos[:, 2])  # atan2(z, x)
    rotation_y = np.unwrap(rotation_y)  # Unwrap to avoid jumps
    angular_velocity_per_8_frames = np.array([rotation_y[(i+1)*8-1] - rotation_y[i*8] for i in range(num_groups)]) 
    angular_velocity_per_8_frames = gaussian_filter1d(angular_velocity_per_8_frames, sigma=3)


    velocity_xz = gaussian_filter1d(velocity_xz, sigma=5)
    acceleration_per_8_frames = gaussian_filter1d(acceleration_per_8_frames, sigma =3)
    velocity_threshold = -100.0001
    active_indices = np.where(velocity_xz > velocity_threshold)[0]
    inactive_indices = np.where(velocity_xz < velocity_threshold)[0]
    velocity_xz[inactive_indices] *= 0
    active_y_values = y_values[active_indices]

    smoothed_y = np.zeros([y_values.shape[1],y_values.shape[0]])
    indicator_array = np.zeros_like(smoothed_y)  # array that will hold indices where we have peaks and valleys in the phase signal
    peaks = []
    valleys = []
    phase = np.zeros_like(smoothed_y)
    num_points = len(smoothed_y[0])
    for j, joint in enumerate(joint_id):
        smoothed_y[j] = gaussian_filter1d(y_values[:,j], sigma=12)

        p, _ = find_peaks(smoothed_y[j],)
        v, _ = find_peaks(-smoothed_y[j],)
        peaks.append(p)
        valleys.append(v)
        
        indicator_array[j, peaks[j]] = 1
        indicator_array[j, valleys[j]] = -1

        min_val = np.min(indicator_array[j])
        max_val = np.max(indicator_array[j])

        if max_val > min_val:
            indicator_array[j] = (indicator_array[j] - min_val) / (max_val - min_val)
        else: # All 0s
            indicator_array[j] = np.zeros_like(indicator_array[j])
        
        phase[j] = construct_signal(peaks[j], valleys[j], num_points, inactive_indices)
    
    if(indices_only):
        return indicator_array, velocity_xz


    possible_values = np.linspace(0.0, 1.0, num=11)  # [0.0, 0.1, …, 1.0]
    distances = np.abs(phase[..., np.newaxis] - possible_values)
    closest_idx = np.argmin(distances, axis=-1)
    quantized_phase = possible_values[closest_idx]          

    return phase, velocity_xz


def construct_signal(peaks, valleys, num_points, inactive):
        """Interpolate a phase-like signal between the supplied peaks and valleys."""
        signal = np.zeros(num_points)
        # Ensure peaks and valleys are sorted
        points = np.concatenate((peaks,valleys))
        points = sorted(points)
        # Iterate through the points to construct the signal
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            x = np.linspace(0, 1, end - start)  # Normalize to 0 to 1
            if points[i] in inactive:
                signal[start:end] = 0
            elif points[i] in peaks:
                signal[start:end] = 0.5 * (1 + np.cos(np.pi * x))  # Descend to valley (sine wave)
            elif points[i] in valleys:
                signal[start:end] = 0.5 * (1 - np.cos(np.pi * x))  # Ascend to peak (sine wave)
        
        return signal

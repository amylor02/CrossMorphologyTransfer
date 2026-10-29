import os
import numpy as np
from pymotion.ops.skeleton import from_root_dual_quat, from_root_dual_quat_to_root
import pymotion.rotations.ortho6d as ortho6d
import pymotion.rotations.quat as quat
from pymotion.rotations.dual_quat import to_rotation_translation


def align_mask_to_frames(mask, n_frames):
    """Pad with False or truncate a per-frame boolean mask to n_frames."""
    aligned = np.zeros(n_frames, dtype=bool)
    k = min(n_frames, len(mask))
    aligned[:k] = np.asarray(mask, dtype=bool)[:k]
    return aligned


def result_to_bvh(
    res,
    means,
    stds,
    bvh,
    filename,
    save=True,
    initial_frame=None,
    feet_idx=None,
    copy_init_frame=False,
    disp_8=None,
    yaw_diff=None,
    initial_sin_cos=None,
    initial_ortho=None,
    stationary_root_mask=None,
):

    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    res = res.cpu().detach().numpy()
    frames = res.shape[0]
    pos = res[:, 6:10]
    root = res[:, :9].copy()
    root = root * stds["rots"].cpu().numpy() + means["rots"].cpu().numpy()

    rots_6d = root[:, :6]
    root_pos = root[:, 6:9]


    dqs = res
    dqs = dqs * stds["dqs"].cpu().numpy() + means["dqs"].cpu().numpy()
    dqs = dqs.reshape(dqs.shape[0], dqs.shape[1] // 9, 9)  # frames, n_joints, 9
    pred_positions = np.copy(dqs[:, 0, 6:9])
    dqs = ortho6d.to_dual_quat(dqs)

    dqs = dqs.reshape(dqs.shape[0], -1, 8)

    import matplotlib.pyplot as plt

    pred_positions[np.abs(pred_positions) < 0.005] = 0  #######################
    # pred_positions[np.abs(pred_positions) < 0.015] = 0#######################
    # Source-guided stationary root lock: zero horizontal increments before cumulative summation,
    # so the integrated trajectory stays continuous. Height and rotations are not touched.
    if stationary_root_mask is not None:
        lock = align_mask_to_frames(stationary_root_mask, pred_positions.shape[0])
        pred_positions[lock, 0] = 0
        pred_positions[lock, 2] = 0
    pred_positions[0, [0, 2]] = bvh.data["positions"][1, 0, [0, 2]] * 0
    pred_positions[:, 0] = np.cumsum(pred_positions[:, 0], axis=0)  #############
    pred_positions[:, 2] = np.cumsum(pred_positions[:, 2], axis=0)  #############




    # pred_positions[:,2] = np.cumsum(pred_positions[:,2],axis=0)
    # pred_positions = pred_positions + bvh.data["positions"][0,0,:3]#######################

    zeros = np.zeros((pos.shape[0], pos.shape[1]))
    pos = pos * stds["displacement"].cpu().numpy() + means["displacement"].cpu().numpy()
    pos[0, :3] = bvh.data["positions"][0, 0, :3]
    pos = pred_positions
    trans, rots = from_root_dual_quat(dqs, np.array(bvh.data["parents"]))
    trans, _ = from_root_dual_quat_to_root(dqs, np.array(bvh.data["parents"]))
    _, trans = to_rotation_translation(dqs)
    diff = np.diff(trans[:, [8, 12, 15, 18]], axis=0)
    velo = np.linalg.norm(diff[:], axis=-1)



    if copy_init_frame:
        min_len = min(rots.shape[0], initial_frame.shape[0])
        rots[:min_len, 0, :] = initial_frame[:min_len, :]


    rot_roder = np.tile(bvh.data["rot_order"][0], (rots.shape[0], rots.shape[1], 1))
    rotations = bvh.to_degrees(quat.to_euler(rots, order=rot_roder))
    rot_roder = np.tile(
        bvh.data["rot_order"][0],
        (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1),
    )  # made a change here

    bvh.data["rotations"] = rotations

    # positions


    # Define smoothing factor (adjust as needed, higher = smoother)
    smoothing_factor = 1
    offset_distance = 1.0  # 1 meter in front of the character

    positions = bvh.data["positions"][: rotations.shape[0]]
    num_frames = positions.shape[0]

    # Get the original root positions - we'll need these for calculations
    original_root_pos = positions[:, 0, :].copy()

    # Calculate the offset points for each frame (1 meter in front of character)
    # Get y-rotation angles (assuming these represent the character's facing direction)
    y_rotations = rotations[:, 0, 0]  # Y-axis rotation for root joint
    y_rotations_rad = np.radians(
        y_rotations[: positions.shape[0]]
    )  # Convert to radians

    """# Calculate offset positions in XZ plane (1 meter in front based on orientation)
    offset_x = original_root_pos[:, 0] + offset_distance * np.sin(y_rotations_rad)
    offset_z = original_root_pos[:, 2] + offset_distance * np.cos(y_rotations_rad)
    offset_y = original_root_pos[:, 1].copy()  # Y position unchanged

    # Now sample and smooth these offset positions
    sample_indices = np.append(np.arange(0, num_frames, 32), num_frames-1)

    # For X dimension
    samples_x = offset_x[sample_indices]
    spline_x = UnivariateSpline(sample_indices, samples_x, s=smoothing_factor)
    smooth_offset_x = spline_x(np.arange(num_frames))

    # For Z dimension
    samples_z = offset_z[sample_indices]
    spline_z = UnivariateSpline(sample_indices, samples_z, s=smoothing_factor)
    smooth_offset_z = spline_z(np.arange(num_frames))

    # For Y dimension (height)
    samples_y = offset_y[sample_indices]
    spline_y = UnivariateSpline(sample_indices, samples_y, s=smoothing_factor)
    smooth_offset_y = spline_y(np.arange(num_frames))"""

    # Apply the smoothed offset positions
    # positions[:, 0, 0] = smooth_offset_x
    # positions[:, 0, 1] = smooth_offset_y
    # positions[:, 0, 2] = smooth_offset_z

    og_positions = np.copy(bvh.data["positions"])
    bvh.data["positions"] = positions
    bvh.data["positions"][:, 0] = pos[: positions.shape[0]]

    bvh.data["parents"][0] = None  # BVH sets root as None
    path = None

    if save:
        path = "data"
        filename = "eval_" + filename
        bvh.save(os.path.join(path, filename))
    return path, filename

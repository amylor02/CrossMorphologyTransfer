import numpy as np


SUPPORTED_ACCELERATION_MODES = {"fairmotion", "corrected"}
SUPPORTED_UP_AXES = {"y", "z"}


def _as_position_array(global_positions):
    """Validate the `[T, J, 3]` world-position contract used by the extractor."""
    positions = np.asarray(global_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "Expected global_positions with shape [num_frames, num_joints, 3]."
        )
    if positions.shape[0] < 3:
        raise ValueError(
            "Kinetic features require at least 3 frames to compute velocity and acceleration windows."
        )
    return positions


def _validate_configuration(positions, root_idx, dt, up_axis, acceleration_mode):
    """Validate the options that define the Fairmotion-compatible feature space."""
    if positions.shape[1] == 0:
        raise ValueError("Expected at least one joint in global_positions.")
    if not 0 <= root_idx < positions.shape[1]:
        raise ValueError(
            f"root_idx {root_idx} is out of range for {positions.shape[1]} joints."
        )
    if dt <= 0:
        raise ValueError("dt must be a positive value in seconds.")
    if up_axis not in SUPPORTED_UP_AXES:
        raise ValueError(f"up_axis must be one of {sorted(SUPPORTED_UP_AXES)}.")
    if acceleration_mode not in SUPPORTED_ACCELERATION_MODES:
        raise ValueError(
            "acceleration_mode must be one of "
            f"{sorted(SUPPORTED_ACCELERATION_MODES)}."
        )


def position_wrt_root(global_positions, root_idx=0):
    """Convert world coordinates to the root-relative coordinates Fairmotion uses."""
    positions = _as_position_array(global_positions)
    root_positions = positions[:, root_idx : root_idx + 1, :]
    return positions - root_positions


def _split_horizontal_vertical(vector, up_axis):
    """Split a 3D displacement into planar and up-axis components."""
    if up_axis == "z":
        horizontal = vector[[0, 1]]
        vertical = vector[2]
    else:
        horizontal = vector[[0, 2]]
        vertical = vector[1]
    return horizontal, vertical


def compute_kinetic_feature_vector(
    global_positions,
    root_idx=0,
    dt=1.0 / 30.0,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
):
    """
    Compute the Fairmotion-style kinetic feature vector of length `3 * num_joints`.

    The output concatenates `[horizontal_energy, vertical_energy, expenditure]`
    for each joint after first converting positions to root-relative coordinates.
    """
    positions = _as_position_array(global_positions)
    _validate_configuration(positions, root_idx, dt, up_axis, acceleration_mode)

    if window_size < 0:
        raise ValueError("window_size must be non-negative.")

    root_relative_positions = position_wrt_root(positions, root_idx=root_idx)
    num_frames, num_joints, _ = root_relative_positions.shape
    features = []

    for joint_idx in range(num_joints):
        horizontal_energy = 0.0
        vertical_energy = 0.0
        expenditure = 0.0

        for frame_idx in range(1, num_frames):
            displacement_sum = np.zeros(3, dtype=np.float64)
            valid_velocity_offsets = 0

            for offset in range(-window_size, window_size + 1):
                prev_idx = frame_idx + offset - 1
                curr_idx = frame_idx + offset
                if prev_idx < 0 or curr_idx >= num_frames:
                    continue
                displacement_sum += (
                    root_relative_positions[curr_idx, joint_idx]
                    - root_relative_positions[prev_idx, joint_idx]
                )
                valid_velocity_offsets += 1

            if valid_velocity_offsets == 0:
                raise ValueError(
                    "Velocity window is empty. Provide a longer motion or a smaller window_size."
                )

            denom = valid_velocity_offsets * dt
            horizontal_disp, vertical_disp = _split_horizontal_vertical(
                displacement_sum, up_axis
            )
            horizontal_speed = np.linalg.norm(horizontal_disp / denom)
            vertical_speed = abs(vertical_disp / denom)

            # Fairmotion stores squared speeds as energy proxies; there is no mass term.
            horizontal_energy += horizontal_speed * horizontal_speed
            vertical_energy += vertical_speed * vertical_speed

            acceleration_sum = np.zeros(3, dtype=np.float64)
            valid_acceleration_offsets = 0
            for offset in range(-window_size, window_size + 1):
                prev_idx = frame_idx + offset - 1
                curr_idx = frame_idx + offset
                next_idx = frame_idx + offset + 1
                if prev_idx < 0 or next_idx >= num_frames:
                    continue

                next_velocity = (
                    root_relative_positions[next_idx, joint_idx]
                    - root_relative_positions[curr_idx, joint_idx]
                ) / dt

                if acceleration_mode == "fairmotion":
                    # This intentionally keeps Fairmotion's current operator precedence so
                    # parity mode matches its exported feature vectors.
                    prev_velocity = (
                        root_relative_positions[curr_idx, joint_idx]
                        - root_relative_positions[prev_idx, joint_idx] / dt
                    )
                else:
                    # This branch uses the corrected backward-difference velocity.
                    prev_velocity = (
                        root_relative_positions[curr_idx, joint_idx]
                        - root_relative_positions[prev_idx, joint_idx]
                    ) / dt

                acceleration_sum += (next_velocity - prev_velocity) / dt
                valid_acceleration_offsets += 1

            if valid_acceleration_offsets == 0:
                raise ValueError(
                    "Acceleration window is empty. Provide a longer motion or a smaller window_size."
                )

            expenditure += np.linalg.norm(
                acceleration_sum / valid_acceleration_offsets
            )

        normalization = num_frames - 1
        features.extend(
            [
                horizontal_energy / normalization,
                vertical_energy / normalization,
                expenditure / normalization,
            ]
        )

    return np.asarray(features, dtype=np.float64)


def compute_kinetic_features(*args, **kwargs):
    """Backward-compatible alias for callers that prefer the shorter name."""
    return compute_kinetic_feature_vector(*args, **kwargs)
import os

import numpy as np
from pymotion.io.bvh import BVH

from utils.kinetic_features import compute_kinetic_feature_vector


def load_bvh_global_positions(bvh_path):
    """Load one BVH and return the global joint positions used by FID metrics."""
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    bvh = BVH()
    bvh.load(bvh_path)
    pos_all_joints = bvh.compute_global_pos()
    return bvh, pos_all_joints


def _ensure_feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    """Normalize feature inputs to `[N, D]` matrices before FID statistics."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return matrix


def _compute_feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and covariance, treating one-sample inputs as zero-variance."""
    matrix = np.asarray(features, dtype=np.float64)
    mean = np.mean(matrix, axis=0)
    feature_dim = matrix.shape[1]

    if matrix.shape[0] == 1:
        covariance = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    else:
        covariance = np.cov(matrix, rowvar=False)
        covariance = np.atleast_2d(covariance).astype(np.float64, copy=False)
        if covariance.shape != (feature_dim, feature_dim):
            covariance = covariance.reshape(feature_dim, feature_dim)

    return mean, covariance


def _normalize_features_with_reference_stats(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both matrices with the reference dataset mean/std."""
    reference_matrix = _ensure_feature_matrix(reference_features, "reference_features")
    query_matrix = _ensure_feature_matrix(query_features, "query_features")
    if reference_matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError(
            "Feature dimensionality mismatch: "
            f"reference={reference_matrix.shape[1]}, query={query_matrix.shape[1]}"
        )

    reference_mean = np.mean(reference_matrix, axis=0, keepdims=True)
    reference_std = np.std(reference_matrix, axis=0, keepdims=True)
    reference_std = np.maximum(reference_std, eps)

    normalized_reference = (reference_matrix - reference_mean) / reference_std
    normalized_query = (query_matrix - reference_mean) / reference_std
    return normalized_reference, normalized_query


def compute_motion_fid(real_features: np.ndarray, generated_features: np.ndarray, eps=1e-6) -> float:
    """Compute Gaussian FID on arbitrary feature matrices with small-sample guards."""
    from scipy import linalg

    real_features = _ensure_feature_matrix(real_features, "real_features")
    generated_features = _ensure_feature_matrix(generated_features, "generated_features")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError(
            "Feature dimensionality mismatch: "
            f"real={real_features.shape[1]}, generated={generated_features.shape[1]}"
        )

    mu_real, sigma_real = _compute_feature_statistics(real_features)
    mu_gen, sigma_gen = _compute_feature_statistics(generated_features)

    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_gen += np.eye(sigma_gen.shape[0]) * eps

    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def _coerce_frame_time(candidate):
    """Extract one positive scalar frame time from the different shapes BVH libs use."""
    if candidate is None:
        return None
    array = np.asarray(candidate).reshape(-1)
    if array.size == 0:
        return None
    value = float(array[0])
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def _resolve_bvh_frame_time(bvh, fallback=1.0 / 30.0):
    """Read frame time from the BVH object and fall back to 30 FPS if absent."""
    candidates = [
        getattr(bvh, "frame_time", None),
        getattr(bvh, "frametime", None),
        getattr(bvh, "frame_time_seconds", None),
    ]

    data = getattr(bvh, "data", None)
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("frame_time"),
                data.get("frame_times"),
                data.get("frametime"),
            ]
        )

    for candidate in candidates:
        frame_time = _coerce_frame_time(candidate)
        if frame_time is not None:
            return frame_time

    return fallback


def _collect_bvh_files(path):
    """Accept either one BVH file or recursively collect all BVHs under a directory."""
    resolved_path = os.fspath(path)
    if os.path.isfile(resolved_path):
        if not resolved_path.lower().endswith(".bvh"):
            raise ValueError(f"Expected a .bvh file, got: {resolved_path}")
        return [resolved_path]

    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    bvh_files = []
    for root, _, filenames in os.walk(resolved_path):
        for filename in filenames:
            if filename.lower().endswith(".bvh"):
                bvh_files.append(os.path.join(root, filename))

    bvh_files.sort()
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found under {resolved_path}")

    return bvh_files


def _resolve_clip_length_frames(clip_length_seconds, frame_time):
    """Convert a requested clip duration to an integer frame count."""
    if clip_length_seconds is None:
        return None

    clip_length_seconds = float(clip_length_seconds)
    if not np.isfinite(clip_length_seconds) or clip_length_seconds <= 0:
        raise ValueError("clip_length_seconds must be a positive number of seconds.")

    clip_length_frames = int(round(clip_length_seconds / frame_time))
    if clip_length_frames < 3:
        raise ValueError(
            "clip_length_seconds is too short for kinetic features at the current frame rate; "
            "need at least 3 frames per clip."
        )
    return clip_length_frames


def _compute_kinetic_feature_matrix_from_positions(
    global_positions,
    root_idx=0,
    dt=1.0 / 30.0,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
    clip_length_seconds=None,
):
    """Embed one motion as one feature vector or many fixed-length clip vectors."""
    positions = np.asarray(global_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "Expected global_positions with shape [num_frames, num_joints, 3]."
        )

    clip_length_frames = _resolve_clip_length_frames(clip_length_seconds, dt)
    if clip_length_frames is None:
        return compute_kinetic_feature_vector(
            positions,
            root_idx=root_idx,
            dt=dt,
            up_axis=up_axis,
            window_size=window_size,
            acceleration_mode=acceleration_mode,
        )[np.newaxis, :]

    feature_dim = positions.shape[1] * 3
    if positions.shape[0] < clip_length_frames:
        return np.empty((0, feature_dim), dtype=np.float64)

    feature_vectors = []
    for start_idx in range(0, positions.shape[0] - clip_length_frames + 1, clip_length_frames):
        clip_positions = positions[start_idx : start_idx + clip_length_frames]
        feature_vectors.append(
            compute_kinetic_feature_vector(
                clip_positions,
                root_idx=root_idx,
                dt=dt,
                up_axis=up_axis,
                window_size=window_size,
                acceleration_mode=acceleration_mode,
            )
        )

    if not feature_vectors:
        return np.empty((0, feature_dim), dtype=np.float64)

    return np.stack(feature_vectors, axis=0)


def compute_kinetic_feature_matrix_from_bvh(
    bvh_path,
    param=None,
    is_human=False,
    root_idx=0,
    frame_time=None,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
    clip_length_seconds=None,
):
    """Embed one BVH motion into one vector or many fixed-length clip vectors."""
    bvh, pos_all_joints = load_bvh_global_positions(bvh_path)
    dt = frame_time if frame_time is not None else _resolve_bvh_frame_time(bvh)
    return _compute_kinetic_feature_matrix_from_positions(
        pos_all_joints,
        root_idx=root_idx,
        dt=dt,
        up_axis=up_axis,
        window_size=window_size,
        acceleration_mode=acceleration_mode,
        clip_length_seconds=clip_length_seconds,
    )


def compute_kinetic_features_from_bvh(
    bvh_path,
    param=None,
    is_human=False,
    root_idx=0,
    frame_time=None,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
):
    """Embed one BVH motion into a single kinetic feature vector."""
    feature_matrix = compute_kinetic_feature_matrix_from_bvh(
        bvh_path,
        param=param,
        is_human=is_human,
        root_idx=root_idx,
        frame_time=frame_time,
        up_axis=up_axis,
        window_size=window_size,
        acceleration_mode=acceleration_mode,
    )
    return feature_matrix[0]


def compute_kinetic_feature_matrix_from_directory(
    directory,
    param=None,
    is_human=False,
    root_idx=0,
    frame_time=None,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
    clip_length_seconds=None,
):
    """Compute one kinetic feature vector per motion or per fixed-length clip."""
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Expected a dataset directory, got: {directory}")

    feature_matrices = []
    for bvh_path in _collect_bvh_files(directory):
        feature_matrix = compute_kinetic_feature_matrix_from_bvh(
            bvh_path,
            param=param,
            is_human=is_human,
            root_idx=root_idx,
            frame_time=frame_time,
            up_axis=up_axis,
            window_size=window_size,
            acceleration_mode=acceleration_mode,
            clip_length_seconds=clip_length_seconds,
        )
        if feature_matrix.shape[0] == 0:
            continue
        feature_matrices.append(feature_matrix)

    if not feature_matrices:
        raise ValueError(
            "No valid kinetic samples were extracted. Short motions may all be shorter than the requested clip length."
        )

    return np.concatenate(feature_matrices, axis=0)


def _resolve_generated_kinetic_features(
    generated_input,
    param=None,
    is_human=False,
    root_idx=0,
    frame_time=None,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
    clip_length_seconds=None,
):
    """Support generated input as either raw positions, one BVH, or a BVH directory."""
    if isinstance(generated_input, np.ndarray):
        dt = frame_time if frame_time is not None else (1.0 / 30.0)
        return _compute_kinetic_feature_matrix_from_positions(
            generated_input,
            root_idx=root_idx,
            dt=dt,
            up_axis=up_axis,
            window_size=window_size,
            acceleration_mode=acceleration_mode,
            clip_length_seconds=clip_length_seconds,
        )

    generated_path = os.fspath(generated_input)
    generated_files = _collect_bvh_files(generated_path)
    feature_matrices = []
    for bvh_path in generated_files:
        feature_matrix = compute_kinetic_feature_matrix_from_bvh(
            bvh_path,
            param=param,
            is_human=is_human,
            root_idx=root_idx,
            frame_time=frame_time,
            up_axis=up_axis,
            window_size=window_size,
            acceleration_mode=acceleration_mode,
            clip_length_seconds=clip_length_seconds,
        )
        if feature_matrix.shape[0] == 0:
            continue
        feature_matrices.append(feature_matrix)

    if not feature_matrices:
        raise ValueError(
            "No valid generated kinetic samples were extracted. Short motions may all be shorter than the requested clip length."
        )

    return np.concatenate(feature_matrices, axis=0)


def compute_kinetic_fid(
    generated_input,
    dataset_dir,
    param=None,
    is_human=False,
    root_idx=0,
    frame_time=None,
    up_axis="z",
    window_size=2,
    acceleration_mode="fairmotion",
    normalize_with_dataset_stats=False,
    clip_length_seconds=None,
):
    """Compute kinetic-feature FID between generated motions and a BVH dataset."""
    generated_features = _resolve_generated_kinetic_features(
        generated_input,
        param=param,
        is_human=is_human,
        root_idx=root_idx,
        frame_time=frame_time,
        up_axis=up_axis,
        window_size=window_size,
        acceleration_mode=acceleration_mode,
        clip_length_seconds=clip_length_seconds,
    )
    dataset_features = compute_kinetic_feature_matrix_from_directory(
        dataset_dir,
        param=param,
        is_human=is_human,
        root_idx=root_idx,
        frame_time=frame_time,
        up_axis=up_axis,
        window_size=window_size,
        acceleration_mode=acceleration_mode,
        clip_length_seconds=clip_length_seconds,
    )

    if generated_features.shape[1] != dataset_features.shape[1]:
        raise ValueError(
            "Generated and dataset kinetic features use different skeleton layouts: "
            f"generated_dim={generated_features.shape[1]}, dataset_dim={dataset_features.shape[1]}"
        )

    if normalize_with_dataset_stats:
        dataset_features, generated_features = _normalize_features_with_reference_stats(
            dataset_features,
            generated_features,
        )

    fid_score = compute_motion_fid(dataset_features, generated_features)
    result = {
        "fid": float(fid_score),
        "num_generated": int(generated_features.shape[0]),
        "num_dataset": int(dataset_features.shape[0]),
        "feature_dim": int(generated_features.shape[1]),
        "acceleration_mode": acceleration_mode,
        "up_axis": up_axis,
        "normalized": bool(normalize_with_dataset_stats),
        "clip_length_seconds": clip_length_seconds,
    }

    print("Kinetic FID report")
    print(f"  Generated samples: {result['num_generated']}")
    print(f"  Dataset samples:   {result['num_dataset']}")
    print(f"  Feature dim:       {result['feature_dim']}")
    print(f"  Accel mode:        {result['acceleration_mode']}")
    print(f"  Up axis:           {result['up_axis']}")
    print(f"  Normalized:        {result['normalized']}")
    print(
        "  Sample mode:       "
        + (
            "full motions"
            if result["clip_length_seconds"] is None
            else f"{result['clip_length_seconds']}s non-overlapping clips"
        )
    )
    print(f"  FID:               {result['fid']}")

    return result


__all__ = [
    "_normalize_features_with_reference_stats",
    "_resolve_generated_kinetic_features",
    "_resolve_bvh_frame_time",
    "compute_kinetic_feature_matrix_from_bvh",
    "compute_kinetic_feature_matrix_from_directory",
    "compute_kinetic_features_from_bvh",
    "compute_kinetic_fid",
    "compute_motion_fid",
    "load_bvh_global_positions",
]
import numpy as np
from . import quat
from . import dual_quat

"""
Ortho6D are represented as arrays of shape [..., 3, 2]
where the last dimension is the ortho6D representations,
which is the first two columns of the rotation matrix.
Matrix order: [[r0.x, r0.y],
               [r1.x, r1.y],
               [r2.x, r2.y]] where ri is row i.
"""


def from_quat(quaternions: np.array) -> np.array:
    """
    Convert quaternions to otho6D representation.

    Parameters
    ----------
        quat : np.array[..., [w,x,y,z]]

    Returns
    -------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.
    """
    return from_matrix(quat.to_matrix(quaternions))


def from_matrix(rotmats: np.array) -> np.array:
    """
    Convert rotation matrices to ortho6D representation.

    Parameters
    ----------
        rotmats : np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                      [r1.x, r1.y, r1.z],
                                                      [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.
    """
    return rotmats[..., :2]


def to_quat(ortho6D: np.array) -> np.array:
    """
    Convert ortho6D to quaternions.

    Parameters
    ----------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.

    Returns
    -------
        quat : np.array[..., [w,x,y,z]]
    """
    return quat.from_matrix(to_matrix(ortho6D))


def to_matrix(ortho6D: np.array) -> np.array:
    """
    Convert ortho6D to rotation matrices.

    Parameters
    ----------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.

    Returns
    -------
        rotmats : np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                      [r1.x, r1.y, r1.z],
                                                      [r2.x, r2.y, r2.z]] where ri is row i.
    """
    c1 = ortho6D[..., 0] / np.linalg.norm(ortho6D[..., 0], axis=-1, keepdims=True)
    c2 = ortho6D[..., 1] - np.sum(c1 * ortho6D[..., 1], axis=-1)[..., np.newaxis] * c1
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2, axis=-1)
    rotations = np.concatenate([c1, c2, c3], axis=-1).reshape(*ortho6D.shape[:-2], 3, 3)
    # transpose last 2 axis
    rotations = np.swapaxes(rotations, -2, -1)
    return rotations


def from_dual_quat(dq: np.array) -> np.array:
    """
    Convert dual quaternion(s) to a 6D+translation vector of shape [..., 9].

    The first 6 entries are the ortho6D rotation columns flattened as
    [r0.x, r0.y, r1.x, r1.y, r2.x, r2.y], followed by the 3D translation.

    """
    dq = np.asarray(dq)
    # reuse existing helper to get rotations (quaternions) and translations
    rotations, translations = dual_quat.to_rotation_translation(dq)  # rotations [...,4], translations [...,3]

    # convert rotations to rotation matrices and take first two columns (ortho6D)
    rotmats = quat.to_matrix(rotations)  # [..., 3, 3]
    ortho6 = rotmats[..., :2]            # [..., 3, 2]

    # flatten the 3x2 ortho6 into 6 values per element
    ortho6_flat = ortho6.reshape(*ortho6.shape[:-2], 6)  # [..., 6]

    # concatenate 6 rotation values with 3 translation values -> [..., 9]
    return np.concatenate((ortho6_flat, translations), axis=-1)

def to_dual_quat(ortho6_flat: np.array) -> np.array:
    """
    Convert flattened ortho6+translation ([...,9]) to dual quaternions ([...,8]).

    Parameters
    ----------
    ortho6_flat : np.array[..., 9]
        Flattened ortho6 columns + translation: [r0.x, r0.y, r1.x, r1.y, r2.x, r2.y, tx, ty, tz]
    
    Returns
    -------
    dq : np.array[..., 8]
        Dual quaternions [..., [w_r,x_r,y_r,z_r, w_d,x_d,y_d,z_d]]
    """
    # expect 9 channels: 6 ortho + 3 translation
    if ortho6_flat.shape[-1] != 9:
        raise ValueError("to_dual_quat expects last dimension size 9 (6 ortho + 3 translation)")

    # split ortho6 and translations
    ortho6_vals = ortho6_flat[..., :6]
    translations = ortho6_flat[..., 6:]

    # reshape flattened ortho6 into [..., 3, 2]
    ortho6 = np.reshape(ortho6_vals, (*ortho6_vals.shape[:-1], 3, 2))
    # get quaternion rotations from ortho6
    rotations = to_quat(ortho6)  # [`rotations.ortho6d.to_quat`](rotations/ortho6d.py)
    # build dual quaternions from rotations and translations
    dq = dual_quat.from_rotation_translation(rotations, translations)  # [`rotations.dual_quat.from_rotation_translation`](rotations/dual_quat.py)
    dq = dual_quat.canonicalize(dq)
    dq = dual_quat.unroll(dq,axis=-3)
    
    return dq

def normalize(ortho6_flat: np.ndarray, translation_scale: float = None) -> np.ndarray:
    """
    Normalize raw network output of shape [...,6] or [...,9].

    - If input has 6 channels: treat as flattened ortho6 ([...,6]) and return normalized [...,6].
    - If input has 9 channels: treat as ortho6+translation ([...,9]) and return normalized [...,9].
    - Orthonormalization is done via [`rotations.ortho6d.to_matrix`](rotations/ortho6d.py)
      + [`rotations.ortho6d.from_matrix`](rotations/ortho6d.py).
    """
    if ortho6_flat.shape[-1] == 9:
        ortho6_vals = ortho6_flat[..., :6]
        translations = ortho6_flat[..., 6:]
    elif ortho6_flat.shape[-1] == 6:
        ortho6_vals = ortho6_flat
        translations = None
    else:
        raise ValueError("normalize expects last dimension size 6 (ortho) or 9 (ortho + translation)")

    # Reshape to [..., 3, 2] and normalize via to_matrix->from_matrix
    ortho6 = np.reshape(ortho6_vals, (*ortho6_vals.shape[:-1], 3, 2))
    rotmats = to_matrix(ortho6)  # [`rotations.ortho6d.to_matrix`](rotations/ortho6d.py)
    ortho6_norm = from_matrix(rotmats)  # [`rotations.ortho6d.from_matrix`](rotations/ortho6d.py)
    ortho6_flat_norm = ortho6_norm.reshape(*ortho6_vals.shape[:-1], 6)

    if translations is None:
        return ortho6_flat_norm

    if translation_scale is not None:
        translations = translations * translation_scale

    return np.concatenate((ortho6_flat_norm, translations), axis=-1)

def compute_incremental(ortho6D: np.array) -> np.array:
    """
    Compute incremental rotations directly in 6D (ortho6D) form.

    Accepts input in [...,3,2] or flattened [...,6]. If input is flattened, output is flattened.
    """
    ortho6D = np.asarray(ortho6D)
    input_was_flat = (ortho6D.shape[-1] == 6 and ortho6D.ndim >= 1)
    # full rotation matrices [..., 3, 3]
    if input_was_flat:
        ortho6D_reshaped = ortho6D.reshape(*ortho6D.shape[:-1], 3, 2)
    else:
        ortho6D_reshaped = ortho6D
    rotmats = to_matrix(ortho6D_reshaped)

    # shift previous frames: prev[0] = rotmats[0], prev[i] = rotmats[i-1]
    prev = np.concatenate([rotmats[0:1], rotmats[:-1]], axis=0)
    # inverse (transpose for rotation matrices)
    prev_inv = np.swapaxes(prev, -2, -1)
    # incremental rotations: prev_inv @ cur
    rot_inc = np.matmul(prev_inv, rotmats)

    ortho_inc = from_matrix(rot_inc)  # [...,3,2]
    # if input_was_flat:
    return ortho_inc.reshape(*ortho_inc.shape[:-2], 6)
    # return ortho_inc


def compute_cumulative(increments: np.array) -> np.array:
    """
    Reconstruct cumulative (absolute) ortho6D rotations from incremental ortho6D increments.

    Accepts increments in [...,3,2] or flattened [...,6]. If input is flattened, output is flattened.
    """
    increments = np.asarray(increments)
    input_was_flat = (increments.shape[-1] == 6 and increments.ndim >= 1)
    if input_was_flat:
        inc_reshaped = increments.reshape(*increments.shape[:-1], 3, 2)
    else:
        inc_reshaped = increments

    rot_inc = to_matrix(inc_reshaped)  # [...,3,3]

    rot_cum = np.empty_like(rot_inc)
    rot_cum[0] = rot_inc[0]
    for i in range(1, rot_inc.shape[0]):
        rot_cum[i] = np.matmul(rot_cum[i - 1], rot_inc[i])

    ortho_cum = from_matrix(rot_cum)  # [...,3,2]
    # if input_was_flat:
    return ortho_cum.reshape(*ortho_cum.shape[:-2], 6)
    # return ortho_cum
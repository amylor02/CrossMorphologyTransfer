import torch
from torch.nn import functional as F
from . import quat_torch as quat
from . import dual_quat_torch as dual_quat

"""
Ortho6D are represented as tensors of shape [..., 3, 2]
where the last dimension is the ortho6D representations,
which is the first two columns of the rotation matrix.
Matrix order: [[r0.x, r0.y],
               [r1.x, r1.y],
               [r2.x, r2.y]] where ri is row i.
"""


def from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to otho6D representation.

    Parameters
    ----------
        quat : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.
    """
    return from_matrix(quat.to_matrix(quaternions))


def from_matrix(rotmats: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to ortho6D representation.

    Parameters
    ----------
        rotmats : torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                          [r1.x, r1.y, r1.z],
                                                          [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.
    """
    return rotmats[..., :2]


def to_quat(ortho6D: torch.Tensor) -> torch.Tensor:
    """
    Convert ortho6D to quaternions.

    Parameters
    ----------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.
    Returns
    -------
        quat : torch.Tensor[..., [w,x,y,z]]
    """

    return quat.from_matrix(to_matrix(ortho6D))


def to_matrix(ortho6D: torch.Tensor) -> torch.Tensor:
    """
    Convert ortho6D to rotation matrices.

    Parameters
    ----------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.

    Returns
    -------
        rotmats : torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                          [r1.x, r1.y, r1.z],
                                                          [r2.x, r2.y, r2.z]] where ri is row i.
    """
    c1 = F.normalize(ortho6D[..., 0], p=2, dim=-1)
    c2 = F.normalize(
        ortho6D[..., 1] - (c1 * ortho6D[..., 1]).sum(-1).unsqueeze(-1) * c1,
        p=2,
        dim=-1,
    )
    c3 = torch.cross(c1, c2, dim=-1)
    rotations = (
        torch.cat([c1, c2, c3], dim=-1)
        .reshape(*ortho6D.shape[:-2], 3, 3)
        .transpose(-2, -1)
    )
    return rotations


def from_dual_quat(dq: torch.Tensor) -> torch.Tensor:
    """
    Convert dual quaternion(s) to a 6D+translation vector of shape [..., 9] (torch).

    The first 6 entries are the ortho6D rotation columns flattened as
    [r0.x, r0.y, r1.x, r1.y, r2.x, r2.y], followed by the 3D translation.
    """
    dq = dq.clone()
    # reuse existing helper to get rotations (quaternions) and translations
    rotations, translations = dual_quat.to_rotation_translation(dq)  # [`rotations.dual_quat_torch.to_rotation_translation`](rotations/dual_quat_torch.py)

    # convert rotations to rotation matrices and take first two columns (ortho6D)
    rotmats = quat.to_matrix(rotations)  # [`rotations.quat_torch.to_matrix`](rotations/quat_torch.py)
    ortho6 = rotmats[..., :2]            # [..., 3, 2]

    # flatten the 3x2 ortho6 into 6 values per element
    ortho6_flat = ortho6.reshape(*ortho6.shape[:-2], 6)  # [..., 6]

    # concatenate 6 rotation values with 3 translation values -> [..., 9]
    return torch.cat((ortho6_flat, translations), dim=-1)



def to_dual_quat(ortho6_flat: torch.Tensor) -> torch.Tensor:
    """
    Convert flattened ortho6+translation ([...,9]) to dual quaternions ([...,8]) (torch).

    Parameters
    ----------
    ortho6_flat : torch.Tensor[..., 9]
        Flattened ortho6 columns + translation: [r0.x, r0.y, r1.x, r1.y, r2.x, r2.y, tx, ty, tz]

    Returns
    -------
    dq : torch.Tensor[..., 8]
        Dual quaternions [..., [w_r,x_r,y_r,z_r, w_d,x_d,y_d,z_d]]
    """
    # expect 9 channels: 6 ortho + 3 translation
    if ortho6_flat.shape[-1] != 9:
        raise ValueError("to_dual_quat expects last dimension size 9 (6 ortho + 3 translation)")

    # split ortho6 and translations
    ortho6_vals = ortho6_flat[..., :6]
    translations = ortho6_flat[..., 6:]

    # reshape flattened ortho6 into [..., 3, 2]
    ortho6 = ortho6_vals.reshape(*ortho6_vals.shape[:-1], 3, 2)
    # get quaternion rotations from ortho6
    rotations = to_quat(ortho6)  # [`rotations.ortho6d_torch.to_quat`](rotations/ortho6d_torch.py)
    # build dual quaternions from rotations and translations
    dq = dual_quat.from_rotation_translation(rotations, translations)  # [`rotations.dual_quat_torch.from_rotation_translation`](rotations/dual_quat_torch.py)

    # apply canonicalization and continuity unroll (match numpy behavior)
    dq = dual_quat.canonicalize(dq)   # [`rotations.dual_quat_torch.canonicalize`](rotations/dual_quat_torch.py)
    dq = dual_quat.unroll(dq,dim=-3)     # [`rotations.dual_quat_torch.unroll`](rotations/dual_quat_torch.py)

    return dq

def normalize(ortho6_flat: torch.Tensor, translation_scale: float = None) -> torch.Tensor:
    """
    Normalize raw network output of shape [...,6] or [...,9] (torch).

    - If input has 6 channels: treat as flattened ortho6 ([...,6]) and return normalized [...,6].
    - If input has 9 channels: treat as ortho6+translation ([...,9]) and return normalized [...,9].
    - Orthonormalization is done via [`rotations.ortho6d_torch.to_matrix`](rotations/ortho6d_torch.py)
      + [`rotations.ortho6d_torch.from_matrix`](rotations/ortho6d_torch.py).
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
    ortho6 = ortho6_vals.reshape(*ortho6_vals.shape[:-1], 3, 2)
    rotmats = to_matrix(ortho6)  # [`rotations.ortho6d_torch.to_matrix`](rotations/ortho6d_torch.py)
    ortho6_norm = from_matrix(rotmats)  # [`rotations.ortho6d_torch.from_matrix`](rotations/ortho6d_torch.py)
    ortho6_flat_norm = ortho6_norm.reshape(*ortho6_vals.shape[:-1], 6)

    if translations is None:
        return ortho6_flat_norm

    if translation_scale is not None:
        translations = translations * translation_scale

    return torch.cat((ortho6_flat_norm, translations), dim=-1)


def compute_incremental(ortho6D: torch.Tensor) -> torch.Tensor:
    """
    Compute incremental rotations directly in 6D (ortho6D) form (torch).

    Accepts input in [...,3,2] or flattened [...,6]. If input is flattened, output is flattened.
    Time / frames axis is expected to be the first axis (axis 0), matching the numpy version.
    """
    ortho6D = ortho6D.clone()
    input_was_flat = (ortho6D.shape[-1] == 6 and ortho6D.ndim >= 1)
    if input_was_flat:
        ortho6D = ortho6D.reshape(*ortho6D.shape[:-1], 3, 2)

    rotmats = to_matrix(ortho6D)  # [`rotations.ortho6d_torch.to_matrix`](rotations/ortho6d_torch.py)

    # shift previous frames: prev[0] = rotmats[0], prev[i] = rotmats[i-1]
    prev = torch.cat([rotmats[0:1], rotmats[:-1]], dim=0)
    # inverse (transpose for rotation matrices)
    prev_inv = prev.transpose(-2, -1)
    # incremental rotations: prev_inv @ cur
    rot_inc = torch.matmul(prev_inv, rotmats)

    ortho_inc = from_matrix(rot_inc)  # [`rotations.ortho6d_torch.from_matrix`](rotations/ortho6d_torch.py)
    # if input_was_flat:
    return ortho_inc.reshape(*ortho_inc.shape[:-2], 6)
    # return ortho_inc


def compute_cumulative(increments: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct cumulative (absolute) ortho6D rotations from incremental ortho6D increments (torch).

    Accepts increments in [...,3,2] or flattened [...,6]. If input is flattened, output is flattened.
    Time / frames axis is expected to be the first axis (axis 0), matching the numpy version.
    """
    increments = increments.clone()
    input_was_flat = (increments.shape[-1] == 6 and increments.ndim >= 1)
    if input_was_flat:
        inc_reshaped = increments.reshape(*increments.shape[:-1], 3, 2)
    else:
        inc_reshaped = increments

    rot_inc = to_matrix(inc_reshaped)  # [`rotations.ortho6d_torch.to_matrix`](rotations/ortho6d_torch.py)

    rot_cum = torch.empty_like(rot_inc)
    rot_cum[0] = rot_inc[0]
    for i in range(1, rot_inc.shape[0]):
        rot_cum[i] = torch.matmul(rot_cum[i - 1], rot_inc[i])

    ortho_cum = from_matrix(rot_cum)  # [`rotations.ortho6d_torch.from_matrix`](rotations/ortho6d_torch.py)
    # if input_was_flat:
    return ortho_cum.reshape(*ortho_cum.shape[:-2], 6)
    # return ortho_cum
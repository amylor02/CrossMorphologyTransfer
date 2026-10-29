import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
def from_scaled_angle_axis(scaledaxis: np.array) -> np.array:
    """
    Create a quaternion from an scaled angle-axis representation.

    Parameters
    ----------
    scaledaxis : np.array[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    angle = np.linalg.norm(scaledaxis, axis=-1)[..., np.newaxis]
    axis = scaledaxis / angle
    return from_angle_axis(angle, axis)


def from_angle_axis(angle: np.array, axis: np.array) -> np.array:
    """
    Create a quaternion from an angle-axis representation.

    Parameters
    ----------
    angle : np.array[..., angle]
    axis : np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    
    return np.concatenate((c, s * axis), axis=-1)


def from_euler(euler: np.array, order: np.array) -> np.array:
    """
    Create a quaternion from an euler representation with a specified order.

    Parameters
    ----------
    euler : np.array[..., [e0, e1, e2]]
        euler angles in radians
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """

    assert (
        euler.shape[:-1] == order.shape[:-1]
    ), "euler and order must have the same shape except for the last dimension"

    axis = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    q0 = from_angle_axis(
        euler[..., 0:1],
        np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 0:1]),
    )
    q1 = from_angle_axis(
        euler[..., 1:2],
        np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 1:2]),
    )
    q2 = from_angle_axis(
        euler[..., 2:3],
        np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 2:3]),
    )
    return mul(q0, mul(q1, q2))


def from_matrix(rotmats: np.array) -> np.array:
    """
    Convert rotation matrices to quaternions.

    Parameters
    ----------
        rotmats: np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                     [r1.x, r1.y, r1.z],
                                                     [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        quat np.array[..., [w,x,y,z]]
    """
    # Separate components
    r0c0 = rotmats[..., 0, 0]
    r0c1 = rotmats[..., 0, 1]
    r0c2 = rotmats[..., 0, 2]
    r1c0 = rotmats[..., 1, 0]
    r1c1 = rotmats[..., 1, 1]
    r1c2 = rotmats[..., 1, 2]
    r2c0 = rotmats[..., 2, 0]
    r2c1 = rotmats[..., 2, 1]
    r2c2 = rotmats[..., 2, 2]

    return normalize(
        np.where(
            (r2c2 < 0.0)[..., np.newaxis],
            np.where(
                (r0c0 > r1c1)[..., np.newaxis],
                np.concatenate(
                    [
                        (r2c1 - r1c2)[..., np.newaxis],
                        (1.0 + r0c0 - r1c1 - r2c2)[..., np.newaxis],
                        (r1c0 + r0c1)[..., np.newaxis],
                        (r0c2 + r2c0)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
                np.concatenate(
                    [
                        (r0c2 - r2c0)[..., np.newaxis],
                        (r1c0 + r0c1)[..., np.newaxis],
                        (1.0 - r0c0 + r1c1 - r2c2)[..., np.newaxis],
                        (r2c1 + r1c2)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
            ),
            np.where(
                (r0c0 < -r1c1)[..., np.newaxis],
                np.concatenate(
                    [
                        (r1c0 - r0c1)[..., np.newaxis],
                        (r0c2 + r2c0)[..., np.newaxis],
                        (r2c1 + r1c2)[..., np.newaxis],
                        (1.0 - r0c0 - r1c1 + r2c2)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
                np.concatenate(
                    [
                        (1.0 + r0c0 + r1c1 + r2c2)[..., np.newaxis],
                        (r2c1 - r1c2)[..., np.newaxis],
                        (r0c2 - r2c0)[..., np.newaxis],
                        (r1c0 - r0c1)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
            ),
        )
    )


def to_euler(quaternions: np.array, order: np.array) -> np.array:
    """
    Convert a quaternion to an intrinsic euler representation with a specified order.
    Does not detect/solve gimbal lock.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    euler : np.array[..., 3]
        euler angles in radians
    """

    assert (
        quaternions.shape[:-1] == order.shape[:-1]
    ), "quaternions and order must have the same shape except for the last dimension"

    aux = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    angle_first = 2
    angle_third = 0

    i = np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 2:3])[
        ..., np.newaxis
    ]
    j = np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 1:2])[
        ..., np.newaxis
    ]
    k = np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 0:1])[
        ..., np.newaxis
    ]

    # check if permutation is even or odd
    sign = (i - j) * (j - k) * (k - i) // 2

    # euler angles
    euler = np.empty(quaternions.shape[:-1] + (3,))

    # permutate quaternion elements
    a = quaternions[..., 0:1] - np.take_along_axis(quaternions, j + 1, axis=-1)
    b = (
        np.take_along_axis(quaternions, i + 1, axis=-1)
        + np.take_along_axis(quaternions, k + 1, axis=-1) * sign
    )
    c = np.take_along_axis(quaternions, j + 1, axis=-1) + quaternions[..., 0:1]
    d = np.take_along_axis(quaternions, k + 1, axis=-1) * sign - np.take_along_axis(
        quaternions, i + 1, axis=-1
    )

    # compute second angle
    euler[..., 1:2] = (2 * np.arctan2(np.hypot(c, d), np.hypot(a, b))) - (np.pi / 2)

    # compute first and third angle
    half_sum = np.arctan2(b, a)
    half_diff = np.arctan2(d, c)
    euler[..., angle_first : angle_first + 1] = half_sum - half_diff
    euler[..., angle_third : angle_third + 1] = (half_sum + half_diff) * sign

    # for i in range(3):
    # if euler[..., i] < -np.pi:
    #    euler[..., i] += 2 * np.pi
    # elif euler[..., i] > np.pi:
    #    euler[..., i] -= 2 * np.pi
    euler = np.mod(euler, 2 * np.pi)

    return euler


def to_scaled_angle_axis(quaternions: np.array) -> np.array:
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    scaledaxis : np.array[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation
    """
    angle, axis = to_angle_axis(quaternions)
    return angle * axis


def to_angle_axis(quaternions: np.array) -> np.array:
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    angle: np.array[..., angle]
    axis : np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation
    """
    q = quaternions
    angle = 2 * np.arccos(q[..., 0:1])
    s = np.sqrt(1 - q[..., 0:1] * q[..., 0:1])
    return angle, q[..., 1:] / s


def to_matrix(quaternions: np.array) -> np.array:
    """
    Convert rotations given as quaternions to rotation matrices.
    Parameters
    ----------
        quaternions: np.array[..., [w,x,y,z]]
    Returns
    -------
        rotmats: np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                     [r1.x, r1.y, r1.z],
                                                     [r2.x, r2.y, r2.z]] where ri is row i.
    """
    qw = quaternions[..., 0]
    qx = quaternions[..., 1]
    qy = quaternions[..., 2]
    qz = quaternions[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = np.empty(quaternions.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def mul_vec(q: np.array, v: np.array) -> np.array:
    """
    Multiply a vector by a quaternion

    Parameters
    ----------
    q : np.array[..., [w,x,y,z]]
    v : np.array[..., [x,y,z]]

    Returns
    -------
    v: np.array[..., [x,y,z]]
    """
    t = 2.0 * _fast_cross(q[..., 1:], v)
    return v + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)

def compute_angular_velocity(quaternion_tensor, dt):
    device = quaternion_tensor.device
    batch_size, _, num_frames = quaternion_tensor.shape
    
    # Ensure tensor is on the GPU
    quaternion_tensor = quaternion_tensor.to(device)
    
    # Get all quaternions
    q1 = quaternion_tensor[:, :, :-1]  # [batch_size, 4, num_frames - 1]
    q2 = quaternion_tensor[:, :, 1:]   # [batch_size, 4, num_frames - 1]

    # Compute the components of angular velocity using batch operations
    q1_w, q1_x, q1_y, q1_z = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    q2_w, q2_x, q2_y, q2_z = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    # Calculate components of angular velocity
    w_x = 2 * (q1_w * q2_x - q1_x * q2_w - q1_y * q2_z + q1_z * q2_y) / dt
    w_y = 2 * (q1_w * q2_y + q1_x * q2_z - q1_y * q2_w - q1_z * q2_x) / dt
    w_z = 2 * (q1_w * q2_z - q1_x * q2_y + q1_y * q2_x - q1_z * q2_w) / dt

    # Stack to get final angular velocity tensor
    angular_velocity = torch.stack([w_x, w_y, w_z], dim=1)  # [batch_size, 3, num_frames - 1]

    return angular_velocity

def mul(q0: np.array, q1: np.array) -> np.array:
    """
    Multiply two quaternions.

    Parameters
    ----------
    q0 : np.array[..., [w,x,y,z]]
    q1 : np.array[..., [w,x,y,z]]

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    w0, x0, y0, z0 = q0[..., 0:1], q0[..., 1:2], q0[..., 2:3], q0[..., 3:4]
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    # (w0,v0)(w1,v1) = (w0w1 - v0·v1, w0v1 + w1v0 + v0 x v1)
    return np.concatenate(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,  # w
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,  # x
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,  # y
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,  # z
        ),
        axis=-1,
    )


def length(quaternions: np.array) -> np.array:
    """
    Get the length or magnitude of the quaternions.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    length : np.array[...]
    """
    return np.linalg.norm(quaternions, axis=-1)


def inverse(quaternions: np.array) -> np.array:
    """
    Inverse of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    # for a unit quaternion the conjugate is the inverse
    # q^-1 = [q0, -q1, -q2, -q3]
    return conjugate(quaternions)


def conjugate(quaternions: np.array) -> np.array:
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return np.concatenate((quaternions[..., 0:1], -quaternions[..., 1:]), axis=-1)


def normalize(quaternions: np.array, eps: float = 1e-8) -> np.array:
    """
    Convert all quaternions to unit quatenrions.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return quaternions / (length(quaternions)[..., np.newaxis] + eps)


def canonicalize_(q: np.ndarray) -> np.ndarray:
    """
    Make quaternion representation canonical per element/frame by ensuring the real
    quaternion scalar (w) is non-negative. If w < 0, flip the whole quaternion (q -> -q).
    Works elementwise across leading dims.
    """
    q = q.copy()
    # scalar part of real quaternion
    w = q[..., 0]
    mask = w < 0  # boolean mask over all leading dims
    if mask.any():
        # expand mask to full last-dim and select-flip
        mask_exp = np.expand_dims(mask, -1)
        mask_exp = np.broadcast_to(mask_exp, q.shape)
        q = np.where(mask_exp, -q, q)
    return q

def unroll(quaternions: np.array, axis: int) -> np.array:
    """
    Avoid the quaternion 'double cover' problem by picking the cover
    of the first quaternion, and then removing sudden switches
    over the cover by ensuring that each frame uses the quaternion
    closest to the one of the previous frame.

    ('double cover': same rotation can be encoded with two
    different quaternions)

    Usage example: Ensuring an animation to have quaternions
    that represent the 'shortest' rotation path. Otherwise,
    if we SLERP between poses we would get joints rotating in
    the "longest" path.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]
    axis : int
        unroll axis (e.g., frames axis)

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    r = quaternions.swapaxes(0, axis)
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, r.shape[0]):
        # distance (dot product) between the previous and current quaternion
        d0 = np.sum(r[i] * r[i - 1], axis=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = np.sum(-r[i] * r[i - 1], axis=-1)
        # if the distance with the flipped quaternion is smaller, use it
        r[i][d0 < d1] = -r[i][d0 < d1]
    r = r.swapaxes(0, axis)
    return r


def _fast_cross(a: np.array, b: np.array) -> np.array:
    """
    Fast cross of two vectors

    Parameters
    ----------
    a : np.array[..., [x,y,z]]
    b : np.array[..., [x,y,z]]

    Returns
    -------
    np.array[..., [x,y,z]]
    """

    return np.concatenate(
        [
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ],
        axis=-1,
    )



def compute_incremental_quaternions(quaternions):
    """
    Compute incremental quaternions from a sequence of quaternions (root only)
    
    Parameters
    ----------
    quaternions : np.array[#frames, #joints, [w, x, y, z]]
        Array of quaternions representing the rotation at each frame.
    
    Returns
    -------
    increments : np.array[..., [w, x, y, z]]
        Incremental quaternions representing the relative rotation between consecutive frames.
    """
    incremental_quats = quaternions.copy()
    incremental_quats[1:,0,:] = mul(inverse(quaternions[:-1,0,:]), quaternions[1:,0,:])
    incremental_quats[0,0,:] = np.array([1,0,0,0])
    
    return incremental_quats

def compute_incremental_quaternions_with_feet(quaternions, foot_1_idx=15, foot_2_idx=18):
    """
    Compute incremental quaternions from a sequence of quaternions.
    (root + feet)
    
    Parameters
    ----------
    quaternions : np.array[#frames, #joints, [w, x, y, z]]
        Array of quaternions representing the rotation at each frame.
    
    Returns
    -------
    increments : np.array[..., [w, x, y, z]]
        Incremental quaternions representing the relative rotation between consecutive frames.
    """
    incremental_quats = quaternions.copy()
    incremental_quats[1:,0,:] = mul(inverse(quaternions[:-1,0,:]), quaternions[1:,0,:])
    incremental_quats[1:,foot_1_idx,:] = mul(inverse(quaternions[:-1,foot_1_idx,:]), quaternions[1:,foot_1_idx,:])
    incremental_quats[1:,foot_2_idx,:] = mul(inverse(quaternions[:-1,foot_2_idx,:]), quaternions[1:,foot_2_idx,:])
    incremental_quats[0,0,:] = np.array([1,0,0,0])
    incremental_quats[0,foot_1_idx,:] = np.array([1,0,0,0])
    incremental_quats[0,foot_2_idx,:] = np.array([1,0,0,0])
    
    return incremental_quats

def compute_incremental_intermittent_quaternions(quaternions):#, pos):
    """
    Compute incremental quaternions from a sequence of quaternions.
    
    Parameters
    ----------
    quaternions : np.array[#frames, #joints, [w, x, y, z]]
        Array of quaternions representing the rotation at each frame.
    pos : np.array[#frames, [x,y,z]]
        Array of position of root joint at each frame.
    Returns
    -------
    increments : np.array[..., [w, x, y, z]]
        Incremental quaternions representing the relative rotation between consecutive frames.
    """
    frame_step = 8
    foot_1_idx = 15
    foot_2_idx = 18
    incremental_quats = quaternions.copy()
    incremental_quats[:,0,:] = 0.0 # set root to zeros
    incremental_quats[:,foot_1_idx,:] = 0.0
    incremental_quats[:,foot_2_idx,:] = 0.0
    incremental_quats[0,0,:] = quaternions[0,0,:]
    #incremental_quats[0,foot_1_idx,:] = quaternions[0,foot_1_idx,:]
    #incremental_quats[0,foot_2_idx,:] = quaternions[0,foot_2_idx,:]
    
    #incremental_pos = np.zeros_like(pos)
    
    for i in range(frame_step, len(quaternions), frame_step):
        incremental_quats[i,0,:] = mul(inverse(quaternions[i-frame_step,0,:]), quaternions[i,0,:])
        #incremental_quats[i,foot_1_idx,:] = mul(inverse(quaternions[i-frame_step,foot_1_idx,:]), quaternions[i,foot_1_idx,:])
        #incremental_quats[i,foot_2_idx,:] = mul(inverse(quaternions[i-frame_step,foot_2_idx,:]), quaternions[i,foot_2_idx,:])
        #incremental_pos[i, :] = pos[i, :] - pos[i-frame_step, :]
        
    incremental_quats[:,:,1] = 0.0 # zero-out x
    incremental_quats[:,:,3] = 0.0 # zero-out z
    
    #increments = np.concatenate([incremental_quats[:, 0, :],incremental_quats[:,foot_1_idx,:],incremental_quats[:,foot_2_idx,:], incremental_pos], axis=-1)
    #increments = np.concatenate([incremental_quats[:, 0, :], incremental_pos], axis=-1)
    return incremental_quats


def compute_filtered_quaternions(quaternions, pos):
    
    foot_1_idx = 15
    foot_2_idx = 18
    sigma = 60
    filtered_quats = quaternions.copy()
    
    filtered_quats[:,0,0] = gaussian_filter1d(filtered_quats[:,0,0],sigma=sigma)
    filtered_quats[:,0,1] = 0.0 #gaussian_filter1d(filtered_quats[:,0,1],sigma=sigma)
    filtered_quats[:,0,2] = gaussian_filter1d(filtered_quats[:,0,2],sigma=sigma)
    filtered_quats[:,0,3] = 0.0 #gaussian_filter1d(filtered_quats[:,0,3],sigma=sigma)
    
    filtered_quats[:,foot_1_idx,0] = gaussian_filter1d(filtered_quats[:,foot_1_idx,0],sigma=sigma)
    filtered_quats[:,foot_1_idx,1] = gaussian_filter1d(filtered_quats[:,foot_1_idx,1],sigma=sigma)
    filtered_quats[:,foot_1_idx,2] = gaussian_filter1d(filtered_quats[:,foot_1_idx,2],sigma=sigma)
    filtered_quats[:,foot_1_idx,3] = gaussian_filter1d(filtered_quats[:,foot_1_idx,3],sigma=sigma)
    
    filtered_quats[:,foot_2_idx,0] = gaussian_filter1d(filtered_quats[:,foot_2_idx,0],sigma=sigma)
    filtered_quats[:,foot_2_idx,1] = gaussian_filter1d(filtered_quats[:,foot_2_idx,1],sigma=sigma)
    filtered_quats[:,foot_2_idx,2] = gaussian_filter1d(filtered_quats[:,foot_2_idx,2],sigma=sigma)
    filtered_quats[:,foot_2_idx,3] = gaussian_filter1d(filtered_quats[:,foot_2_idx,3],sigma=sigma)
    
    displacement = np.concatenate([np.array([[0,0,0]]),pos[1:]-pos[:-1]],axis=0)
    displacement[:,0] = gaussian_filter1d(displacement[:,0],sigma=sigma)
    displacement[:,1] = gaussian_filter1d(displacement[:,1],sigma=sigma)
    displacement[:,2] = gaussian_filter1d(displacement[:,2],sigma=sigma)
    filtered_pos = displacement
    
    #filtered = np.concatenate([filtered_quats[:,0,:],filtered_quats[:,foot_1_idx,:],filtered_quats[:,foot_2_idx,:],filtered_pos],axis=-1)
    filtered = np.concatenate([filtered_quats[:,0,:],filtered_pos],axis=-1)
    return filtered
    
    

def compute_cumulative_quaternions(increments, feet_idx = None):
    """
    Computes cumulative quaternions from incremental quaternions, starting with the initial quaternion.

    Parameters
    ----------
    increments : np.array[..., [w,x,y,z]]
        Incremental quaternions representing the rotation difference between frames, with the 
        first quaternion being the initial starting frame.

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
        Cumulative quaternions for each frame.
    """

    quaternions = increments.copy()
    # quaternions[0, 0, :] = np.array([1,0,0,0])
    if(feet_idx is not None):
        for i in range(1, increments.shape[0]):
            quaternions[i,0,:] = mul(quaternions[i-1,0,:], increments[i,0,:])
            quaternions[i,feet_idx[0],:] = mul(quaternions[i-1,feet_idx[0],:], increments[i,feet_idx[0],:])
            quaternions[i,feet_idx[1],:] = mul(quaternions[i-1,feet_idx[1],:], increments[i,feet_idx[1],:])
    else:
        for i in range(1, increments.shape[0]):
            quaternions[i,0,:] = mul(quaternions[i-1,0,:], increments[i,0,:])
            
    return quaternions

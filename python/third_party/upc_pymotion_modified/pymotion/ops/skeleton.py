import numpy as np
import pymotion.rotations.quat as quat
import pymotion.rotations.dual_quat as dquat
import pymotion.rotations.quat_torch as quat_torch
import pymotion.rotations.dual_quat_torch as dquat_torch
import torch.nn.functional as F
import torch
"""
A skeleton is a set of joints connected by bones.
The skeleton is defined by:
    - the local offsets of the joints
    - the parents of the joints
    - the local rotations of the joints
    - the global position of the root joint

This functions convert skeletal information to 
root-centered dual quaternions and vice-versa.

Root-centered dual quaternions are useful when training
neural networks, as all information is local to the root
and the neural network does not need to learn the FK function.

"""


def from_root_dual_quat_torch(dq: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    Convert root-centered dual quaternion to the skeleton information in PyTorch.

    Parameters
    ----------
    dq: torch.Tensor[..., n_joints, 8]
        Dual quaternion tensor, where the first element includes the global position of the root joint.
    parents: torch.Tensor[n_joints]
        Parent indices for each joint.

    Returns
    -------
    rotations : torch.Tensor[..., n_joints, 4]
    translations : torch.Tensor[..., n_joints, 3]
    """
    n_joints = dq.shape[-2]
    
    # Convert dual quaternions to rotations and translations
    rotations, translations = dquat_torch.to_rotation_translation(dq.clone())
    
    # Make transformations local to the parent joints
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # Already in root space
           continue
            
        # Compute inverse rotation of the parent joint
        inv = quat_torch.inverse(rotations[..., parent, :])
        
        # Update translations and rotations to be local to parent
        translations[..., j, :] = quat_torch.mul_vec(
            inv,
            translations[..., j, :] - translations[..., parent, :]
        )
        rotations[..., j, :] = quat_torch.mul(inv, rotations[..., j, :])
    
    return translations, rotations


def from_root_dual_quat(dq: np.array, parents: np.array) -> np.array:
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq: np.array[..., n_joints, 8]
        Includes as first element the global position of the root joint
    parents: np.array[n_joints]

    Returns
    -------
    rotations : np.array[..., n_joints, 4]
    translations : np.array[..., n_joints, 3]
    """
    n_joints = dq.shape[1]
    # rotations has shape (frames, n_joints, 4)
    # translations has shape (frames, n_joints, 3)
    rotations, translations = dquat.to_rotation_translation(dq.copy())

    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
           continue
        inv = quat.inverse(rotations[..., parent, :])
        translations[..., j, :] = quat.mul_vec(
            inv,
            translations[..., j, :] - translations[..., parent, :],
        )
        rotations[..., j, :] = quat.mul(inv, rotations[..., j, :])
        #print(translations[:15,0,:])
    return translations, rotations


def from_root_dual_quat_to_root(dq: np.array, parents: np.array) -> np.array:
    """
    Convert root-centered dual quaternion to rotations and translations relative to the root joint.

    Parameters
    ----------
    dq: np.array[..., n_joints, 8]
        Dual quaternion tensor, where the first element includes the global position of the root joint.
    parents: np.array[n_joints]
        Parent indices for each joint.

    Returns
    -------
    root_relative_rotations : np.array[..., n_joints, 4]
        Rotations of all joints relative to the root joint.
    root_relative_translations : np.array[..., n_joints, 3]
        Translations of all joints relative to the root joint.
    """
    n_joints = dq.shape[-2]

    # Convert dual quaternions to rotations and translations
    rotations, translations = dquat.to_rotation_translation(dq.copy())

    # Initialize root-relative rotations and translations
    root_relative_rotations = rotations.copy()
    root_relative_translations = translations.copy()

    # Compute rotations and translations relative to the root joint
    for j in range(1, n_joints):
        # Compute the inverse rotation of the root joint
        #if(parents[j] == 0):
        #    continue
        
        root_inv = quat.inverse(rotations[..., 0, :])

        # Update translations to be relative to the root
        root_relative_translations[..., j, :] = quat.mul_vec(
            root_inv, translations[..., j, :] - translations[..., 0, :]
        )

        # Update rotations to be relative to the root
        root_relative_rotations[..., j, :] = quat.mul(
            root_inv, rotations[..., j, :]
        )

    return root_relative_translations, root_relative_rotations

def to_root_dual_quat(
    rotations: np.array, global_pos: np.array, parents: np.array, offsets: np.array
):
    """
    Convert the skeleton information to root-centered dual quaternions.

    Parameters
    ----------
    rotations : np.array[..., n_joints, 4]
        The rotation of the joint.
    global_pos: np.array[..., 3]
        The global position of the root joint.
    parents : np.array[n_joints]
        The parent of the joint.
    offsets : np.array[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    dual_quat : np.array[..., n_joints, 8]
        The root-centered dual quaternion representation of the skeleton.
    """
    assert (offsets[0] == np.zeros(3)).all()
    
    n_joints = rotations.shape[1]
    # translations has shape (..., n_joints, 3)
    rotations = rotations.copy()
    translations = np.tile(offsets, rotations.shape[:-2] + (1, 1))
    translations[..., 0, :] = global_pos
    # make transformations local to the root
    for j in range(1, n_joints):
        parent = parents[j]
        if parent == 0:  # already in root space
           continue
        translations[..., j, :] = (
            quat.mul_vec(rotations[..., parent, :], translations[..., j, :])
            + translations[..., parent, :]
        )
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])
    # convert to dual quaternions
    
    dual_quat = dquat.from_rotation_translation(rotations, translations)
    return dual_quat



def translation_each_joint(
    rotations: np.array, global_pos: np.array, parents: np.array, offsets: np.array
):
    """
    Compute translation of each joint

    Parameters
    ----------
    rotations : np.array[..., n_joints, 4]
        The rotation of the joint.
    global_pos: np.array[..., 3]
        The global position of the root joint.
    parents : np.array[n_joints]
        The parent of the joint.
    offsets : np.array[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    translations : np.array[..., n_joints, 3]
        The global positions of all joints.
    """
    assert (offsets[0] == np.zeros(3)).all()
    n_joints = rotations.shape[1]
    # translations has shape (..., n_joints, 3)
    rotations = rotations.copy()
    translations = np.tile(offsets, rotations.shape[:-2] + (1, 1))
    translations[..., 0, :] = global_pos
    # make transformations local to the root
    for j in range(1, n_joints):
        parent = parents[j]
        #if parent == 0:  # already in root space
        #    continue
        #    translations[..., j, :] = translations[..., j, :] + global_pos
        #else:
        translations[..., j, :] = (
            quat.mul_vec(rotations[..., parent, :], translations[..., j, :])
            + translations[..., parent, :]
        )
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])
        
    #for j in range(1, n_joints):
    #    translations[..., j, :] = translations[..., j, :] - global_pos
    

    return translations

def translation_each_joint_torch(
    rotations: torch.Tensor, global_pos: torch.Tensor, parents: torch.Tensor, offsets: torch.Tensor
):
    """
    Compute translation of each joint in a skeleton using root-centered dual quaternions.

    Parameters
    ----------
    rotations : torch.Tensor[..., n_joints, 4]
        The rotation of each joint in quaternion form.
    global_pos: torch.Tensor[..., 3]
        The global position of the root joint.
    parents : torch.Tensor[n_joints]
        The index of each joint's parent.
    offsets : torch.Tensor[n_joints, 3]
        The local offset of each joint from its parent.

    Returns
    -------
    translations : torch.Tensor[..., n_joints, 3]
        The computed translations for each joint in the skeleton.
    """
    assert torch.all(offsets[0] == 0), "The root joint offset should be zero."
    
    
    n_joints = rotations.shape[-2]
    translations = offsets.expand(*rotations.shape[:-2], -1, -1).clone()
    #print(translations.shape, rotations.shape, global_pos.shape, offsets.shape) # torch.Size([batch, frames, joints, 3]) torch.Size([batch, frames, joints, 4]) torch.Size([batch, frames, 3]) torch.Size([joints, 3])
    translations[..., 0, :] = global_pos[...,:,:].clone()
    # Compute transformations for each joint based on parent relationships
    for j in range(1, n_joints):
        parent = parents[j].item()  # Get the parent index for joint j
        #if parent == 0:
        #    continue
        # Transform offset by the parent's rotation, then add the parent's translation
        translations[..., j, :] = (
            quat_torch.mul_vec(rotations[..., parent, :].clone(), translations[..., j, :].clone()) +
            translations[..., parent, :].clone()
        )
        # Update rotation to be relative to the parent rotation
        rotations[..., j, :] = quat_torch.mul(rotations[..., parent, :].clone(), rotations[..., j, :].clone())

    return translations



def to_root_space(
    rotations: np.array, global_pos: np.array, parents: np.array, offsets: np.array
):
    """
    Compute rotations and positions of each joint with respect to the root joint.

    Parameters
    ----------
    rotations : np.array[..., n_joints, 4]
        The rotation of each joint relative to its parent.
    global_pos: np.array[..., 3]
        The global position of the root joint.
    parents : np.array[n_joints]
        The parent of each joint.
    offsets : np.array[n_joints, 3]
        The offset of each joint from its parent.

    Returns
    -------
    root_relative_positions : np.array[..., n_joints, 3]
        The positions of all joints relative to the root joint.
    root_relative_rotations : np.array[..., n_joints, 4]
        The rotations of all joints relative to the root joint.
    """
    assert (offsets[0] == np.zeros(3)).all(), "Root joint offset must be zero."
    n_joints = rotations.shape[1]

    # Initialize positions and rotations
    rotations = rotations.copy()
    positions = np.tile(offsets, rotations.shape[:-2] + (1, 1))
    positions[..., 0, :] = global_pos

    # Compute global positions and rotations
    for j in range(1, n_joints):
        parent = parents[j]
        positions[..., j, :] = (
            quat.mul_vec(rotations[..., parent, :], positions[..., j, :])
            + positions[..., parent, :]
        )
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])

    # Convert to root-relative positions and rotations
    root_relative_positions = positions #- positions[..., 0:1, :]
    root_relative_positions[...,0  ,:] = global_pos 
    root_relative_rotations = rotations.copy()
    for j in range(1, n_joints):
        root_relative_rotations[..., j, :] = quat.mul(
            quat.inverse(rotations[..., 0, :]), rotations[..., j, :]
        )

    return root_relative_positions


def compute_global_pos_torch_(rotations: torch.Tensor, positions,    
                            offsets: torch.Tensor, parents: torch.Tensor):
    """
    Computes the global positions and rotations of all joints using quaternion rotations,
    without any in-place ops that would break autograd.
    """
    device = rotations.device
    batch_size, n_frames, n_joints = rotations.shape[0], rotations.shape[1], rotations.shape[2]

    # 1) Prepare root positions: if `positions` has fewer frames, repeat last frame to match n_frames.
    root_pos = positions[..., 0, :]  # shape [batch, n_frames_before, 3]
    if root_pos.shape[1] < n_frames:
        last = root_pos[..., -1:, :]          # [batch, 1, 3]
        extra = last.repeat(1, n_frames - root_pos.shape[1], 1)  # [batch, (n_frames - old), 3]
        root_pos = torch.cat([root_pos, extra], dim=1)           # [batch, n_frames, 3]

    # 2) Convert all quaternions to 3×3 local‐rotation matrices at once:
    local_rot_mats = quat_torch.to_matrix(rotations)  # [batch, n_frames, n_joints, 3, 3]

    # 3) Initialize lists to collect results
    global_positions_list = []
    global_rot_mats_list = []

    # 4) Set root values (first joint)
    global_positions_list.append(root_pos.unsqueeze(2))  # [batch, n_frames, 1, 3]
    global_rot_mats_list.append(local_rot_mats[..., 0:1, :, :])  # [batch, n_frames, 1, 3, 3]
    # 5) Forward kinematics for each joint
    for j in range(1, n_joints):
        parent = parents[j]
        
        # Get parent's global rotation and position from the list
        parent_global_rot = global_rot_mats_list[parent]  # [batch, n_frames, 1, 3, 3]
        parent_global_pos = global_positions_list[parent]  # [batch, n_frames, 1, 3]
        
        # Remove singleton dimension for computation
        parent_global_rot = parent_global_rot.squeeze(2)  # [batch, n_frames, 3, 3]
        parent_global_pos = parent_global_pos.squeeze(2)  # [batch, n_frames, 3]
        
        # Get current joint's local rotation
        local_rot_j = local_rot_mats[..., j, :, :]  # [batch, n_frames, 3, 3]
        
        # Compute global rotation: parent_global_rot @ local_rot_j
        joint_global_rot = torch.matmul(parent_global_rot, local_rot_j)  # [batch, n_frames, 3, 3]
        
        # Compute global position
        offset_j = offsets[:, j, :].unsqueeze(1).expand(-1, n_frames, -1)  # [batch, n_frames, 3]
        offset_j_col = offset_j.unsqueeze(-1)  # [batch, n_frames, 3, 1]
        world_offset_j = torch.matmul(parent_global_rot, offset_j_col).squeeze(-1)  # [batch, n_frames, 3]
        
        joint_global_pos = parent_global_pos + world_offset_j  # [batch, n_frames, 3]
        
        # Add to lists with singleton dimension for joint index
        global_positions_list.append(joint_global_pos.unsqueeze(2))  # [batch, n_frames, 1, 3]
        global_rot_mats_list.append(joint_global_rot.unsqueeze(2))   # [batch, n_frames, 1, 3, 3]

    # 6) Concatenate all results
    global_rot_mats = torch.cat(global_rot_mats_list, dim=2)    # [batch, n_frames, n_joints, 3, 3]
    global_positions = torch.cat(global_positions_list, dim=2)  # [batch, n_frames, n_joints, 3]

    return global_positions, global_rot_mats

def compute_global_pos_torch(rotations: torch.Tensor, positions: torch.Tensor,    
                                      offsets: torch.Tensor, parents, end_sites=None, end_sites_parents=None):
    """
    Vectorized forward kinematics that processes all joints in parallel waves.
    This avoids the sequential dependency that causes memory issues.
    """
    device = rotations.device
    batch_size, n_frames, n_joints = rotations.shape[0], rotations.shape[1], rotations.shape[2]
    
    # 1) Prepare root positions
    root_pos = positions.clone()
    if root_pos.shape[1] < n_frames:
        last = root_pos[..., -1:, :]
        extra = last.repeat(1, n_frames - root_pos.shape[1], 1)
        root_pos = torch.cat([root_pos, extra], dim=1)
    elif root_pos.shape[1] > n_frames:
        root_pos = root_pos[:, :n_frames, :]
    
    # 2) Convert quaternions to rotation matrices
    local_rot_mats = quat_torch.to_matrix(rotations)  # [batch, n_frames, n_joints, 3, 3]
    
    # 3) Initialize global transforms
    global_positions = torch.zeros(batch_size, n_frames, n_joints, 3, device=device)
    global_rot_mats = torch.zeros(batch_size, n_frames, n_joints, 3, 3, device=device)
    
    # 4) Set root
    global_positions[..., 0, :] = root_pos
    global_rot_mats[..., 0, :, :] = local_rot_mats[..., 0, :, :]
    
    # 5) Create depth levels for parallel processing
    # Group joints by their depth in the hierarchy
    max_depth = 0
    joint_depths = torch.zeros(n_joints, dtype=torch.long)
    
    for j in range(1, n_joints):
        depth = 1
        parent = parents[j]  # parents[j] is already an integer
        while parent != 0:
            depth += 1
            parent = parents[parent]  # parents[parent] is already an integer
        joint_depths[j] = depth
        max_depth = max(max_depth, depth)
    
    # Process joints level by level
    for depth in range(1, max_depth + 1):
        joints_at_depth = torch.where(joint_depths == depth)[0]
        
        if len(joints_at_depth) == 0:
            continue
            
        # Vectorized computation for all joints at this depth
        parent_indices = [parents[j.item()] for j in joints_at_depth]  # List of parent indices
        parent_indices = torch.tensor(parent_indices, device=device)  # Convert back to tensor
        
        # Get parent transforms
        parent_rot = global_rot_mats[..., parent_indices, :, :]  # [batch, frames, num_joints, 3, 3]
        parent_pos = global_positions[..., parent_indices, :]     # [batch, frames, num_joints, 3]
        
        # Get local rotations
        local_rot = local_rot_mats[..., joints_at_depth, :, :]   # [batch, frames, num_joints, 3, 3]
        
        # Compute global rotations
        joint_global_rot = torch.matmul(parent_rot, local_rot)
        
        # Compute global positions
        joint_offsets = offsets[:, joints_at_depth, :].unsqueeze(1)  # [batch, 1, num_joints, 3]
        joint_offsets = joint_offsets.expand(-1, n_frames, -1, -1)   # [batch, frames, num_joints, 3]
        
        # Transform offsets to world space
        world_offsets = torch.matmul(
            parent_rot, 
            joint_offsets.unsqueeze(-1)
        ).squeeze(-1)
        
        joint_global_pos = parent_pos + world_offsets
        
        # Update global transforms
        global_positions[..., joints_at_depth, :] = joint_global_pos
        global_rot_mats[..., joints_at_depth, :, :] = joint_global_rot
    
    # assume `end_sites` and `end_sites_parents` are tensors on device (see io/bvh.BVH.compute_global_pos)
    parent_idx = end_sites_parents.to(dtype=torch.long)
    
    # parent positions and rotations for each end-site: [batch, frames, E, ...]
    parent_pos_es = global_positions[..., parent_idx[-1], :]       # [batch, frames, E, 3]
    parent_rot_es = global_rot_mats[..., parent_idx[-1], :, :]    # [batch, frames, E, 3, 3]
    # ensure end_sites has XYZ as last 3 channels; if extra channels present, take last 3
   
    # prepare end_sites for broadcast: accept [E,3] or [batch,E,3]
    if end_sites.dim() == 2:  # [E,3]
        end_sites_exp = end_sites.unsqueeze(0).unsqueeze(0)   # [1,1,E,3]
    elif end_sites.dim() == 3 and end_sites.shape[0] == batch_size:  # [batch,E,3]
        end_sites_exp = end_sites.unsqueeze(1)                 # [batch,1,E,3]
    else:
        # fallback: insert frame dim -> [...,1,E,3]
        end_sites_exp = end_sites.unsqueeze(1)
    # rotate offsets and add parent position


    rotated = torch.matmul(parent_rot_es, end_sites_exp.unsqueeze(-1)).squeeze(-1)  # [batch, frames, E, 3]
    end_sites_global = parent_pos_es + rotated

    global_positions[..., parent_idx[-1], :] = end_sites_global
    
    return global_positions, global_rot_mats
import torch
import train_vae
from pymotion.ops.forward_kinematics_torch import fk


def check_sparse_joint_indices(sparse_joint_indices, joint_count):
    """Return the indices as a list, or raise ValueError unless they are nonempty, unique and within joint_count."""
    indices = list(sparse_joint_indices)
    if not indices:
        raise ValueError("sparse_joint_indices is empty: MEEPE needs at least one non-root sparse joint")
    if len(set(indices)) != len(indices):
        raise ValueError(f"sparse_joint_indices must be unique, got {indices}")
    out_of_range = [i for i in indices if not 0 <= i < joint_count]
    if out_of_range:
        raise ValueError(
            f"sparse_joint_indices {out_of_range} are outside the evaluated skeleton's {joint_count} joints"
        )
    return indices


def eval_pos_error(gt_bvh, eval_bvh, device, sparse_joint_indices):
    """Return (MPJPE, MEEPE).

    MPJPE is the mean position error over every joint. MEEPE is the mean position error over
    sparse_joint_indices, which callers pass as param["sparse_joints"][1:] (the root is excluded).
    These are the configured non-root sparse joints, not strictly anatomical end effectors.
    """
    gt_rots, gt_pos, gt_parents, gt_offsets, _ = train_vae.get_info_from_bvh(gt_bvh)
    gt_rots = torch.from_numpy(gt_rots).float().to(device).unsqueeze(0)
    gt_pos = torch.zeros(gt_rots.shape[0], gt_rots.shape[1], 3)
    gt_offsets = torch.from_numpy(gt_offsets).float().to(device)
    gt_joint_poses, _ = fk(
        gt_rots, gt_pos, gt_offsets, torch.Tensor(gt_parents).long().to(device)
    )
    rots, pos, parents, offsets, _ = train_vae.get_info_from_bvh(eval_bvh)
    # pos = (
    #     torch.from_numpy(pos).float().to(device)[:, 0, :].permute(1, 0).unsqueeze(0)
    # )  # only global position
    rots = torch.from_numpy(rots).float().to(device).unsqueeze(0)

    pos = torch.zeros(rots.shape[0], rots.shape[1], 3)
    offsets = torch.from_numpy(offsets).float().to(device)
    joint_poses, _ = fk(rots, pos, offsets, torch.Tensor(parents).long().to(device))   
    # error

    # Modifying layer number needs this
    error = torch.norm(
        joint_poses[:,:min(joint_poses.shape[1],gt_joint_poses.shape[1]),...] 
        - gt_joint_poses[:, :min(joint_poses.shape[1],gt_joint_poses.shape[1]), ...], dim=-1   #<----------------------
    )

    # original error 
    #error = torch.norm(
    #joint_poses - gt_joint_poses[:, : joint_poses.shape[1], ...], dim=-1
    #)
    
    # Indices come from the caller's morphology, so the evaluated skeleton's joint count is what they are checked against
    sparse_joint_indices = check_sparse_joint_indices(sparse_joint_indices, error.shape[-1])
    sparse_error = error[:, :, sparse_joint_indices]
    return torch.mean(error).item(), torch.mean(sparse_error).item()

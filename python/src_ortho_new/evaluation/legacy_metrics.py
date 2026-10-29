import os

import numpy as np
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import translation_each_joint
import pymotion.rotations.quat as quat
from motion.signals import compute_tags
from evaluation.phase_visualization import visualize_contact_motion
from utils.fid_metrics import load_bvh_global_positions
from utils.phase_metrics import (
    compute_continuous_phase_error,
    prepare_contact_labels,
)


def load_bvh_and_compute_tags(bvh_path, param=None, is_human=False):
    """
    Load a BVH file and compute motion tags.
    
    Args:
        bvh_path: str, path to BVH file
        param: dict, parameters for skeleton processing (optional)
        is_human: bool, whether skeleton is human or not
    
    Returns:
        tags: dict, computed motion tags
        bvh: BVH object
        pos_all_joints: np.ndarray, global joint positions
    """
    bvh, pos_all_joints = load_bvh_global_positions(bvh_path)
    
    rot_order = np.tile(bvh.data["rot_order"][0], (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_order),
        axis=0,
    )
    rots = quat.normalize(rots)
    
    
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    
    # Use BVH global positions so end sites are available to downstream visualization.
    
    if param is None:
        param = {
            "head_idx": 4 if is_human else 2,
            "head_height": 1.0,
            "skeleton_height": 0.90,
            "feet_idxs": [15, 18] if is_human else [8, 12, 15, 18],
            "not_dog": is_human,
            "feet_contact_threshold": 0.02
        }
    
    rots_euler = bvh.data["rotations"][:, 0]  # root y rotation
    
    tags = compute_tags(
        pos_all_joints,
        head_idx=param["head_idx"],
        head_height=param["head_height"],
        downsample_factor=8,
        is_human=is_human,
        skeleton_height=param["skeleton_height"],
        rots=np.deg2rad(rots_euler),
        is_deg=False,
        scalar=1.0,
        feet_idx=param["feet_idxs"],
        quats=rots,
        padded_frames=0,
        shoulder_idx=[5, 9] if is_human else [5, 9],
        not_dog=param["not_dog"],
        feet_contact_threshold=param["feet_contact_threshold"]
    )
    
    return tags, bvh, pos_all_joints


def compare_motion_tags(bvh_path1, bvh_path2, param1=None, param2=None, is_human1=False, is_human2=False,
                        return_positions=False):
    """
    Load two BVH files and compute tags for comparison.
    
    Args:
        bvh_path1: str, path to first BVH file
        bvh_path2: str, path to second BVH file  
        param1: dict, parameters for first skeleton
        param2: dict, parameters for second skeleton
        is_human1: bool, whether first skeleton is human
        is_human2: bool, whether second skeleton is human
    
    Returns:
        tags1: dict, tags for first motion
        tags2: dict, tags for second motion
        bvh1: BVH object for first file
        bvh2: BVH object for second file
        pos1/pos2: optional global joint positions when return_positions=True
    """
    print(f"Loading BVH file 1: {bvh_path1}")
    tags1, bvh1, pos1 = load_bvh_and_compute_tags(bvh_path1, param1, is_human1)
    
    print(f"Loading BVH file 2: {bvh_path2}")
    tags2, bvh2, pos2 = load_bvh_and_compute_tags(bvh_path2, param2, is_human2)
    
    print(f"Motion 1: {pos1.shape[0]} frames, {pos1.shape[1]} joints")
    print(f"Motion 2: {pos2.shape[0]} frames, {pos2.shape[1]} joints")
    
    # Print available tag keys for both motions
    print(f"Tags in motion 1: {list(tags1.keys())}")
    print(f"Tags in motion 2: {list(tags2.keys())}")
    
    if return_positions:
        return tags1, tags2, bvh1, bvh2, pos1, pos2

    return tags1, tags2, bvh1, bvh2


# new phase helpers

def timing(bvh_path1, bvh_path2, param1, param2, save_visualization=False):
    try:
        # Load and compute tags for both motions
        tags1, tags2, bvh1, bvh2, pos1, pos2 = compare_motion_tags(
            bvh_path1, bvh_path2,
            param1, param2,
            is_human1=True, is_human2=False,
            return_positions=True,
        )
        
        # Example: compare velocity tags
        if "velocity" in tags1 and "velocity" in tags2:
            print(f"Motion 1 velocity range: {np.min(tags1['velocity'])} - {np.max(tags1['velocity'])}")
            print(f"Motion 2 velocity range: {np.min(tags2['velocity'])} - {np.max(tags2['velocity'])}")
        
        labels1 = tags1["binary_foot_labels"][:, :2]
        labels2 = tags2["binary_foot_labels"][:, :2]

        shared_feet = min(labels1.shape[1], labels2.shape[1])
        if shared_feet == 0:
            raise ValueError("No shared foot channels available for timing metrics.")
        if shared_feet < 2:
            print("Only one shared foot channel available; average/std will use that single foot.")
        labels1 = labels1[:, :shared_feet]
        labels2 = labels2[:, :shared_feet]

        phase_results = compute_continuous_phase_error(
            labels1,
            labels2,
            labels_are_contact=False,
        )

        print("\nContinuous contact-phase comparison")
        print("===================================================")
        print(
            f"Average phase error: {phase_results['avg_phase_error_rad']:.3f} rad "
            f"({phase_results['avg_phase_error_deg']:.1f} deg)"
        )
        print(
            f"Phase error std: {phase_results['std_phase_error_rad']:.3f} rad "
            f"({phase_results['std_phase_error_deg']:.1f} deg)"
        )
        print(f"Aggregate feet used for avg/std: {phase_results['aggregate_num_feet']}")
        print(f"Per-foot phase error (rad): {phase_results['per_foot_phase_error_rad']}")
        print(f"Per-foot contact-on error (frames): {phase_results['per_foot_contact_on_error_frames']}")
        print(f"Per-foot contact-off error (frames): {phase_results['per_foot_contact_off_error_frames']}")
        print(
            f"Aggregate contact disagreement: {phase_results['aggregate_contact_disagreement_frames']} / "
            f"{phase_results['aggregate_contact_compared_frames']} frames "
            f"({phase_results['aggregate_contact_disagreement_pct']:.2f}%)"
        )
        print(
            f"Aggregate long-run contact disagreement (2+ frames): "
            f"{phase_results['aggregate_long_contact_disagreement_frames']} / "
            f"{phase_results['aggregate_contact_compared_frames']} frames "
            f"({phase_results['aggregate_long_contact_disagreement_pct']:.2f}%)"
        )
        print(f"Per-foot contact disagreement (frames): {phase_results['per_foot_contact_disagreement_frames']}")
        print(f"Per-foot contact disagreement (%): {phase_results['per_foot_contact_disagreement_pct']}")
        print(f"Per-foot long-run disagreement (frames): {phase_results['per_foot_long_contact_disagreement_frames']}")
        print(f"Per-foot long-run disagreement (%): {phase_results['per_foot_long_contact_disagreement_pct']}")
        print(f"Valid phase frames per foot: {phase_results['valid_frames_per_foot']}")

        visualize_contacts = prepare_contact_labels(
            tags1["binary_foot_labels"],
            labels_are_contact=False,
        )
        if(save_visualization):
            visualize_path = os.path.join(
                os.path.dirname(bvh_path1) or ".",
                f"{os.path.splitext(os.path.basename(bvh_path1))[0]}_contact_phase.gif",
            )
        else:
            visualize_path = None
            
        skeleton_to_visualize = 1 # 0 or 1
        if skeleton_to_visualize == 0:
            visualize_contact_motion(
                pos1,
                bvh1.data["parents"],
                visualize_contacts,
                param1["feet_idxs"],
                output_path=visualize_path,
                title=f"Contact Phase: {os.path.basename(bvh_path1)}",
                phase_values=phase_results["phase_1"],
                show_phase_panel=True,
                end_site_parents=bvh1.data.get("end_sites_parents"),
                show=True,
            )
        else:
            visualize_contact_motion(
                pos2,
                bvh2.data["parents"],
                visualize_contacts,
                param2["feet_idxs"],
                output_path=visualize_path,
                title=f"Contact Phase: {os.path.basename(bvh_path2)}",
                phase_values=phase_results["phase_2"],
                show_phase_panel=True,
                end_site_parents=bvh2.data.get("end_sites_parents"),
                show=True,
            )
            
        if save_visualization:
            print(f"Saved contact visualization to: {visualize_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the bvh_path1 and bvh_path2 variables with valid file paths")


def compute_jitter_from_bvh(bvh_path, param=None, is_human=False, fps=30.0):
    """
    Compute jitter (squared jerk) for a single BVH file.
    
    Jitter = 10^2 * m/s^3 (squared magnitude of jerk)
    Jerk = third derivative of position (m/s^3)
    
    Args:
        bvh_path: str, path to BVH file
        param: dict, parameters for skeleton processing
        is_human: bool, whether skeleton is human
        fps: float, frames per second (default 30.0)
    
    Returns:
        dict with:
            - 'avg_jitter': float, average jitter across all joints and all frames
            - 'per_joint_jitter': np.ndarray, average jitter per joint
            - 'jerk_magnitude': np.ndarray, jerk magnitude over time [frames, joints]
            - 'jitter_over_time': np.ndarray, jitter over time [frames, joints]
    """
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    
    bvh = BVH()
    bvh.load(bvh_path)
    
    rot_order = np.tile(bvh.data["rot_order"][0], 
                       (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_order),
        axis=0,
    )
    rots = quat.normalize(rots)
    
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)
    
    pos_all_joints = translation_each_joint(rots, pos[:, 0, :], parents, offsets)
    
    num_frames = pos_all_joints.shape[0]
    num_joints = pos_all_joints.shape[1]
    
    # Time step between frames
    dt = 1.0 / fps
    
    # First derivative: velocity [frames-1, joints, 3]
    velocity = np.diff(pos_all_joints, axis=0) / dt
    
    # Second derivative: acceleration [frames-2, joints, 3]
    acceleration = np.diff(velocity, axis=0) / dt
    
    # Third derivative: jerk [frames-3, joints, 3]
    jerk = np.diff(acceleration, axis=0) / dt
    
    # Compute jerk magnitude per joint per frame: [frames-3, joints]
    jerk_magnitude = np.linalg.norm(jerk, axis=2)
    
    # Compute jitter: squared jerk magnitude * 10^2
    # Jitter = (jerk_magnitude)^2 * 100
    # jitter_over_time = (jerk_magnitude ** 2) * 100.0
    jitter_over_time = jerk_magnitude / 100
    
    # Average jitter per joint across all frames
    per_joint_jitter = np.mean(jitter_over_time, axis=0)
    
    # Average jitter across all joints and all frames
    avg_jitter = np.mean(jitter_over_time)
    
    return {
        'avg_jitter': avg_jitter,
        'per_joint_jitter': per_joint_jitter,
        'jerk_magnitude': jerk_magnitude,
        'jitter_over_time': jitter_over_time,
        'num_frames': num_frames,
        'num_joints': num_joints
    }


def compute_jitter_for_directory(directory_path, param=None, is_human=False, fps=30.0, 
                                 visualize=False, save_results=False):
    """
    Compute average jitter for all BVH files in a directory.
    
    Args:
        directory_path: str, path to directory containing BVH files
        param: dict, parameters for skeleton processing
        is_human: bool, whether skeletons are human
        fps: float, frames per second
        visualize: bool, whether to create visualization plots
        save_results: bool, whether to save results to CSV
    
    Returns:
        dict with:
            - 'overall_avg_jitter': float, average jitter across all files
            - 'per_file_jitter': dict, {filename: jitter_value}
            - 'per_file_results': dict, {filename: full_results_dict}
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all BVH files
    bvh_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".bvh"):
                bvh_files.append(os.path.join(root, file))
    
    if not bvh_files:
        print(f"No BVH files found in {directory_path}")
        return None
    
    print(f"Found {len(bvh_files)} BVH files")
    print("Computing jitter for each file...\n")
    
    per_file_jitter = {}
    per_file_results = {}
    all_jitter_values = []
    
    for i, bvh_path in enumerate(bvh_files):
        filename = os.path.basename(bvh_path)
        try:
            print(f"[{i+1}/{len(bvh_files)}] Processing: {filename}")
            
            results = compute_jitter_from_bvh(bvh_path, param, is_human, fps)
            
            per_file_jitter[filename] = results['avg_jitter']
            per_file_results[filename] = results
            all_jitter_values.append(results['avg_jitter'])
            
            print(f"  Avg jitter: {results['avg_jitter']:.4f} (10^2 m/s^3)")
            print(f"  Frames: {results['num_frames']}, Joints: {results['num_joints']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not all_jitter_values:
        print("No files processed successfully")
        return None
    
    # Compute overall statistics
    overall_avg_jitter = np.mean(all_jitter_values)
    overall_std_jitter = np.std(all_jitter_values)
    overall_min_jitter = np.min(all_jitter_values)
    overall_max_jitter = np.max(all_jitter_values)
    
    print("\n" + "="*60)
    print("JITTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(all_jitter_values)}")
    print(f"Overall average jitter: {overall_avg_jitter:.4f} (10^2 m/s^3)")
    print(f"Standard deviation: {overall_std_jitter:.4f}")
    print(f"Min jitter: {overall_min_jitter:.4f} (file: {min(per_file_jitter, key=per_file_jitter.get)})")
    print(f"Max jitter: {overall_max_jitter:.4f} (file: {max(per_file_jitter, key=per_file_jitter.get)})")
    print("="*60)
    
    # Save results to CSV
    if save_results:
        output_csv = os.path.join(directory_path, "jitter_analysis.csv")
        with open(output_csv, 'w') as f:
            f.write("filename,avg_jitter,num_frames,num_joints\n")
            for filename, jitter in per_file_jitter.items():
                results = per_file_results[filename]
                f.write(f"{filename},{jitter:.6f},{results['num_frames']},{results['num_joints']}\n")
        print(f"\nResults saved to: {output_csv}")
    
    # Visualization
    if visualize:
        import matplotlib.pyplot as plt
        
        # Plot 1: Histogram of jitter values
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(all_jitter_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(overall_avg_jitter, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {overall_avg_jitter:.4f}')
        plt.xlabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Average Jitter Across Files', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Sorted bar chart of jitter per file (top 20)
        plt.subplot(2, 2, 2)
        sorted_files = sorted(per_file_jitter.items(), key=lambda x: x[1], reverse=True)[:20]
        filenames = [os.path.splitext(f[0])[0][:20] for f in sorted_files]  # Truncate names
        jitter_vals = [f[1] for f in sorted_files]
        
        plt.barh(range(len(filenames)), jitter_vals, color='coral', edgecolor='black')
        plt.yticks(range(len(filenames)), filenames, fontsize=8)
        plt.xlabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.title('Top 20 Files by Jitter', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Per-joint jitter for a sample file (use file with median jitter)
        plt.subplot(2, 2, 3)
        median_idx = np.argsort(all_jitter_values)[len(all_jitter_values)//2]
        sample_filename = list(per_file_jitter.keys())[median_idx]
        sample_results = per_file_results[sample_filename]
        
        joint_indices = np.arange(sample_results['num_joints'])
        plt.bar(joint_indices, sample_results['per_joint_jitter'], 
               color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Joint Index', fontsize=12)
        plt.ylabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.title(f'Per-Joint Jitter (Sample: {os.path.splitext(sample_filename)[0]})', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Jitter over time for sample file
        plt.subplot(2, 2, 4)
        jitter_time = sample_results['jitter_over_time']
        avg_jitter_per_frame = np.mean(jitter_time, axis=1)
        
        plt.plot(avg_jitter_per_frame, linewidth=1.5, color='purple', alpha=0.7)
        plt.axhline(sample_results['avg_jitter'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {sample_results['avg_jitter']:.4f}")
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Jitter (10^2 m/s^3)', fontsize=12)
        plt.title(f'Jitter Over Time (Sample: {os.path.splitext(sample_filename)[0]})', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_fig = os.path.join(directory_path, "jitter_analysis.png")
        plt.savefig(output_fig, dpi=150)
        print(f"Visualization saved to: {output_fig}")
        
        plt.show()
    
    return {
        'overall_avg_jitter': overall_avg_jitter,
        'overall_std_jitter': overall_std_jitter,
        'overall_min_jitter': overall_min_jitter,
        'overall_max_jitter': overall_max_jitter,
        'per_file_jitter': per_file_jitter,
        'per_file_results': per_file_results
    }


def compute_foot_skating_from_bvh(bvh_path, param=None, is_human=False, fps=30.0, 
                                  height_thresh=0.2, vel_thresh=0.02): # 5cm height, 2cm/frame velocity
    """
    Compute foot skating metrics for a single BVH file.
    
    Skating is defined as: Foot is close to ground (height < height_thresh) 
    BUT moving fast (velocity > vel_thresh).
    
    Args:
        bvh_path: str, path to BVH file
        param: dict, parameters containing 'feet_idxs'
        is_human: bool, whether skeleton is human
        fps: float, frames per second
        height_thresh: float, max height to be considered "on ground" (meters)
        vel_thresh: float, max velocity to be considered "planted" (meters/frame)
    
    Returns:
        dict with:
            - 'skating_ratio': float, % of frames where skating occurs
            - 'avg_skating_dist': float, average distance slid per skating frame
            - 'per_foot_skating': list, skating ratio per foot
    """
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")
    
    bvh = BVH()
    bvh.load(bvh_path)
    
    rot_order = np.tile(bvh.data["rot_order"][0], 
                       (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_order),
        axis=0,
    )
    rots = quat.normalize(rots)
    
    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)
    
    pos_all_joints = translation_each_joint(rots, pos[:, 0, :], parents, offsets)
    
    if param is None:
        feet_idx = [15, 18] if is_human else [8, 12, 15, 18]
    else:
        feet_idx = param["feet_idxs"][:2]
        
    if(is_human):
        pos_all_joints = pos_all_joints[3:]
    total_skating_frames = 0
    total_frames_analyzed = 0
    total_skating_dist = 0.0
    per_foot_skating = []
    
    root_velo_xz = np.linalg.norm(np.diff(pos_all_joints[:,0,[0,2]], axis=0, prepend=pos_all_joints[0:1,0,[0,2]]), axis=1)
    dist = np.sum(root_velo_xz)
    print("Total root distance:", dist)
    
    ground_y = 0.0
    import matplotlib.pyplot as plt
    print(pos_all_joints[:,feet_idx].shape)
    plt.plot(pos_all_joints[:,feet_idx][:,:,1])
    plt.show()
    for foot_id in feet_idx:
        foot_pos = pos_all_joints[:, foot_id, :] # [frames, 3]
        
        # 1. Calculate Height
        foot_height = foot_pos[:, 1] - ground_y
        
        # 2. Calculate Velocity (displacement per frame)
        # We only care about horizontal sliding (XZ plane)
        foot_vel_xz = np.linalg.norm(np.diff(foot_pos[:, [0, 2]], axis=0, prepend=foot_pos[0:1, [0, 2]]), axis=1)
        # 3. Detect Skating
        # Condition: Foot is on ground AND Foot is moving
        is_contact = foot_height < height_thresh
        is_moving = foot_vel_xz > vel_thresh
        
        is_skating = np.logical_and(is_contact, is_moving)
        
        skating_frames = np.sum(is_skating)
        num_frames = len(foot_pos)
        plt.plot(foot_vel_xz)
        # Calculate metrics
        ratio = skating_frames / num_frames if num_frames > 0 else 0
        per_foot_skating.append(ratio)
        
        # Accumulate distance slid during skating frames
        skating_dist = np.sum(foot_vel_xz[is_skating])
        
        total_skating_frames += skating_frames
        total_frames_analyzed += num_frames
        total_skating_dist += skating_dist
    plt.show()
    avg_skating_ratio = total_skating_frames / total_frames_analyzed if total_frames_analyzed > 0 else 0
    avg_skating_dist = total_skating_dist / total_skating_frames if total_skating_frames > 0 else 0
    avg_skating_cm_per_frame = (
        (total_skating_dist / total_frames_analyzed) * 100.0
        if total_frames_analyzed > 0 else 0
    )
    
    return {
        'skating_ratio': avg_skating_ratio, # Percentage of time feet slide
        'avg_skating_dist': avg_skating_dist, # Avg distance slid per slide event
        'per_foot_skating': per_foot_skating,
        'total_skating_dist': total_skating_dist,
        'avg_skating_cm_per_frame': avg_skating_cm_per_frame,
    }

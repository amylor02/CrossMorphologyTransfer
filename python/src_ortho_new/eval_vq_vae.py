import os
import json
import argparse
import torch
import train
from motion_data import TestMotionData, compute_phase, construct_signal, compute_tags
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
from networks.unified.foot_contact import foot_count, map_source_foot_contact
import eval_metrics
import numpy as np
import train_vq_vae
import skeleton_recognition as skeletons
from architectures import ARCHITECTURE_MODULES
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d_torch as ortho6d
import pymotion.rotations.ortho6d as ortho6d_np
import matplotlib.pyplot as plt
from pymotion.ops.skeleton import translation_each_joint, from_root_dual_quat_torch, translation_each_joint_torch, to_root_space
# Evaluation Modes
GENERATOR = 1
IK = 2


def plot_prior_logits_interactive(logits, step=20, max_frames=100):
    if logits is None:
        return

    logits = logits.detach().cpu()
    if logits.dim() != 4 or logits.size(0) == 0:
        return

    probs = torch.softmax(logits[0, :, 0, :], dim=-1).numpy()
    total_frames = min(max_frames, probs.shape[0])

    frame_indices = list(range(0, total_frames, step))
    if not frame_indices:
        return

    plt.ion()
    for frame_idx in frame_indices:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(np.arange(probs.shape[1]), probs[frame_idx], width=1.0)
        ax.set_title(f"Prior categorical distribution at frame {frame_idx}")
        ax.set_xlabel("Codebook index")
        ax.set_ylabel("Probability")
        ax.set_xlim(0, probs.shape[1] - 1)
        ax.set_ylim(0.0, max(1e-6, probs[frame_idx].max()) * 1.1)

        should_advance = {"value": False}

        def on_key(event):
            if event.key == "q":
                should_advance["value"] = True
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=False)
        while plt.fignum_exists(fig.number) and not should_advance["value"]:
            plt.pause(0.1)
    plt.ioff()

RETARGETING_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "retargeting.json")

# Source-guided stationary root locking (cross-morphology retargeting only).
# Thresholds apply to the source ctrl_velocity at scalar 1.0 (file units/s, assumed metres/s).
STATIONARY_ROOT_LOCK_THRESHOLD = 0.1
STATIONARY_ROOT_LOCK_MIN_FRAMES = 30  # 1 s at 30 FPS
STATIONARY_ROOT_LOCK_MARGIN = 5  # frames excluded at each end of a qualifying interval


def stationary_root_lock_mask(source_ctrl_velocity, threshold=STATIONARY_ROOT_LOCK_THRESHOLD,
                              min_frames=STATIONARY_ROOT_LOCK_MIN_FRAMES, margin=STATIONARY_ROOT_LOCK_MARGIN):
    """Boolean mask over source frames: True inside sustained stationary intervals, minus the margins.

    Returns (mask, number_of_qualifying_intervals). Intervals shorter than min_frames (before
    margins are removed) do not qualify, so short pauses are never locked.
    """
    ctrl = np.asarray(source_ctrl_velocity, dtype=float).reshape(-1)
    below = np.concatenate([[0], (ctrl < threshold).astype(np.int8), [0]])
    edges = np.diff(below)
    mask = np.zeros(ctrl.size, dtype=bool)
    intervals = 0
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start < min_frames:
            continue
        intervals += 1
        lo, hi = start + margin, end - margin
        if hi > lo:
            mask[lo:hi] = True
    return mask, intervals


class CheckpointMorphologyError(RuntimeError):
    """A checkpoint's tensors do not fit the model built for the target morphology."""


def select_morphology_parameters(target_path, source_path=None):
    """Recognize the target (and optional source) BVH skeleton and set train_vq_vae.param / param2.

    Runs first in main(), before datasets, models or checkpoints are constructed.
    """
    registry = skeletons.load_registry()
    target = skeletons.recognize(target_path, registry)
    train_vq_vae.param = train_vq_vae.parameter_copy(target.param_name)
    source = None
    if source_path:
        source = skeletons.recognize(source_path, registry)
        train_vq_vae.param2 = train_vq_vae.parameter_copy(source.param_name)
    return target, source


def _load_checkpoint(model, checkpoint_path, train_data, device, target, architecture):
    try:
        return train_vq_vae.load_model(model, checkpoint_path, train_data, device)
    except RuntimeError as exc:
        lines = str(exc).splitlines() or [str(exc)]
        mismatch = next((line.strip() for line in lines if "size mismatch" in line), lines[0].strip()).rstrip(".")
        raise CheckpointMorphologyError(
            f"checkpoint {checkpoint_path} does not fit the target morphology {target.morphology} "
            f"({target.param_name}, architecture {architecture}): {mismatch}. "
            "Use a checkpoint trained for this morphology and architecture."
        ) from exc


def resolve_speed_scalar(source_name, target_name, cli_scalar=None, config_path=RETARGETING_CONFIG):
    """Scalar for source -> target. Precedence: --speed-scalar, pair_scalars, default_scalar, 1.0."""
    config = {"default_scalar": 1.0, "pair_scalars": {}}
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    pair = config.get("pair_scalars", {}).get(source_name, {}).get(target_name)
    if cli_scalar is not None:
        scalar, origin = cli_scalar, "--speed-scalar"
    elif pair is not None:
        scalar, origin = pair, "config pair"
    else:
        scalar, origin = config.get("default_scalar", 1.0), "config default"
    scalar = float(scalar)
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"speed scalar must be a finite positive number, got {scalar}")
    return scalar, origin


def main(args):
    # Morphology is recognized from the BVH hierarchies before any dataset, model or checkpoint is built.
    target, source = select_morphology_parameters(args.input_path, args.input_path2)
    train_vq_vae.resolve_runtime_config(args, train_vq_vae.param)
    if getattr(args, "prior_stage_architecture", False):
        train_vq_vae.apply_prior_architecture(train_vq_vae.param)

    speed_scale = 1.0
    scale_origin = None
    root_lock_mask = None  # source-guided stationary root lock; set for cross-morphology runs only
    if source is not None:
        speed_scale, scale_origin = resolve_speed_scalar(source.param_name, target.param_name, args.speed_scalar)
    architecture, architecture_origin = train_vq_vae.effective_architecture(target, args.architecture)
    print(f"Target: {target.param_name}")
    print(f"Source: {source.param_name}" if source is not None else "Source: none")
    # print(f"Architecture: {architecture} ({architecture_origin})")
    print(f"Checkpoint: {args.model_path}")
    # if source is not None:
        # print(f"Speed scalar: {speed_scale!r} ({scale_origin})")
        # print(f"Stationary root lock: {'disabled' if args.no_stationary_root_lock else 'enabled'}")
    # else:
        # print("Speed scalar: not applicable (no source motion)")
        # print("Stationary root lock: not applicable (no source motion)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def merge_cross_morphology_tags(base_tags, source_tags):
        merged = dict(base_tags)
        source_frame_count = source_tags["yaw_quats"].shape[0] if "yaw_quats" in source_tags else None
        base_frame_count = base_tags["yaw_quats"].shape[0] if "yaw_quats" in base_tags else None

        def trim_to_source_frames(value):
            if source_frame_count is None or value.shape[0] <= source_frame_count:
                return value
            return value[:source_frame_count].clone() if isinstance(value, torch.Tensor) else np.copy(value[:source_frame_count])

        def copy_matching_prefix(base_value, source_value):
            copied = base_value.clone() if isinstance(base_value, torch.Tensor) else np.copy(base_value)
            if copied.ndim != source_value.ndim:
                return trim_to_source_frames(copied)

            if source_value.shape[0] < copied.shape[0]:
                copied = copied[: source_value.shape[0]].clone() if isinstance(copied, torch.Tensor) else np.copy(copied[: source_value.shape[0]])

            slices = tuple(slice(0, min(dst, src)) for dst, src in zip(copied.shape, source_value.shape))
            if any(s.stop == 0 for s in slices):
                return copied

            copied[slices] = source_value[slices]
            return copied

        def copy_first_foot_channels(base_value, source_value, foot_axis=1):
            copied = base_value.clone() if isinstance(base_value, torch.Tensor) else np.copy(base_value)
            if source_value.shape[0] < copied.shape[0]:
                copied = copied[: source_value.shape[0]].clone() if isinstance(copied, torch.Tensor) else np.copy(copied[: source_value.shape[0]])

            axis = foot_axis if foot_axis >= 0 else copied.ndim + foot_axis
            if copied.ndim != source_value.ndim or copied.ndim <= axis:
                return trim_to_source_frames(copied)

            copy_frames = min(copied.shape[0], source_value.shape[0])
            copy_feet = min(copied.shape[axis], source_value.shape[axis])
            if copy_frames == 0 or copy_feet == 0:
                return copied

            base_slices = [slice(None)] * copied.ndim
            source_slices = [slice(None)] * source_value.ndim
            base_slices[0] = slice(0, copy_frames)
            source_slices[0] = slice(0, copy_frames)
            base_slices[axis] = slice(0, copy_feet)
            source_slices[axis] = slice(0, copy_feet)

            copied[tuple(base_slices)] = source_value[tuple(source_slices)]
            return copied

        control_keys = (
            "ctrl_velocity",
            "ctrl_acceleration",
            "ctrl_yaw_rate",
            "ctrl_yaw_accel",
            "ctrl_height",
            "ctrl_head_height",
            "ctrl_vertical_velocity",
            "ctrl_roll",
            "yaw_sin",
            "yaw_cos",
            "yaw_quats",
            "sin_diff",
            "cos_diff",
            "smooth_root_pos",
        )
        for key in control_keys:
            if key in source_tags and key in base_tags:
                merged[key] = copy_matching_prefix(base_tags[key], source_tags[key])

        foot_keys = {
            "foot_labels": 1,
            "ctrl_foot_labels": 1,
            "binary_foot_labels": 1,
            "velo_foot": -1,
        }
        for key, foot_axis in foot_keys.items():
            if key in source_tags and key in base_tags:
                merged[key] = copy_first_foot_channels(
                    base_tags[key], source_tags[key], foot_axis=foot_axis
                )

        if train_vq_vae.param.get("foot_contact_conditioning", False) and "foot_contact" in source_tags and base_frame_count is not None:
            # contact comes only from the first min(Ns, Nt) source feet; the rest of the target is invalid
            merged["foot_contact"], merged["foot_contact_valid"] = map_source_foot_contact(
                source_tags["foot_contact"], base_frame_count, foot_count(train_vq_vae.param["feet_idxs"])
            )
        if source_frame_count is not None:
            for key, value in list(merged.items()):
                if (
                    base_frame_count is not None
                    and hasattr(value, "shape")
                    and value.shape[0] == base_frame_count
                ):
                    merged[key] = trim_to_source_frames(value)
        return merged

    eval_dataset = TestMotionData(train_vq_vae.param, train_vq_vae.scale, device)

    filename = os.path.basename(args.input_path)
    dir = args.input_path[: -len(filename)]
    rots, pos, parents, offsets, bvh , og_rots = train_vq_vae.get_info_from_bvh(
        train_vq_vae.get_bvh_from_disk(dir, filename), incremental_rots = False, get_missing_frames = False
    )
    root_rots_euler_y = bvh.data["rotations"][:,0,0] # y x z
    sin = np.sin(np.deg2rad(root_rots_euler_y))
    cos = np.cos(np.deg2rad(root_rots_euler_y))
    sin_cos = np.concatenate([sin[:,None], cos[:,None]], axis=1)
    
    initial_frame = rots[:,0,:] # store for later
    initial_ortho = ortho6d_np.from_quat(initial_frame).reshape(initial_frame.shape[0], -1)
    pos_all_joints = bvh.compute_global_pos()




    if(args.input_path2):
            filename2 = os.path.basename(args.input_path2)
            dir2 = args.input_path2[: -len(filename2)]
            rots2, pos2, parents2, offsets2, bvh2, og_rots2 = train_vq_vae.get_info_from_bvh(
                train_vq_vae.get_bvh_from_disk(dir2, filename2), incremental_rots=False, 
            )
            root_rots_euler_y = bvh2.data["rotations"][:,0,0] # y x z
            sin = np.sin(np.deg2rad(root_rots_euler_y))
            cos = np.cos(np.deg2rad(root_rots_euler_y))
            sin_cos = np.concatenate([sin[:,None], cos[:,None]], axis=1)
            initial_frame = rots2[0:,0,:].copy() # store for later
            initial_ortho = ortho6d_np.from_quat(initial_frame).reshape(initial_frame.shape[0], -1)
            rots2 = quat.compute_incremental_quaternions(rots2)
            root_rots_euler = bvh2.data["rotations"][:,0,:] # y x z
            pos2[0] = np.copy(pos2[1])


            train_data2 = Train_Data(device, train_vq_vae.param2)
            eval_dataset2 = TestMotionData(train_vq_vae.param2, train_vq_vae.scale, device)

            
            pos_all_joints2 = bvh2.compute_global_pos()
            min_len = min(rots.shape[0], rots2.shape[0])
            if min_len % train_vq_vae.param["window_size"] != 0:
               pad_size = train_vq_vae.param["window_size"] - min_len % train_vq_vae.param["window_size"]
               min_len += pad_size # temporary solution, should be fixed in the future

            scalar = speed_scale
            param2 = train_vq_vae.param2


            
            
            def compute_source_tags(source_scalar):
                return compute_tags(pos_all_joints2[:min_len] , downsample_factor=8, is_human=param2["not_dog"], skeleton_height=param2["skeleton_height"],
                                  head_idx = param2["head_idx"], head_height=param2["head_height"], rots=np.deg2rad(root_rots_euler[:min_len])
                                  , scalar = source_scalar, quats = rots2[:min_len], feet_idx=param2["feet_idxs"], shoulder_idx=[5,9], not_dog=param2["not_dog"]
                                  , feet_contact_threshold=param2["feet_contact_threshold"], yaw_error=train_vq_vae.param.get("yaw_error", 0.0))

            tags2 = compute_source_tags(scalar)
            if not args.no_stationary_root_lock:
                # Lock from the unscaled source ctrl_velocity (scalar 1.0), before the pair scalar.
                source_ctrl_unit = compute_source_tags(1.0)["ctrl_velocity"]
                root_lock_mask, n_lock_intervals = stationary_root_lock_mask(source_ctrl_unit)
                # print(
                #     f"Stationary root lock: {int(root_lock_mask.sum())} source frames in {n_lock_intervals} interval(s) "
                #     f"(threshold {STATIONARY_ROOT_LOCK_THRESHOLD} m/s, min {STATIONARY_ROOT_LOCK_MIN_FRAMES} frames, "
                #     f"margin {STATIONARY_ROOT_LOCK_MARGIN})"
                # )

            for key in tags2:
                if not isinstance(tags2[key], torch.Tensor):
                    tags2[key] = torch.tensor(tags2[key]).float().to(device)

            min_len = min(rots.shape[0], rots2.shape[0])
            rots[:min_len,0,:] = rots2[:min_len,0,:]
            og_rots[:min_len,0,:] = og_rots2[:min_len,0,:]
                    



    train_data = Train_Data(device, train_vq_vae.param)
    if args.foot_contact_conditioning:
        train_vq_vae.param["foot_contact_conditioning"] = True
    # The target morphology selects the architecture (or the CLI override); the source only sets param2.
    generator_model = Generator_Model(
        device, train_vq_vae.param, parents, train_data, is_vq_vae=True, architecture=architecture
    ).to(device)
    if args.eval_mode & IK != 0:
        ik_model = IK_Model(device, train_vq_vae.param, parents, train_data).to(device)
    
    generator_model_path = os.path.join(args.model_path, "generator.pt")
    ik_model_path = os.path.join(args.model_path, "ik.pt")
    
    means, stds = _load_checkpoint(generator_model, generator_model_path, train_data, device, target, architecture)
    if args.eval_mode & IK != 0:
        means, stds = _load_checkpoint(ik_model, ik_model_path, train_data, device, target, architecture)
    
    # TestMotionData
    eval_dataset.set_means_stds(means, stds)
    pos_all_joints = bvh.compute_global_pos()
  
    eval_dataset.add_motion(
        offsets,
        pos[:, 0, :],  # only global position
        rots,
        parents,
        bvh,
        filename,
        pos_all_joints,
        og_rots=og_rots,
        end_sites=bvh.data["end_sites"],
        end_sites_parents=bvh.data["end_sites_parents"],
    )
   
    if(args.input_path2):
        eval_dataset.set_tags(
            merge_cross_morphology_tags(eval_dataset.norm_motions[0]["tags"], tags2)
        )
    eval_dataset.normalize()

    phase = eval_dataset.get_phase()

    
    train_data.set_phase(torch.tensor(phase).unsqueeze(0))
    
    results, disp_8, yaw_diff = train_vq_vae.evaluate_generator(generator_model, train_data, eval_dataset)
    if args.plot_prior_logits:
        plot_prior_logits_interactive(
            getattr(generator_model, "output_logits", None),
            step=args.plot_logits_step,
            max_frames=args.plot_logits_max_frames,
        )
    dqs = results[0][0].clone()
    dqs = dqs.permute(0,2,1)
    dqs = dqs.flatten(0,1)
    dqs = dqs * stds["dqs"] + means["dqs"]
    dqs = dqs.reshape(dqs.shape[0], -1, 9)
    dqs = ortho6d.to_dual_quat(dqs)
    trans, rots = from_root_dual_quat_torch(dqs, torch.tensor((bvh.data["parents"])))
    
    if args.eval_mode & IK != 0:
        results_ik = train.evaluate_ik(ik_model, results, train_data, eval_dataset)
        results = results_ik
    
        
    
    eval_path, eval_filename = train_vq_vae.result_to_bvh(
        results[0][0], means, stds, bvh, filename, save=True, disp_8=disp_8[0], initial_frame=initial_frame,
        feet_idx= None, copy_init_frame = False, yaw_diff=yaw_diff[0], initial_sin_cos=sin_cos, initial_ortho = initial_ortho,
        stationary_root_mask = root_lock_mask
    )
    
    # Evaluate Positional Error against the recognized target morphology's non-root sparse joints
    mpjpe, mpeepe = eval_metrics.eval_pos_error(
        train_vq_vae.get_bvh_from_disk(dir, filename),
        train_vq_vae.get_bvh_from_disk(eval_path, eval_filename,True), #<-------------
        device,
        train_vq_vae.param["sparse_joints"][1:],
    )

    print("Evaluate Loss: {}".format(mpjpe + mpeepe))
    print("Mean Per Joint Position Error: {}".format(mpjpe))
    print("Mean End Effector Position Error (MEEPE, non-root sparse joints): {}".format(mpeepe))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Motion Upsampling Network")
    parser.add_argument(
        "model_path",
        type=str,
        help="path to pytorch model folder",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="path to the target-morphology .bvh file: its skeleton selects the model and it names the output",
    )
    parser.add_argument(
        "--input_path2",
        "--control-bvh",
        dest="input_path2",
        type=str,
        help="path to the source .bvh file whose control signals drive the target",
    )
    parser.add_argument(
        "eval_mode",
        type=str.lower,
        choices=["generator", "ik"],
        help="evaluation mode",
    )
    parser.add_argument(
        "--training-stage",
        type=str.lower,
        choices=train_vq_vae.TRAINING_STAGES,
        default=None,
        help="optional stage used when decoder-source is set to stage",
    )
    parser.add_argument(
        "--decoder-source",
        type=str.lower,
        choices=train_vq_vae.DECODER_SOURCES,
        default=None,
        help="decoder input source to use during evaluation",
    )
    parser.add_argument(
        "--speed-scalar",
        type=float,
        default=None,
        help="absolute speed scalar for retargeting; overrides config/retargeting.json (default 1.0)",
    )
    parser.add_argument(
        "--no-stationary-root-lock",
        action="store_true",
        help="disable source-guided stationary root locking (on by default with --control-bvh / --input_path2)",
    )
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=None,
        help="prior sampling temperature during evaluation",
    )
    parser.add_argument(
        "--inference-top-k",
        type=int,
        default=None,
        help="optional top-k restriction for prior sampling during evaluation",
    )
    parser.add_argument(
        "--foot-contact-conditioning",
        action="store_true",
        help="opt-in: prior uses per-foot contact channels (needs a checkpoint trained with it)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=tuple(ARCHITECTURE_MODULES),
        default=None,
        help="autoencoder architecture override for checkpoints trained with a non-registry architecture; "
        "default is the architecture registered for the target morphology",
    )
    parser.add_argument(
        "--prior-stage-architecture",
        action="store_true",
        help="load a prior checkpoint trained with the prior stage's GRU dimensions (stage_overrides['prior']); "
        "historical checkpoints need the default",
    )
    parser.add_argument(
        "--plot-prior-logits",
        action="store_true",
        help="plot prior categorical distributions interactively for sampled frames",
    )
    parser.add_argument(
        "--plot-logits-step",
        type=int,
        default=20,
        help="step size between plotted frames for prior logits",
    )
    parser.add_argument(
        "--plot-logits-max-frames",
        type=int,
        default=100,
        help="maximum number of initial frames considered by the prior logits plotter",
    )
    args = parser.parse_args()
    if args.eval_mode == "generator":
        args.eval_mode = GENERATOR
    elif args.eval_mode == "ik":
        args.eval_mode = IK
    try:
        main(args)
    except (skeletons.SkeletonError, CheckpointMorphologyError) as exc:
        parser.exit(2, f"error: {exc}\n")

    

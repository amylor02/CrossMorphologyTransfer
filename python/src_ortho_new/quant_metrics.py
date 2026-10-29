import argparse
import os
import numpy as np
import torch
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import translation_each_joint
import pymotion.rotations.quat as quat
from motion_data import compute_tags
import train_vq_vae
from generator_architecture import Generator_Model
from train_data import Train_Data
from motion_data import TestMotionData
from phase_visualizer import visualize_contact_motion
from utils.fid_metrics import (
    compute_kinetic_fid,
    compute_motion_fid,
    load_bvh_global_positions,
)
from utils.kinetic_features import (
    SUPPORTED_ACCELERATION_MODES,
    SUPPORTED_UP_AXES,
)
from utils.phase_metrics import (
    compute_continuous_phase_error,
    prepare_contact_labels,
)
from evaluation.legacy_metrics import (
    compare_motion_tags,
    compute_foot_skating_from_bvh,
    compute_jitter_for_directory,
    compute_jitter_from_bvh,
    load_bvh_and_compute_tags,
    timing,
)


def _build_cli_parser():
    """Build the CLI for the motion metric commands."""
    parser = argparse.ArgumentParser(
        description="Motion quality metrics CLI."
    )
    subparsers = parser.add_subparsers(dest="command")

    kinetic_parser = subparsers.add_parser(
        "kinetic-fid",
        help="Compute FID from Fairmotion-style kinetic features.",
    )
    kinetic_parser.add_argument(
        "generated_input",
        help="Path to a generated .bvh file or a directory containing generated .bvh files.",
    )
    kinetic_parser.add_argument(
        "dataset_dir",
        help="Directory containing the real dataset .bvh files.",
    )
    kinetic_parser.add_argument(
        "--root-idx",
        type=int,
        default=0,
        help="Root joint index used for root-relative coordinates.",
    )
    kinetic_parser.add_argument(
        "--frame-time",
        type=float,
        default=None,
        help="Override the BVH frame time in seconds.",
    )
    kinetic_parser.add_argument(
        "--up-axis",
        choices=sorted(SUPPORTED_UP_AXES),
        default="y",
        help="Up axis convention used by the motion data.",
    )
    kinetic_parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Half-window size used by the Fairmotion smoothing rules.",
    )
    kinetic_parser.add_argument(
        "--acceleration-mode",
        choices=sorted(SUPPORTED_ACCELERATION_MODES),
        default="fairmotion",
        help="Use exact Fairmotion parity or the corrected acceleration formula.",
    )
    kinetic_parser.add_argument(
        "--normalize-with-dataset-stats",
        action="store_true",
        help="Normalize generated and dataset kinetic features with dataset mean/std before FID.",
    )
    kinetic_parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="Split motions into non-overlapping full clips of this duration before computing FID.",
    )
    kinetic_parser.set_defaults(handler=_run_kinetic_fid_cli)

    return parser


def _run_kinetic_fid_cli(args):
    """Dispatch the `kinetic-fid` subcommand to the shared Python API."""
    return compute_kinetic_fid(
        generated_input=args.generated_input,
        dataset_dir=args.dataset_dir,
        root_idx=args.root_idx,
        frame_time=args.frame_time,
        up_axis=args.up_axis,
        window_size=args.window_size,
        acceleration_mode=args.acceleration_mode,
        normalize_with_dataset_stats=args.normalize_with_dataset_stats,
        clip_length_seconds=args.clip_seconds,
    )


def main(argv=None):
    """CLI entry point for metric commands."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

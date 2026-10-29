import contextlib
import copy
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from motion_data import TestMotionData, TrainMotionData, compute_phase
from motion.signals import DOWNSAMPLE_FACTOR
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as ortho6d
from pymotion.ops.skeleton import (
    from_root_dual_quat,
    from_root_dual_quat_to_root,
)
from pymotion.io.bvh import BVH
from pymotion.rotations.dual_quat import to_rotation_translation
import numpy as np
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
import time
import argparse
import os
import eval_metrics
import skeleton_recognition as skeletons
from architectures import ARCHITECTURE_MODULES
from training.checkpoints import (
    load_model,
    get_model_paths,
    get_periodic_model_paths,
    save_model,
    save_periodic_model,
)
from training.bvh_export import align_mask_to_frames, result_to_bvh
from evaluation.generation import evaluate_generator, evaluate_ik

scale = 1

# Train Modes
GENERATOR = 1
IK = 2

TRAINING_STAGES = ("autoencoder", "prior")
DECODER_SOURCES = ("stage", "encoder_output", "predicted_codebook", "quantized")

# Continuous AE warmup (--vq-warmup-epochs): latent tokens drawn per source recording for the codebook k-means.
# Windows overlap (window_step < window_size) and recordings differ in length, so without a cap a few long or
# heavily overlapped recordings would dominate the centroids. Recordings under the cap contribute all their valid window tokens.
CODEBOOK_INIT_MAX_TOKENS_PER_RECORDING = 1024

incr_rots = True

human_param = {
    "batch_size": 64,
    "epochs": 1000,
    "kernel_size_temporal_dim": 7,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        # 11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        # 22,  #dummy
    ],
    "window_size": 128,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -11,
    "ema_updates": 96,  # 36,#205,
    "codebook_size": 512,
    "ema_decay": 0.9,
    "skeleton_height": 0.96,
    "head_idx": 13,
    "head_height": 1.6, 
    "feet_idxs": [3, 7, 3, 7], # bwd compatibility
    "feet_contact_threshold": 0.008,
    "not_dog": True,
    "root_branch_dim": 64,
    "gru_window": 8,
    "gru_layers": 2,
    "gru_hidden_dim": 256,
    "gru_vq_dim": 64,
    "input_proj": -1,  # 16,
    "stage_overrides": {
            "prior": {
                "epochs": 35,
                "learning_rate": 1e-3,
                "window_size": 256,
                "window_step": 32,
                "batch_size": 128,
    
                "gru_window": 4,
                "gru_layers": 2,
                "gru_hidden_dim": 512,
                "gru_vq_dim": 128,
    
                "prior_teacher_forcing_prob": 0.8,
                "prior_ctrl_channel_dropout": 0.1,
                "prior_gru_dropout": 0.2,
                "prior_dropout": 0.2,
            },
        },
}


# dog params <-------
dog_param = {
    "batch_size": 128,  # 128,
    "epochs": 1000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # head
        8,  # left hand
        12,  # right hand
        15,  # left foot
        18,  # right foot
        20,  # tail
    ],
    "window_size": 128,  # 128,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 61,  # 131,
    "codebook_size": 512,
    "ema_decay": 0.9,
    "skeleton_height": 0.9,
    "head_idx": 2,
    "head_height": 1.0,
    "feet_idxs": [8, 12, 15, 18],
    "feet_contact_threshold": 0.07,
    "not_dog": False,
    "root_branch_dim": 32,
    "gru_window": 8,
    "gru_layers": 2,
    "gru_hidden_dim": 256,
    "gru_vq_dim": 64,
    "input_proj": -1,
    # Settings of one training stage, applied on top of the keys above by apply_stage_overrides.
    "stage_overrides": {
        "autoencoder": {
            "epochs": 1000,
            "learning_rate": 1e-3,
            "window_size": 128,
            "window_step": 16,
            "batch_size": 128,
        },
        "prior": {
            "epochs": 61,
            "learning_rate": 1e-3,
            "window_size": 128,
            "window_step": 32,
            "batch_size": 128,

            "gru_window": 4,
            "gru_layers": 2,
            "gru_hidden_dim": 512,
            "gru_vq_dim": 128,

            "prior_teacher_forcing_prob": 0.8,
            "prior_ctrl_channel_dropout": 0.1,
            "prior_gru_dropout": 0.2,
            "prior_dropout": 0.2,
        },
    },
}


alligator_param = {
    "batch_size": 32,
    "epochs": 4000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        7,  # head
        10,  # left hand
        13,  # right hand
        18,  # left foot
        21,  # right foot
        24,  # tail
    ],
    "window_size": 32,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 35,
    "codebook_size": 64,
    "ema_decay": 0.5,
}


ostrich_param_ = {
    "batch_size": 32,
    "epochs": 4000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,
        5,
        12,
        13,
        # 16,
        19,
        20,
        # 26,
        26,
        28,
        # 35,
        33,
        35,
        # 43,
        # 50,
        # 51,
        # 52,
        40,
        42,
    ],
    "window_size": 32,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 35,
    "codebook_size": 32,
    "ema_decay": 0.5,
}



cat_param = {
    "batch_size": 16,
    "epochs": 2000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        5,  # tail
        12,  # r toe
        18,  # l toe
        27,  # r finger
        33,  # l finger
        43,  # head
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1,
}


monkey_param = {
    "batch_size": 4,
    "epochs": 300,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        8,  # tail tip
        15,  # right toe 2 tip
        18,  # right toe 1 tip
        21,  # right toe 0 tip
        27,  # left toe 2 tip
        30,  # left toe 1 tip
        33,  # left toe 0 tip
        39,  # left hand
        43,  # left finger 3 tip
        47,  # left finger 2 tip
        51,  # left finger 1 tip
        55,  # left finger 0 tip
        59,  # right hand
        63,  # right finger 3 tip
        67,  # right finger 2 tip
        71,  # right finger 1 tip
        75,  # right finger 0 tip
        78,  # upper lip
        79,  # right eyelid
        80,  # left eyelid
        81,  # right beard
        83,  # lower lip
        84,  # left beard
        85,  # head tip
    ],
    "window_size": 512,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 8, # was 16
    "ema_decay": 0.5,
    "skeleton_height": 0.706879,
    "head_idx": 77,
    "head_height": 0.030743,
    "yaw_error": -np.pi / 2,
    "feet_idxs": [12, 24, 12, 24],
    "feet_contact_threshold": 0.04,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 2,
    "gru_layers": 1,
    "gru_hidden_dim": 32,
    "gru_vq_dim": 32,
    "use_hidden_state": False,
    "input_proj": 32,
    "floor_root_rotation_std": True, # was False
    "root_rotation_std_floor": 0.1,
}


ostrich_param = {
    "batch_size": 16,
    "epochs": 300,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,
        5,
        12,  #
        13,
        15,  #
        16,
        22,  #
        23,
        25,
        26,
        32,
        34,
        35,
        40,
        42,
        43,
        48,  #
        50,
        51,
        52,
        53,
    ],
    "window_size": 512,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 16,
    "ema_decay": 0.5,
    "skeleton_height": 0.69,
    "head_idx": 53,
    "head_height": 1.12,
    "feet_idxs": [20, 10, 20, 10],
    "feet_contact_threshold": 0.04,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 2,
    "gru_layers": 1,
    "gru_hidden_dim": 32,
    "gru_vq_dim": 32,
    "input_proj": 32,
    "floor_root_rotation_std": True,
    "root_rotation_std_floor": 0.1,
}

trex_param = {
    "batch_size": 4,
    "epochs": 300,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,   # Hips/root; first should be root (as assumed by loss.py)
        8,   # jt_Jaw_C
        10,  # jt_Tongue2_C
        11,  # jt_Eye_L
        12,  # jt_Eye_R
        13,  # jt_NeckMuscle_C
        18,  # jt_FingerInner_L
        19,  # jt_HandClawInner_L
        20,  # jt_FingerOutter_L
        21,  # jt_HandClawOutter_L
        26,  # jt_FingerInner_R
        27,  # jt_HandClawInner_R
        28,  # jt_FingerOutter_R
        29,  # jt_HandClawOutter_R
        36,  # jt_Tail6_C
        42,  # jt_ToeMiddle_R
        43,  # jt_ClawMiddle_R
        44,  # jt_ToeInner_R
        45,  # jt_ClawInner_R
        46,  # jt_ToeOutter_R
        47,  # jt_ClawOutter_R
        53,  # jt_ToeMiddle_L
        54,  # jt_ClawMiddle_L
        55,  # jt_ToeInner_L
        56,  # jt_ClawInner_L
        57,  # jt_ToeOutter_L
        58,  # jt_ClawOutter_L
    ],
    "window_size": 512,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 8,
    "ema_decay": 0.5,
    "skeleton_height": 1.09,
    "head_idx": 7,
    "head_height": 1.02,
    "feet_idxs": [41, 52, 42, 53],
    "feet_contact_threshold": 0.04,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 2,
    "gru_layers": 1,
    "gru_hidden_dim": 32,
    "gru_vq_dim": 32,
    "use_hidden_state": True,
    "input_proj": 32,
    "floor_root_rotation_std": True,
    "root_rotation_std_floor": 0.1,
}

scorpion_param = {
    "batch_size": 4,
    "epochs": 300,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,   # Hips/root; first should be root (as assumed by loss.py)
        1,   # Bip01_Neck1
        9,   # Bip01_Tail5
        11,  # Bip01_TailNub
        12,  # Bip01_SpineR
        13,  # Bip01_SpineL
        18,  # Bip01_R_FootNub_1
        23,  # Bip01_R_Finger0
        28,  # Bip01_L_Finger0
        33,  # Bip01_L_FootNub_4
        38,  # Bip01_L_FootNub_3
        43,  # Bip01_L_FootNub_2
        48,  # Bip01_L_FootNub_1
        53,  # Bip01_R_FootNub_2
        58,  # Bip01_R_FootNub_3
        63,  # Bip01_R_FootNub_4
    ],
    "window_size": 512,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 4,
    "ema_decay": 0.5,
    "skeleton_height": 0.073,
    "head_idx": 1,
    "head_height": 0.06,
    "feet_idxs": [18, 48, 53, 43, 58, 38, 63, 33],
    "feet_contact_threshold": 0.01,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 2,
    "gru_layers": 1,
    "gru_hidden_dim": 32,
    "gru_vq_dim": 32,
    "use_hidden_state": False,
    "input_proj": 32,
    "floor_root_rotation_std": True,
    "root_rotation_std_floor": 0.1,
}

dragon_param = {
    "batch_size": 4,
    "epochs": 130,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        # 3-layer pooled survivors from the dragon BVH, mapped back to original joint ids.
        0,    # Hips/root; first should be root (as assumed by loss.py)
        6,    # BN_Feelers08
        8,    # BN_Tail06
        10,   # BN_Feelers07
        16,   # Bip01_R_Toe21
        17,   # Bip01_R_Toe2Nub
        19,   # Bip01_R_Toe11
        20,   # Bip01_R_Toe1Nub
        22,   # Bip01_R_Toe01
        23,   # Bip01_R_Toe0Nub
        29,   # Bip01_L_Toe21
        30,   # Bip01_L_Toe2Nub
        32,   # Bip01_L_Toe11
        33,   # Bip01_L_Toe1Nub
        35,   # Bip01_L_Toe01
        36,   # Bip01_L_Toe0Nub
        39,   # Bip01_Xtra02Nub
        41,   # BN_Feelers06
        42,   # BN_Feelers04
        45,   # Bip01_Xtra03Nub
        46,   # BN_Feelers03
        50,   # BN_RWing01
        54,   # Bip01_Xtra07Nub
        55,   # BN_RWing11
        59,   # Bip01_Xtra08Nub
        62,   # Bip01_Xtra06Nub
        63,   # BN_RWing31
        66,   # Bip01_Xtra05Nub
        67,   # BN_RWing41
        69,   # Bip01_Xtra04Nub
        70,   # Bip01_R_Hand
        74,   # Bip01_R_Finger2Nub
        78,   # Bip01_R_Finger1Nub
        82,   # Bip01_R_Finger0Nub
        86,   # BN_LWing41
        88,   # Bip01_Xtra04OppNub
        89,   # BN_LWing31
        92,   # Bip01_Xtra05OppNub
        93,   # BN_LWing11
        97,   # Bip01_Xtra08OppNub
        100,  # Bip01_Xtra06OppNub
        101,  # BN_LWing01
        105,  # Bip01_Xtra07OppNub
        106,  # Bip01_L_Hand
        110,  # Bip01_L_Finger2Nub
        114,  # Bip01_L_Finger1Nub
        118,  # Bip01_L_Finger0Nub
        122,  # BN_Feelers02
        124,  # Bip01_Xtra_Neck01
        125,  # Bip01_Xtra01Nub
        126,  # Bip01_Head
        127,  # BN_REyeball
        128,  # BN_LEyeball
        132,  # BN_RBeard03
        135,  # BN_LBeard03
        138,  # BN_Tongue03
        140,  # BN_FBeard02
        141,  # Bip01_Ponytail1Nub
        142,  # Bip01_HeadNub
    ],
    "window_size": 512,
    "window_step": 16,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 8,
    "ema_decay": 0.5,
    "skeleton_height": 0.80,
    "head_idx": 142,
    "head_height": 0.43,
    "feet_idxs": [17, 20, 23, 30, 33, 36],
    "feet_contact_threshold": 0.04,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 4,
    "gru_layers": 2,
    "gru_hidden_dim": 32,
    "gru_vq_dim": 32,
    "use_hidden_state": False,
    "input_proj": 32,
    "floor_root_rotation_std": True,
    "root_rotation_std_floor": 0.1,
}

param = ostrich_param.copy()
param2 = human_param.copy()

def ensure_stage_config(config):
    config.setdefault("training_stage", "prior")
    config.setdefault("decoder_source", "stage")
    config.setdefault("soft_codebook_targets", False)
    config.setdefault("soft_codebook_temperature", 0.1)
    config.setdefault("soft_codebook_alpha", 0.5)
    return config


def resolve_runtime_config(args, config):
    ensure_stage_config(config)

    training_stage = getattr(args, "training_stage", None)
    decoder_source = getattr(args, "decoder_source", None)
    soft_codebook_targets = getattr(args, "soft_codebook_targets", None)
    foot_contact_conditioning = getattr(args, "foot_contact_conditioning", None)
    soft_codebook_temperature = getattr(args, "soft_codebook_temperature", None)
    soft_codebook_alpha = getattr(args, "soft_codebook_alpha", None)
    inference_temperature = getattr(args, "inference_temperature", None)
    inference_top_k = getattr(args, "inference_top_k", None)
    if training_stage is not None:
        config["training_stage"] = training_stage
    if decoder_source is not None:
        config["decoder_source"] = decoder_source
    if foot_contact_conditioning:
        config["foot_contact_conditioning"] = True
    if soft_codebook_targets is not None:
        config["soft_codebook_targets"] = soft_codebook_targets
    if soft_codebook_temperature is not None:
        config["soft_codebook_temperature"] = soft_codebook_temperature
    if soft_codebook_alpha is not None:
        config["soft_codebook_alpha"] = soft_codebook_alpha
    if inference_temperature is not None:
        config["inference_temperature"] = inference_temperature
    if inference_top_k is not None:
        config["inference_top_k"] = inference_top_k

    # Keep the VAE bottleneck aligned with the chosen sparse-joint set.
    config["vae_latent_dim"] = (len(config["sparse_joints"]) + 1) * 9 * 8

    return config


def apply_stage_overrides(config, stage=None):
    """Fresh copy of config with the overrides of its training stage applied on top.

    The stage is config["training_stage"] unless one is given. Keys in stage_overrides[stage] replace the flat keys of the
    same name. A config without stage_overrides, or without an entry for the stage, keeps its values.
    """
    stage = str(config.get("training_stage", "prior") if stage is None else stage).lower()
    resolved = dict(config)
    overrides = resolved.pop("stage_overrides", None) or {}
    resolved.update(overrides.get(stage, {}))
    return resolved


PRIOR_ARCHITECTURE_KEYS = ("gru_window", "gru_layers", "gru_hidden_dim", "gru_vq_dim")


def apply_prior_architecture(config):
    """Set config's prior GRU dimensions from stage_overrides["prior"], in place. Used to evaluate prior-stage checkpoints.

    Only the architecture keys are copied. The training window, learning rate, epochs and dropout keep their values, as they
    do not change the shapes of the saved tensors.
    """
    overrides = config.get("stage_overrides", {}).get("prior", {})
    missing = [key for key in PRIOR_ARCHITECTURE_KEYS if key not in overrides]
    if missing:
        raise ValueError(f"no prior-stage architecture for this morphology (missing {', '.join(missing)})")
    for key in PRIOR_ARCHITECTURE_KEYS:
        config[key] = overrides[key]


def pretrained_prior_stats(args, param, train_eval_dir, device):
    """Means and stds saved beside the checkpoint that a prior run loads, or None when the run computes its own.

    The prior is trained on longer windows than the pretrained VQ-VAE, so statistics computed from them would differ from
    the ones that model was trained with. Only a prior run with --load therefore reuses the checkpoint's statistics.
    """
    if not args.load or param["training_stage"] != "prior":
        return None
    data_path, _, _ = get_model_paths(args.name, train_eval_dir)
    data = torch.load(data_path, map_location=device)
    return data["means"], data["stds"]


def configure_ema_reset_interval(config, train_dataloader):
    num_minibatches = len(train_dataloader)
    config["ema_updates"] = max(1, num_minibatches)
    return config["ema_updates"]


def normalize_window_dataset_with_stats(dataset, means, stds):
    dataset.means = means
    dataset.stds = stds
    dataset.norm_motions = []

    for motion in dataset.motions:
        norm_motion = {
            "offsets": (motion["offsets"] - means["offsets"]) / stds["offsets"],
            "dqs": (motion["dqs"] - means["dqs"]) / stds["dqs"],
            "displacement": (motion["displacement"] - means["displacement"])
            / stds["displacement"],
            "disp_8": (motion["disp_8"] - means["disp_8"]) / stds["disp_8"],
            "phase": motion["phase"],
            "loss_weights": motion["loss_weights"],
            "tags": {
                "velo": (motion["tags"]["velo"] - means["velo"]) / stds["velo"],
                "dx": (motion["tags"]["dx"] - means["dx"]) / stds["dx"],
                "dz": (motion["tags"]["dz"] - means["dz"]) / stds["dz"],
                "height_full": (motion["tags"]["height_full"] - means["height_full"])
                / stds["height_full"],
                "alignment": (motion["tags"]["alignment"] - means["alignment"])
                / stds["alignment"],
                "root_rots": (motion["tags"]["root_rots"] - means["root_rots"])
                / stds["root_rots"],
                "direction": (motion["tags"]["direction"] - means["direction"])
                / stds["direction"],
                "velo_foot": motion["tags"]["velo_foot"],
                "head_height": (motion["tags"]["head_height"] - means["head_height"])
                / stds["head_height"],
                "amps": (motion["tags"]["amps"] - means["amps"]) / stds["amps"],
                "sin": (motion["tags"]["sin"] - means["sin"]) / stds["sin"],
                "cos": (motion["tags"]["cos"] - means["cos"]) / stds["cos"],
                "foot_labels": (motion["tags"]["foot_labels"] - means["foot_labels"])
                / stds["foot_labels"],
                "yaw_sin": (motion["tags"]["yaw_sin"] - means["yaw_sin"])
                / stds["yaw_sin"],
                "yaw_cos": (motion["tags"]["yaw_cos"] - means["yaw_cos"])
                / stds["yaw_cos"],
                "yaw_quats": (motion["tags"]["yaw_quats"] - means["yaw_quats"])
                / stds["yaw_quats"],
                "sin_diff": (motion["tags"]["sin_diff"] - means["sin_diff"])
                / stds["sin_diff"],
                "cos_diff": (motion["tags"]["cos_diff"] - means["cos_diff"])
                / stds["cos_diff"],
                "ctrl_velocity": (motion["tags"]["ctrl_velocity"] - means["ctrl_velocity"])
                / stds["ctrl_velocity"],
                "ctrl_acceleration": (
                    motion["tags"]["ctrl_acceleration"] - means["ctrl_acceleration"]
                )
                / stds["ctrl_acceleration"],
                "ctrl_yaw_accel": (motion["tags"]["ctrl_yaw_accel"] - means["ctrl_yaw_accel"])
                / stds["ctrl_yaw_accel"],
                "ctrl_yaw_rate": (motion["tags"]["ctrl_yaw_rate"] - means["ctrl_yaw_rate"])
                / stds["ctrl_yaw_rate"],
                "ctrl_height": (motion["tags"]["ctrl_height"] - means["ctrl_height"])
                / stds["ctrl_height"],
                "ctrl_head_height": (
                    motion["tags"]["ctrl_head_height"] - means["ctrl_head_height"]
                )
                / stds["ctrl_head_height"],
                "ctrl_roll": (motion["tags"]["ctrl_roll"] - means["ctrl_roll"])
                / stds["ctrl_roll"],
                "ctrl_foot_labels": motion["tags"]["ctrl_foot_labels"],
                "foot_contact": motion["tags"]["foot_contact"],
                "foot_contact_valid": motion["tags"]["foot_contact_valid"],
                "ctrl_vertical_velocity": (
                    motion["tags"]["ctrl_vertical_velocity"]
                    - means["ctrl_vertical_velocity"]
                )
                / stds["ctrl_vertical_velocity"],
                "smooth_root_pos": (motion["tags"]["smooth_root_pos"] - means["smooth_root_pos"])
                / stds["smooth_root_pos"],
            },
            "rots": (motion["rots"] - means["rots"]) / stds["rots"],
        }
        dataset.norm_motions.append(norm_motion)


def load_batch_into_train_data(train_data, denorm_motion, norm_motion, rot_order, device):
    train_data.set_offsets(norm_motion["offsets"], denorm_motion["offsets"])
    train_data.set_rot_order(rot_order)
    train_data.set_end_sites(
        denorm_motion["end_sites"], denorm_motion["end_sites_parents"]
    )
    train_data.set_motions(
        norm_motion["dqs"],
        norm_motion["displacement"],
        norm_motion["disp_8"],
        norm_motion["tags"]["sin_diff"],
        norm_motion["tags"]["cos_diff"],
        norm_motion["loss_weights"],
    )
    train_data.set_phase(denorm_motion["phase"])
    train_data.set_phase_per_8_frames(denorm_motion["phase_per_8_frames"])
    train_data.set_velocity_per_8_frames(denorm_motion["velocity_per_8_frames"])
    train_data.set_tags(norm_motion["tags"])
    train_data.set_rots(norm_motion["rots"])
    train_data.set_global_pos(denorm_motion["global_pos"])
    train_data.set_energy(
        denorm_motion["energy_feet"].clone().detach().to(device, dtype=torch.float32)
    )


@torch.no_grad()
def evaluate_prior_validation(generator_model, train_data, eval_dataloader, rot_order, device, seed):
    """Token-weighted prior cross-entropy on validation windows, teacher-forced and free-running.

    teacher_forced_ce: ground-truth codes are fed back. This is the primary checkpoint criterion.
    rollout_ce: the prior free-runs on its own samples, as at inference. A secondary diagnostic.
    The prior runs in eval mode, so dropout is off. Each mode is seeded, the global RNG state is restored, and the
    training mode of every module is restored afterwards. No parameter or buffer is written.
    """
    modes = [(module, module.training) for module in generator_model.modules()]
    generator_model.eval()
    scores = {}
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for key, prior_mode in (("teacher_forced_ce", "teacher_forced"), ("rollout_ce", "rollout")):
            torch.manual_seed(seed)
            cross_entropy = 0.0
            tokens = 0
            for denorm_motion, norm_motion in eval_dataloader:
                load_batch_into_train_data(train_data, denorm_motion, norm_motion, rot_order, device)
                generator_model.forward(prior_eval_mode=prior_mode)
                logits = generator_model.output_logits
                targets = generator_model.target_indices
                cross_entropy += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
                ).item()
                tokens += targets.numel()
            scores[key] = cross_entropy / max(tokens, 1)
    for module, training in modes:
        module.training = training
    return scores


def select_prior_checkpoints(teacher_forced_ce, rollout_position, best_teacher_forced_ce, best_rollout_position):
    """Decide which prior checkpoints an epoch replaces, and the updated best scores.

    The primary checkpoint is selected on teacher-forced CE alone. The rollout checkpoint is selected on rollout
    MPJPE + MEEPE alone. The two comparisons are independent, so both can be saved in the same epoch.
    Returns (save_primary, save_rollout, label, best_teacher_forced_ce, best_rollout_position).
    """
    save_primary = teacher_forced_ce < best_teacher_forced_ce
    save_rollout = rollout_position < best_rollout_position
    if save_primary:
        best_teacher_forced_ce = teacher_forced_ce
    if save_rollout:
        best_rollout_position = rollout_position
    if save_primary and save_rollout:
        label = "Both"
    elif save_primary:
        label = "TF_CE"
    elif save_rollout:
        label = "Rollout_Pos"
    else:
        label = "None"
    return save_primary, save_rollout, label, best_teacher_forced_ce, best_rollout_position


def checkpoint_generator_path(name, train_eval_dir):
    """generator.pt path that get_model_paths returns for name, without creating the folder."""
    model_name = "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    return os.path.join("models", model_name, "generator.pt")


@contextlib.contextmanager
def temporary_weights(generator_model, generator_path):
    """Load the weights of an existing checkpoint for scoring, and restore the model state on exit.

    Used so that a tracker can start from an existing best checkpoint's score. Only the weights are read: the
    train_data normalisation is left unchanged, as load_model would change it.
    """
    snapshot = {key: value.detach().clone() for key, value in generator_model.state_dict().items()}
    device = next(generator_model.parameters()).device
    state = torch.load(generator_path, map_location=device)["model_state_dict"]
    strict = bool(getattr(generator_model.autoencoder, "foot_contact", False))
    generator_model.load_state_dict(state, strict=strict)
    try:
        yield
    finally:
        generator_model.load_state_dict(snapshot)


def evaluate_positions(generator_model, train_data, eval_dataset, train_dataset, eval_dir, device):
    """Positional MPJPE and MEEPE of the full generated motions. Nothing is written to disk, and the RNG state is kept
    so that scoring a checkpoint at startup does not change the training that follows."""
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        results, disp_8, yaw_diff = evaluate_generator(generator_model, train_data, eval_dataset)
        return eval_save_result(
            results,
            train_dataset.means,
            train_dataset.stds,
            eval_dir,
            device,
            save=False,
            disp_8=disp_8,
            yaw_diff=yaw_diff,
        )

param = ensure_stage_config(param)
rm_flag = False
frame_step = 1

assert param["kernel_size_temporal_dim"] % 2 == 1


def parameter_copy(name):
    """Fresh copy of the module-level parameter dictionary named by a registry param_name."""
    params = globals().get(name)
    if not isinstance(params, dict):
        raise skeletons.SkeletonError(
            f"registry refers to parameter set '{name}', which train_vq_vae.py does not define"
        )
    return params.copy()


def effective_architecture(recognition, override=None):
    """(architecture, origin). An explicit override replaces only the registry architecture."""
    if override is not None:
        return override, "CLI override"
    return recognition.architecture, "registry"


def _hierarchy_label(recognition):
    return f"{recognition.morphology}/{recognition.variant}"


def recognize_bvh_dir(directory, registry):
    """Recognition of every .bvh in directory, keyed by file name. Missing or empty directories are rejected."""
    if not os.path.isdir(directory):
        raise skeletons.SkeletonError(f"{directory} does not exist")
    files = sorted(name for name in os.listdir(directory) if name[-4:] == ".bvh")
    if not files:
        raise skeletons.SkeletonError(f"{directory} contains no .bvh files")
    return {name: skeletons.recognize(os.path.join(directory, name), registry) for name in files}


def select_training_morphology(data_path, architecture=None):
    """Recognize the training and eval BVHs of data_path against the skeleton registry.

    All training BVHs must share one registered hierarchy, and every eval BVH must match it.
    Returns (recognition, architecture, origin). Raises SkeletonError before any parameter,
    dataset or model is built.
    """
    registry = skeletons.load_registry()
    train_dir = os.path.join(data_path, "train")
    eval_dir = os.path.join(data_path, "eval")
    train = recognize_bvh_dir(train_dir, registry)
    hierarchies = {}
    for name, found in train.items():
        hierarchies.setdefault(_hierarchy_label(found), []).append(name)
    if len(hierarchies) > 1:
        listing = "; ".join(f"{label} ({', '.join(names)})" for label, names in sorted(hierarchies.items()))
        raise skeletons.SkeletonError(f"training BVHs in {train_dir} mix hierarchies: {listing}")
    recognition = next(iter(train.values()))
    evaluation = recognize_bvh_dir(eval_dir, registry)
    mismatched = [
        f"{name} ({_hierarchy_label(found)})"
        for name, found in evaluation.items()
        if _hierarchy_label(found) != _hierarchy_label(recognition)
    ]
    if mismatched:
        raise skeletons.SkeletonError(
            f"eval BVHs in {eval_dir} do not match the training hierarchy {_hierarchy_label(recognition)}: "
            + ", ".join(mismatched)
        )
    chosen, origin = effective_architecture(recognition, architecture)
    return recognition, chosen, origin


# dog training frames as zero-based [start, end) intervals. end None means the motion end.
# an empty tuple excludes the motion. keys are lower-case filenames including .bvh.
# a mirror_ copy uses the intervals of its original.
DOG_TRAINING_INTERVALS = {
    "d1_053_kan01_001.bvh": ((0, 1454), (1456, None)),
    "d1_ex03_kan02_003.bvh": (),
    "d1_ex03_kan02_006.bvh": (),
    "d1_ex03_kan02_012.bvh": ((691, None),),
    "d1_ex04_kan02_002.bvh": ((198, None),),
    "d1_ex04_kan02_003.bvh": ((0, 53), (126, None)),
}


def dog_curation_source(filename):
    """File whose intervals apply to filename. A mirror_ copy has the same frame timing as its original."""
    key = filename.lower()
    return key[len("mirror_"):] if key.startswith("mirror_") else key


def dog_training_segments(filename, frames, window):
    """Dog training segments of one file as (kept, short) lists of (start, end) frame pairs.

    kept segments are at least one window long. short segments are skipped, since they would need padding.
    Files without an entry in DOG_TRAINING_INTERVALS keep the whole motion.
    """
    intervals = DOG_TRAINING_INTERVALS.get(dog_curation_source(filename), ((0, None),))
    kept, short = [], []
    for start, end in intervals:
        end = frames if end is None else end
        if end > frames:
            raise ValueError(f"{filename}: interval [{start},{end}) is past the {frames} frames")
        if end - start < window:
            short.append((start, end))
        else:
            kept.append((start, end))
    return kept, short


def add_training_motion(train_dataset, filename, is_dog, offsets, rots, pos, parents, pos_all_joints, og_rots, bvh):
    """Add one training BVH. for dog, only the retained intervals are added, each as its own motion."""
    if not is_dog:
        train_dataset.add_motion(
            offsets,
            pos[:, 0, :],  # only global position
            rots,
            parents,
            pos_all_joints,
            og_rots=og_rots,
            end_sites=bvh.data["end_sites"],
            end_sites_parents=bvh.data["end_sites_parents"],
        )
        return

    frames = rots.shape[0]
    kept, short = dog_training_segments(filename, frames, train_dataset.param["window_size"])
    before = len(train_dataset)
    for start, end in kept:
        # copies keep the source arrays unchanged, since add_motion writes into its inputs.
        train_dataset.add_motion(
            offsets.copy(),
            pos[start:end, 0, :].copy(),
            rots[start:end].copy(),
            parents,
            pos_all_joints[start:end].copy(),
            og_rots=None if og_rots is None else og_rots[start:end].copy(),
            end_sites=bvh.data["end_sites"],
            end_sites_parents=bvh.data["end_sites_parents"],
        )
    if dog_curation_source(filename) in DOG_TRAINING_INTERVALS:
        retained = ", ".join(f"[{s},{e})" for s, e in kept) or "excluded"
        skipped = ", ".join(f"[{s},{e})" for s, e in short) or "none"
        print(
            f"dog curation {filename}: {frames} frames, retained {retained}, "
            f"skipped short {skipped}, windows {len(train_dataset) - before}"
        )


def validate_vq_warmup_args(args):
    """Message for an unsupported --vq-warmup-epochs combination, or None when the arguments are accepted."""
    if args.vq_warmup_epochs < 0:
        return "--vq-warmup-epochs must not be negative"
    if args.vq_warmup_epochs == 0:
        return None
    if args.train_mode != GENERATOR:
        return "--vq-warmup-epochs requires train_mode generator"
    if args.training_stage != "autoencoder":
        return "--vq-warmup-epochs requires --training-stage autoencoder"
    if args.load:
        return "--vq-warmup-epochs cannot be combined with --load"
    if args.decoder_source == "encoder_output":
        return "--vq-warmup-epochs cannot be combined with --decoder-source encoder_output (checkpoints would be selected on continuous reconstruction)"
    return None


def validate_reset_prior_args(args):
    """Message for an invalid --reset-prior combination, or None when the arguments are accepted."""
    if not args.reset_prior:
        return None
    if not args.load:
        return "--reset-prior requires --load"
    if args.training_stage != "prior":
        return "--reset-prior requires --training-stage prior"
    if args.train_mode != GENERATOR:
        return "--reset-prior requires train_mode generator"
    return None


def select_codebook_init_windows(train_dataset, recordings, tokens_per_window, max_tokens_per_recording, seed):
    """Deterministic choice of (window, latent token) pairs for the codebook k-means, as {window index: token indices}.

    Only tokens whose eight frames are all real are candidates, so padding never reaches the sample. A recording
    with more candidates than max_tokens_per_recording keeps a seeded random subset of that size.
    """
    rng = np.random.default_rng(seed)
    selection = {}
    for _, first, end in recordings:
        candidates = [
            (window, token)
            for window in range(first, end)
            for token in range(min(train_dataset.motions[window]["valid_frames"] // DOWNSAMPLE_FACTOR, tokens_per_window))
        ]
        if len(candidates) > max_tokens_per_recording:
            picked = rng.choice(len(candidates), size=max_tokens_per_recording, replace=False)
            candidates = [candidates[index] for index in sorted(picked)]
        for window, token in candidates:
            selection.setdefault(window, []).append(token)
    return selection


@torch.no_grad()
def collect_warmup_latents(generator_model, train_data, train_dataset, selection, rot_order, device, batch_size):
    """Pre-quantisation encoder latents of the selected tokens, as [N, channels] on the CPU.

    The warmup model runs in eval mode on the selected windows, so inputs follow the same path as training:
    normalisation, prepare_encoder_input() and the encoder. The RNG state is restored, so training is unaffected.
    """
    windows = sorted(selection)
    loader = DataLoader(Subset(train_dataset, windows), batch_size=batch_size, shuffle=False)
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    latents = []
    generator_model.eval()
    with torch.random.fork_rng(devices=devices):
        position = 0
        for denorm_motion, norm_motion in loader:
            load_batch_into_train_data(train_data, denorm_motion, norm_motion, rot_order, device)
            generator_model.forward()
            encoder_output = generator_model.autoencoder.encoder.enc_out  # [batch, channels, tokens]
            for row in range(encoder_output.shape[0]):
                tokens = torch.as_tensor(selection[windows[position + row]], device=encoder_output.device)
                latents.append(encoder_output[row][:, tokens].transpose(0, 1).cpu())
            position += encoder_output.shape[0]
    return torch.cat(latents)


def fit_codebook_centroids(latents, num_clusters, seed):
    """MiniBatchKMeans centroids of latents [N, D] (float32 numpy), and the number of latents assigned to each."""
    # imported here: only the warmup path needs scikit-learn
    from sklearn.cluster import MiniBatchKMeans

    if latents.shape[0] < num_clusters:
        raise ValueError(f"need at least {num_clusters} latent tokens for the codebook, got {latents.shape[0]}")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=seed, batch_size=4096, n_init=3)
    labels = kmeans.fit(latents).predict(latents)
    counts = np.bincount(labels, minlength=num_clusters)
    return kmeans.cluster_centers_.astype(np.float32), counts


@torch.no_grad()
def codebook_quality(latents, codebook):
    """Nearest-code perplexity, active code count and mean squared quantisation error of latents [N, D]."""
    indices = torch.cdist(latents, codebook).argmin(dim=1)
    used = torch.bincount(indices, minlength=codebook.shape[0]).double()
    used = used[used > 0]
    probs = used / used.sum()
    return {
        "perplexity": torch.exp(-(probs * probs.log()).sum()).item(),
        "active_codes": used.numel(),
        "distortion": ((latents - codebook[indices]) ** 2).mean().item(),
    }


def initialize_codebook_from_warmup(generator_model, train_data, train_dataset, recordings, rot_order, device, param):
    """End the continuous AE warmup: replace the codebook with k-means centroids of the warmup encoder's latents.

    Codebook size, EMA decay and the commitment weight are unchanged. Each centroid's EMA mass is scaled to one training
    minibatch, so the first EMA updates move the codebook at the rate they would in steady state.
    """
    encoder = generator_model.autoencoder.encoder
    tokens_per_window = param["window_size"] // DOWNSAMPLE_FACTOR
    selection = select_codebook_init_windows(
        train_dataset, recordings, tokens_per_window, CODEBOOK_INIT_MAX_TOKENS_PER_RECORDING, param["seed"]
    )
    latents = collect_warmup_latents(
        generator_model, train_data, train_dataset, selection, rot_order, device, param["batch_size"]
    )
    centroids, counts = fit_codebook_centroids(latents.numpy(), encoder.num_embeddings, param["seed"])
    if not np.isfinite(centroids).all():
        raise RuntimeError("codebook k-means produced non-finite centroids")
    tokens_per_minibatch = param["batch_size"] * tokens_per_window
    ema_mass = counts * (tokens_per_minibatch / latents.shape[0])
    encoder.init_codebook_from_centroids(torch.from_numpy(centroids), torch.from_numpy(ema_mass).float())

    generator_model.autoencoder.set_continuous_warmup(False)
    generator_model.train()
    if generator_model.autoencoder._get_decoder_source() != "quantized":
        raise RuntimeError("the decoder does not receive quantized vectors after the warmup")

    stats = codebook_quality(latents, encoder.vq_codebooks[0].weight.detach().cpu())
    print(f"Codebook initialized: {encoder.num_embeddings} entries from {latents.shape[0]} latent tokens in {len(selection)} windows")
    print(
        f"  perplexity: {stats['perplexity']:.2f} - active codes: {stats['active_codes']}/{encoder.num_embeddings}"
        f" - quantization distortion: {stats['distortion']:.6f}"
    )


class CodebookUsage:
    """Code usage and quantisation distortion over one VQ epoch. Collected only in runs that use a warmup."""

    def __init__(self, num_codes, device):
        self.counts = torch.zeros(num_codes, dtype=torch.float64, device=device)
        self.distortion = 0.0
        self.steps = 0

    @torch.no_grad()
    def update(self, generator_model):
        autoencoder = generator_model.autoencoder
        indices = generator_model.target_indices.reshape(-1)
        self.counts += torch.bincount(indices, minlength=self.counts.numel()).double()
        self.distortion += F.mse_loss(autoencoder.encoder.enc_out, autoencoder.quantized_latent).item()
        self.steps += 1

    def summary(self, epoch):
        used = self.counts[self.counts > 0]
        probs = used / used.sum()
        perplexity = torch.exp(-(probs * probs.log()).sum()).item()
        return (
            "Epoch: {} [vq] codebook - perplexity: {:.2f} - active codes: {}/{} - quantization distortion: {:.6f}".format(
                epoch, perplexity, used.numel(), self.counts.numel(), self.distortion / max(self.steps, 1)
            )
        )


def main(args):
    global param
    # Morphology and architecture come from the BVH hierarchies, before any parameter, dataset or model is built.
    recognition, architecture, origin = select_training_morphology(
        args.data_path, getattr(args, "architecture", None)
    )
    param = parameter_copy(recognition.param_name)
    resolve_runtime_config(args, param)
    param = apply_stage_overrides(param)
    print(f"Morphology: {_hierarchy_label(recognition)} -> {recognition.param_name}")
    print(f"Architecture: {architecture} ({origin})")

    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    train_eval_dir = args.data_path
    best_root_dir = os.path.join(train_eval_dir, "best_root")
    if not os.path.exists(best_root_dir):
        os.makedirs(best_root_dir)
    train_dir = os.path.join(train_eval_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("train directory does not exist")
    train_files = os.listdir(train_dir)
    eval_dir = os.path.join(train_eval_dir, "eval")
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")
    eval_files = os.listdir(eval_dir)
    is_dog = recognition.param_name == "dog_param"
    train_dataset = TrainMotionData(param, scale, device)
    # (filename, first, end) window range of each training BVH in train_dataset, for the codebook init sampling cap
    train_recordings = []
    eval_dataset = TestMotionData(param, scale, device)
    # evaluation keeps the historical window rule
    eval_window_dataset = TrainMotionData(param, scale, device, include_final_window=False)
    reference_parents = None  # used to make sure all bvh have the same structure

    rots_missing_frames_list = []
    ground_truth_rots_list = []
    phase_list = []
    velocities_list = []
    phase_chunks = []
    velocity_chunks = []
    for filename in train_files:
        if filename[-4:] == ".bvh":

            bvh_from_disk = get_bvh_from_disk(train_dir, filename)
            rots, pos, parents, offsets, _, og_rots = get_info_from_bvh(
                bvh_from_disk, get_missing_frames=False
            )
            # rots[0,0,:] = rots_missing_frames[0,0,:]
            ########################
            # create chunks of 256 frames each (for root predicotr)
            # list_og_rots = rots_missing_frames[0]
            # list_missing = rots_missing_frames[1]
            # num_frames = list_missing[0].shape[0]
            # chunk_size = 256
            # for i in range(frame_step):
            #     rots_missing_frames = list_missing[i]
            #     true_rots = list_og_rots[i]
            #     for start in range(0, num_frames, chunk_size):
            #         end = start + chunk_size
            #         current_chunk = torch.tensor(rots_missing_frames[start:end]).float()

            #         # Check if current chunk is smaller than chunk_size and pad if necessary
            #         if current_chunk.shape[0] < chunk_size:
            #             padding = torch.zeros((chunk_size - current_chunk.shape[0], current_chunk.shape[1]))
            #             padding_gr_tr = torch.zeros((chunk_size - current_chunk.shape[0], ground_truth_chunk.shape[1]))
            #             current_chunk = torch.cat((current_chunk, padding), dim=0)
            #             ground_truth_chunk = torch.tensor(true_rots[start:end, 0, :]).float()  # actual root rotations
            #             ground_truth_chunk = torch.cat((ground_truth_chunk, padding_gr_tr.clone()), dim=0)
            #         else:
            #             ground_truth_chunk = torch.tensor(true_rots[start:end, 0, :]).float()


            if reference_parents is None:
                reference_parents = parents.copy()
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            pos_all_joints = bvh_from_disk.compute_global_pos()

            first_window = len(train_dataset)
            add_training_motion(
                train_dataset, filename, is_dog, offsets, rots, pos, parents, pos_all_joints, og_rots, bvh_from_disk
            )
            train_recordings.append((filename, first_window, len(train_dataset)))
            chunk_size = 512
            num_frames = pos_all_joints.shape[0]

            # Initialize lists to store results

    #         # Process pos_all_joints in chunks of 512 frames
    #         for start_index in range(0, num_frames, chunk_size):
    #             end_index = min(start_index + chunk_size, num_frames)  # Ensure we do not exceed array bounds
    #             pos_chunk = pos_all_joints[start_index:end_index]  # Shape: [frames_in_chunk, joints, 3]

    #             # Compute phase and velocity for the current chunk
    #             phase, velocity_xz = compute_phase(pos_chunk, indices_only=False)
    #             # import matplotlib.pyplot as plt
    #             # plt.figure(figsize=(12, 6))
    #             # # Plot Part 1
    #             # plt.subplot(2, 1, 1)
    #             # plt.plot(phase[0], label='Real Phase Part 1', color='blue')
    #             # plt.legend()

    #             # plt.tight_layout()
    #             # plt.show()

    #             # Pad phase and velocity if necessary
    #             if phase.shape[1] < chunk_size:
    #                 phase = np.pad(phase, ((0, 0), (0, chunk_size - phase.shape[1])), 'constant')  # Pad along frames dimension
    #             if len(velocity_xz) < chunk_size:
    #                 velocity_xz = np.pad(velocity_xz, (0, chunk_size - len(velocity_xz)), 'constant')  # Pad with zeros


    # print(len(rots_missing_frames_list), len(ground_truth_rots_list))


    # Once all train files are added, compute the means and stds and normalize. A prior run with --load uses the
    # statistics of the loaded checkpoint instead, so its windows, validation windows and saved data.pt all match it.
    pretrained_stats = pretrained_prior_stats(args, param, train_eval_dir, device)
    if pretrained_stats is None:
        train_dataset.normalize()
    else:
        normalize_window_dataset_with_stats(train_dataset, *pretrained_stats)
        print("Normalization: statistics of the loaded checkpoint")
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh, og_rots = get_info_from_bvh(
                get_bvh_from_disk(eval_dir, filename)
            )
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            pos_all_joints = bvh.compute_global_pos()
            # Eval Dataset

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
            eval_window_dataset.add_motion(
                offsets,
                pos[:, 0, :],
                rots,
                parents,
                pos_all_joints,
                og_rots=og_rots,
                end_sites=bvh.data["end_sites"],
                end_sites_parents=bvh.data["end_sites_parents"],
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()
    normalize_window_dataset_with_stats(
        eval_window_dataset, train_dataset.means, train_dataset.stds
    )

    train_dataloader = DataLoader(train_dataset, param["batch_size"], shuffle=True)
    eval_window_dataloader = DataLoader(
        eval_window_dataset, param["batch_size"], shuffle=False
    )
    ema_reset_interval = configure_ema_reset_interval(param, train_dataloader)
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(
        device, param, reference_parents, train_data, is_vq_vae=True, architecture=architecture
    ).to(device)
    if args.train_mode & IK != 0:
        ik_model = IK_Model(device, param, reference_parents, train_data).to(device)
    train_data.set_means(train_dataset.means["dqs"])
    train_data.set_stds(train_dataset.stds["dqs"])
    train_data.set_root_means_stds(
        train_dataset.means["rots"], train_dataset.stds["rots"]
    )
    train_data.set_sin_cos_means_stds(
        train_dataset.means["yaw_sin"],
        train_dataset.means["yaw_cos"],
        train_dataset.stds["yaw_sin"],
        train_dataset.stds["yaw_cos"],
    )
    reference_rot_order = bvh.data["rot_order"]
    print(args)
    print("EMA dead-code reset interval (minibatches):", ema_reset_interval)
    print(args.train_mode & IK)
    _, generator_path, ik_path = get_model_paths(args.name, train_eval_dir)
    reset_prior = bool(getattr(args, "reset_prior", False))
    if args.train_mode & GENERATOR == 0 or (args.load):
        print("loading pretrained model")
        # Generator is always needed with IK, load it if not training it
        load_model(generator_model, generator_path, train_data, device, reset_prior=reset_prior)
    if args.train_mode & IK != 0 and args.load:
        load_model(ik_model, ik_path, train_data, device)

    use_prior_eval_metric = (
        args.train_mode & GENERATOR != 0 and not generator_model._is_autoencoder_stage()
    )
    if use_prior_eval_metric:
        # freeze the backbone before the first prior forward, which includes the initial evaluation below
        generator_model._set_trainable_parameters()

    # Prior trackers. A fresh run, or a reset run (--reset-prior), starts both at infinity and inspects no existing
    # checkpoint. A resumed run starts each tracker from its existing checkpoint, and saves the loaded model as the
    # rollout checkpoint when that one is missing, so every tracked best score has a saved checkpoint.
    best_teacher_forced_ce = float("inf")
    best_rollout_position = float("inf")
    rollout_generator_path = None
    if use_prior_eval_metric:
        rollout_generator_path = checkpoint_generator_path(args.name + "_rollout", train_eval_dir)
        if args.load and not reset_prior:
            if os.path.exists(generator_path):
                with temporary_weights(generator_model, generator_path):
                    best_teacher_forced_ce = evaluate_prior_validation(
                        generator_model, train_data, eval_window_dataloader, reference_rot_order, device, param["seed"]
                    )["teacher_forced_ce"]
            if os.path.exists(rollout_generator_path):
                with temporary_weights(generator_model, rollout_generator_path):
                    mpjpe, mpeepe = evaluate_positions(
                        generator_model, train_data, eval_dataset, train_dataset, eval_dir, device
                    )
                best_rollout_position = mpjpe + mpeepe
            else:
                mpjpe, mpeepe = evaluate_positions(
                    generator_model, train_data, eval_dataset, train_dataset, eval_dir, device
                )
                best_rollout_position = mpjpe + mpeepe
                save_model(generator_model, None, train_dataset, args.name + "_rollout", train_eval_dir)

    if (args.train_mode & GENERATOR == 0 or args.train_mode & IK == 0) and args.load:
        # Resumed runs start the best-checkpoint tracker from the loaded generator (prior runs set theirs above).
        if not use_prior_eval_metric:
            results, disp_8, yaw_diff = evaluate_generator(
                generator_model, train_data, eval_dataset
            )
            if args.train_mode & IK != 0:
                results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
                results = results_ik
            mpjpe, mpeepe = eval_save_result(
                results,
                train_dataset.means,
                train_dataset.stds,
                eval_dir,
                device,
                save=False,
                disp_8=disp_8,
                yaw_diff=yaw_diff,
            )
            best_evaluation = mpjpe + mpeepe
    else:
        best_evaluation = float("inf")
    # Training Loop
    # With --vq-warmup-epochs N, epochs 0..N-1 train encoder and decoder without quantisation (phase "warmup").
    # Epoch N initialises the codebook and starts VQ training, which runs for param["epochs"] epochs (phase "vq").
    warmup_epochs = int(getattr(args, "vq_warmup_epochs", 0) or 0)
    if warmup_epochs > 0:
        generator_model.autoencoder.set_continuous_warmup(True)
    start_time = time.time()

    for epoch in range(warmup_epochs + param["epochs"]):
        in_warmup = epoch < warmup_epochs
        vq_epoch = epoch - warmup_epochs  # equals epoch when warmup is disabled
        if warmup_epochs > 0 and epoch == warmup_epochs:
            print("Initializing VQ codebook from pretrained encoder...")
            initialize_codebook_from_warmup(
                generator_model, train_data, train_dataset, train_recordings, reference_rot_order, device, param
            )
            # warmup scores never selected a checkpoint, so VQ checkpoint selection starts from scratch
            best_evaluation = float("inf")
        phase_tag = "" if warmup_epochs == 0 else (" [warmup]" if in_warmup else " [vq]")
        usage = CodebookUsage(param["codebook_size"], device) if warmup_epochs > 0 and not in_warmup else None
        avg_train_loss = 0.0
        avg_vq_loss = 0.0
        avg_velo_loss = 0.0
        mpjpe = float("nan")
        mpeepe = float("nan")

        if use_prior_eval_metric and hasattr(generator_model.autoencoder, "codebook_predictor2"):
            generator_model.autoencoder.codebook_predictor2.anneal_teacher_forcing(
                factor=0.999999999
            )

        for step, (denorm_motion, norm_motion) in enumerate(train_dataloader):
            load_batch_into_train_data(
                train_data, denorm_motion, norm_motion, reference_rot_order, device
            )

            if args.train_mode & GENERATOR != 0:
                generator_model.train()
                generator_model.configure_prior_train_mode()
            if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                res_decoder, disp_8, yaw_diff = generator_model.forward()
                if usage is not None:
                    usage.update(generator_model)
            if args.train_mode & IK != 0:
                ik_model.train()
                ik_model.forward(res_decoder)
            loss = 0.0
            kld_loss = 0.0
            if args.train_mode & GENERATOR != 0:

                # Decoder-root MSE is diagnostic-only (zero-weighted) and is not logged.
                loss_generator, vq_loss, velo_loss, _ = (
                    generator_model.optimize_parameters_vq_vae(
                        debug_context={
                            "epoch": epoch,
                            "step": step,
                            "num_steps": len(train_dataloader),
                        }
                    )
                )
                loss = loss_generator.item()
                vq_loss = vq_loss.item()
                velo_loss = velo_loss.item()

            if args.train_mode & IK != 0:
                loss_ik = ik_model.optimize_parameters(res_decoder)
                loss += loss_ik.item()

            avg_train_loss += loss
            avg_vq_loss += vq_loss
            avg_velo_loss += velo_loss
            if step == len(train_dataloader) - 1:
                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    if use_prior_eval_metric:
                        prior_scores = evaluate_prior_validation(
                            generator_model, train_data, eval_window_dataloader, reference_rot_order, device, param["seed"]
                        )
                    results, disp_8, yaw_diff = evaluate_generator(
                        generator_model, train_data, eval_dataset
                    )

                    if args.train_mode & IK != 0:
                        results_ik = evaluate_ik(
                            ik_model,
                            results,
                            train_data,
                            eval_dataset,
                        )
                        results = results_ik
                    # warmup evaluations write nothing; VQ epochs count from the start of VQ training
                    save = not in_warmup and vq_epoch % 50 == 0
                    mpjpe, mpeepe = eval_save_result(
                        results,
                        train_dataset.means,
                        train_dataset.stds,
                        eval_dir,
                        device,
                        save=save,
                        disp_8=disp_8,
                        yaw_diff=yaw_diff,
                    )
                    if not use_prior_eval_metric:
                        evaluation_loss = mpjpe + mpeepe
                was_best = False
                best_label = None
                if in_warmup:
                    pass  # warmup scores are only reported, never used to select or save a checkpoint
                elif use_prior_eval_metric:
                    # two independent selections: teacher-forced CE for the primary checkpoint, rollout MPJPE + MEEPE
                    # for the rollout checkpoint. Both are saved when both improve in the same epoch.
                    save_primary, save_rollout, best_label, best_teacher_forced_ce, best_rollout_position = (
                        select_prior_checkpoints(
                            prior_scores["teacher_forced_ce"], mpjpe + mpeepe,
                            best_teacher_forced_ce, best_rollout_position,
                        )
                    )
                    if save_primary:
                        save_model(
                            generator_model if args.train_mode & GENERATOR != 0 else None,
                            ik_model if args.train_mode & IK != 0 else None,
                            train_dataset,
                            args.name,
                            train_eval_dir,
                        )
                    if save_rollout:
                        save_model(
                            generator_model if args.train_mode & GENERATOR != 0 else None,
                            None,
                            train_dataset,
                            args.name + "_rollout",
                            train_eval_dir,
                        )
                    was_best = save_primary or save_rollout
                    if not was_best and vq_epoch % 10 == 0 and vq_epoch != 0:
                        save_periodic_model(
                            generator_model if args.train_mode & GENERATOR != 0 else None,
                            ik_model if args.train_mode & IK != 0 else None,
                            train_dataset,
                            args.name,
                            train_eval_dir,
                        )
                elif evaluation_loss < best_evaluation:
                    save_model(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )
                    best_evaluation = evaluation_loss
                    was_best = True

                elif vq_epoch % 10 == 0 and vq_epoch != 0:
                    # Periodic snapshot every 10 epochs, written to best_root/ (not the best-model checkpoint).
                    save_periodic_model(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )

                # Print
                avg_train_loss /= len(train_dataloader)
                avg_vq_loss /= len(train_dataloader)
                avg_velo_loss /= len(train_dataloader)

                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    if in_warmup:
                        print(
                            "Epoch: {} [warmup] - Recon Loss: {:.4f} - MPJPE: {:.4f} - MEEPE: {:.4f}".format(
                                epoch, avg_train_loss, mpjpe, mpeepe
                            )
                        )
                    elif use_prior_eval_metric:
                        print(
                            "Epoch: {} "
                            "Train CE: {:.4f} "
                            "TF Val CE: {:.4f} "
                            "Rollout Val CE: {:.4f} "
                            "Rollout MPJPE: {:.4f} "
                            "Rollout MEEPE: {:.4f} "
                            "Best: {}".format(
                                epoch,
                                avg_train_loss,
                                prior_scores["teacher_forced_ce"],
                                prior_scores["rollout_ce"],
                                mpjpe,
                                mpeepe,
                                best_label,
                            )
                        )
                    else:
                        print(
                            "Epoch: {}{} - Train Loss: {:.4f} - vq_loss: {:.14f} - velo_loss: {:.4f} - Eval Loss: {:.4f}".format(
                                vq_epoch,
                                phase_tag,
                                avg_train_loss,
                                avg_vq_loss,
                                avg_velo_loss,
                                evaluation_loss,
                            )
                            + ("*" if was_best else "")
                        )
                if usage is not None:
                    print(usage.summary(vq_epoch))

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
        load_model(generator_model, generator_path, train_data, device)
        if use_prior_eval_metric:
            final_prior_scores = evaluate_prior_validation(
                generator_model, train_data, eval_window_dataloader, reference_rot_order, device, param["seed"]
            )
            evaluation_loss = final_prior_scores["teacher_forced_ce"]
        results, disp_8, yaw_diff = evaluate_generator(
            generator_model, train_data, eval_dataset
        )
        if args.train_mode & IK != 0:
            load_model(ik_model, ik_path, train_data, device)
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik

        mpjpe, mpeepe = eval_save_result(
            results,
            train_dataset.means,
            train_dataset.stds,
            eval_dir,
            device,
            disp_8=disp_8,
            yaw_diff=yaw_diff,
        )
        if not use_prior_eval_metric:
            evaluation_loss = mpjpe + mpeepe

    print("Evaluate Loss: {}".format(evaluation_loss))
    if args.train_mode & (GENERATOR | IK) != 0:
        print("Mean Per Joint Position Error: {}".format(mpjpe))
        print("Mean End Effector Position Error (MEEPE, non-root sparse joints): {}".format(mpeepe))
    if use_prior_eval_metric:
        print("Primary checkpoint (teacher-forced CE {}): {}".format(evaluation_loss, generator_path))
        print("Rollout validation CE: {}".format(final_prior_scores["rollout_ce"]))
        rollout_location = rollout_generator_path if os.path.exists(rollout_generator_path) else "none"
        print("Best rollout positional score (MPJPE + MEEPE): {} at {}".format(best_rollout_position, rollout_location))


def eval_save_result(
    results,
    train_means,
    train_stds,
    eval_dir,
    device,
    save=True,
    disp_8=None,
    yaw_diff=None,
):
    # MEEPE uses the non-root sparse joints of the morphology being evaluated (module-level param)
    sparse_joint_indices = param["sparse_joints"][1:]
    array_mpjpe = np.empty((len(results),))
    array_mpeepe = np.empty((len(results),))
    for step, (res, bvh, filename) in enumerate(results):
        # result_to_bvh overwrites the BVH it is given. Reconstruct into a copy so the cached evaluation BVH
        # (shared by every evaluation of this dataset) keeps its source positions and rotations.
        recon_bvh = copy.deepcopy(bvh)
        if save:
            eval_path, eval_filename = result_to_bvh(
                res,
                train_means,
                train_stds,
                recon_bvh,
                filename,
                disp_8=disp_8[step],
                yaw_diff=yaw_diff[step],
            )
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                get_bvh_from_disk(eval_path, eval_filename, True),  # <---------
                device,
                sparse_joint_indices,
            )
        else:
            result_to_bvh(res, train_means, train_stds, recon_bvh, None, save=False)
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                recon_bvh,
                device,
                sparse_joint_indices,
            )

        array_mpjpe[step] = mpjpe
        array_mpeepe[step] = mpeepe

    return np.mean(array_mpjpe), np.mean(array_mpeepe)














def get_bvh_from_disk(path, filename, remove=True):
    remove = rm_flag
    path = os.path.join(path, filename)
    bvh = BVH()
    bvh.load(path)
    # remove lower body joints
    if remove:
        bvh.remove_joints([19, 20])
    return bvh


def get_info_from_bvh(
    bvh, incremental_rots=False, get_missing_frames=False, get_phase=False
):
    rot_roder = np.tile(
        bvh.data["rot_order"][0],
        (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1),
    )  # made a change here
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    og_rots = rots.copy()

    pos = bvh.data["positions"]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint

    if incremental_rots:
        rots = quat.compute_incremental_quaternions(og_rots)

    if get_missing_frames:
        list_og_rots = []
        list_missing_frames = []
        foot_1_idx = 15
        foot_2_idx = 18

        for i in range(frame_step):
            rots_missing_frames = quat.compute_incremental_intermittent_quaternions(
                og_rots[i:, :, :], pos[i:, 0, :]
            )
            phase, _ = compute_phase(pos[i:], joint_id=[0])
            phase = phase.squeeze()
            rots_missing_frames = np.concatenate(
                [rots_missing_frames, phase[:, np.newaxis]], axis=1
            )
            og_quats = og_rots[
                i:, :, :
            ]  # og_quats = quat.compute_incremental_quaternions(og_rots[i:,:,:])
            displacement = pos[i + 1 :, 0, :] - pos[i:-1, 0, :]
            displacement = np.vstack((np.zeros((1, 3)), displacement))
            true_values = np.zeros(
                (og_quats.shape[0], og_quats.shape[1], 7)
            )  # [frames, joints, quat(4)+displacement(3)+4+4(feet)]
            true_values[:, 0, :] = np.concatenate(
                (og_quats[:, 0, :], displacement), axis=-1
            )
            list_og_rots.append(true_values)  # actual rotations and displacements
            list_missing_frames.append(rots_missing_frames)

        return rots, pos, parents, offsets, bvh, (list_og_rots, list_missing_frames)

    if get_phase:
        list_og_rots = []
        list_missing_frames = []
        foot_1_idx = 15
        foot_2_idx = 18

        return rots, pos, parents, offsets, bvh, (list_og_rots, list_missing_frames)

    return rots, pos, parents, offsets, bvh, og_rots





















# def compute_phase(pos):
#     # Assuming x_values, z_values, and y_values are 1D arrays representing the movement over time
#     x_values = pos[:,0,0]  # X-axis movement data
#     z_values = pos[:,0,2]  # Z-axis movement data
#     y_values = pos[:,0,1]  # Y-axis movement data
#     epsilon = 0.01  # Stationary threshold

#     # Step 2: Compute xz plane velocity
#     vx = np.diff(x_values, prepend=x_values[0])
#     vz = np.diff(z_values, prepend=z_values[0])
#     velocity_xz = np.sqrt(vx**2 + vz**2)

#     # Step 3: Discard low velocities
#     velocity_threshold = 0.01  # Set your threshold here
#     active_indices = np.where(velocity_xz > velocity_threshold)[0]
#     inactive_indices = np.where(velocity_xz < velocity_threshold)[0]
#     # Step 4: Extract active y values
#     active_y_values = y_values[active_indices]

#     # Step 5: Smooth the active y values for better peak detection
#     smoothed_y = gaussian_filter1d(y_values, sigma=5)

#     # Step 7: Locate peaks and valleys
#     peaks, _ = find_peaks(smoothed_y,)
#     valleys, _ = find_peaks(-smoothed_y,)


#     def construct_signal(peaks, valleys, num_points):
#         """
#         """
#         # Initialize the signal array
#         signal = np.zeros(num_points)
#         # Ensure peaks and valleys are sorted
#         points = np.concatenate((peaks,valleys))
#         points = sorted(points)
#         # Iterate through the points to construct the signal
#         for i in range(len(points) - 1):
#             start = points[i]
#             end = points[i + 1]
#             x = np.linspace(0, 1, end - start)  # Normalize to 0 to 1
#             if points[i] in peaks:
#                 # If current point is a peak
#                 signal[start:end] = 0.5 * (1 + np.cos(np.pi * x))  # Descend to valley (sine wave)
#             elif points[i] in valleys:
#                 # If current point is a valley
#                 signal[start:end] = 0.5 * (1 - np.cos(np.pi * x))  # Ascend to peak (sine wave)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of the experiment, used to save the model and the logs",
    )
    parser.add_argument(
        "train_mode",
        type=str.lower,
        choices=["generator", "ik", "all"],
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="load the model(s) from a checkpoint",
    )
    parser.add_argument(
        "--reset-prior",
        action="store_true",
        default=False,
        help="with --load and --training-stage prior: keep the pretrained generator but a freshly initialised prior, and start the best-checkpoint trackers from scratch. Requires train_mode generator",
    )
    parser.add_argument(
        "--training-stage",
        type=str.lower,
        choices=TRAINING_STAGES,
        default=None,
        help="training stage: autoencoder trains the VQ-VAE, prior trains only the recurrent prior",
    )
    parser.add_argument(
        "--decoder-source",
        type=str.lower,
        choices=DECODER_SOURCES,
        default=None,
        help="decoder input source used during evaluation paths",
    )
    parser.add_argument(
        "--foot-contact-conditioning",
        action="store_true",
        default=None,
        help="opt-in: prior uses per-foot contact channels instead of velocity and acceleration (needs a checkpoint trained with it)",
    )
    parser.add_argument(
        "--soft-codebook-targets",
        action="store_true",
        default=None,
        help="train prior with mixed hard CE and soft codebook-neighbour targets",
    )
    parser.add_argument(
        "--soft-codebook-temperature",
        type=float,
        default=None,
        help="temperature for soft codebook-neighbour targets",
    )
    parser.add_argument(
        "--soft-codebook-alpha",
        type=float,
        default=None,
        help="weight for soft codebook loss in mixed prior CE loss",
    )
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=None,
        help="prior sampling temperature used by evaluation/inference paths",
    )
    parser.add_argument(
        "--inference-top-k",
        type=int,
        default=None,
        help="optional top-k restriction for prior sampling during evaluation/inference",
    )
    parser.add_argument(
        "--vq-warmup-epochs",
        type=int,
        default=0,
        help="continuous AE pretraining epochs before VQ training (default: disabled). Needs --training-stage autoencoder; these epochs are added to the VQ epoch budget",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=tuple(ARCHITECTURE_MODULES),
        default=None,
        help="autoencoder architecture override; default is the architecture registered for the training morphology",
    )
    args = parser.parse_args()
    if args.train_mode == "generator":
        args.train_mode = GENERATOR
    elif args.train_mode == "ik":
        args.train_mode = IK
    elif args.train_mode == "all":
        args.train_mode = GENERATOR | IK
    error = validate_vq_warmup_args(args)
    if error is not None:
        parser.error(error)
    error = validate_reset_prior_args(args)
    if error is not None:
        parser.error(error)
    try:
        main(args)
    except skeletons.SkeletonError as exc:
        parser.exit(2, f"error: {exc}\n")

"""Backward-compatible imports for motion datasets and helpers."""

import numpy as np
import torch
from torch.utils.data import Dataset
import pymotion.rotations.dual_quat as dquat
from pymotion.ops.skeleton import to_root_dual_quat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d_torch as ortho6d
import matplotlib.pyplot as plt
from utils import network_timing

from motion.datasets import (
    TrainMotionData,
    TestMotionData,
    RunMotionData,
)
from motion.signals import (
    USE_GLOBAL_HEIGHT,
    DOWNSAMPLE_FACTOR,
    WINDOW_STEP_SIZE,
    calc_velocity_from_pos,
    calc_feet_energy,
    calc_feet_energy_torch,
    compute_foot_labels,
    compute_foot_labels2,
    compute_tags,
)
from motion.phase import (
    compute_phase,
    construct_signal,
)

__all__ = [
    'TrainMotionData',
    'TestMotionData',
    'RunMotionData',
    'USE_GLOBAL_HEIGHT',
    'DOWNSAMPLE_FACTOR',
    'WINDOW_STEP_SIZE',
    'calc_velocity_from_pos',
    'calc_feet_energy',
    'calc_feet_energy_torch',
    'compute_foot_labels',
    'compute_foot_labels2',
    'compute_tags',
    'compute_phase',
    'construct_signal',
]

"""Backward-compatible imports for networks.generator."""

import importlib
import torch
import torch.nn as nn
from architectures import ARCHITECTURE_MODULES, DEFAULT_ARCHITECTURE
from loss import MSE_DQ
from pymotion.ops.skeleton import (
    from_root_dual_quat_torch,
    translation_each_joint_torch,
    compute_global_pos_torch,
)
from pymotion.rotations.dual_quat_torch import to_rotation_translation
import pymotion.rotations.ortho6d_torch as ortho6d
from motion_data import calc_feet_energy_torch

from networks.generator import (
    resolve_architecture,
    architecture_classes,
    Generator_Model,
)

__all__ = [
    'resolve_architecture',
    'architecture_classes',
    'Generator_Model',
]

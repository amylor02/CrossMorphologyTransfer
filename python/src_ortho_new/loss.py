"""Backward-compatible imports for training.losses."""

import torch
import torch.nn as nn
from pymotion.ops.forward_kinematics_torch import fk
from pymotion.ops.skeleton_torch import from_root_dual_quat
import torch.nn.functional as F
import pymotion.rotations.ortho6d_torch as ortho6d

from training.losses import (
    GeneratorLoss,
    MSE_DQ,
    MSE_DQ_FK,
    MSE_Latent,
)

__all__ = [
    'GeneratorLoss',
    'MSE_DQ',
    'MSE_DQ_FK',
    'MSE_Latent',
]

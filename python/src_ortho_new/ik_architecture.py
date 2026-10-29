"""Backward-compatible imports for networks.ik.model."""

import torch
import torch.nn as nn
from ik_net import IK_NET
from loss import MSE_DQ_FK

from networks.ik.model import (
    IK_Model,
)

__all__ = [
    'IK_Model',
]

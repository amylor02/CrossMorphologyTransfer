"""Backward-compatible imports for networks.skeleton_layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from networks.skeleton_layers import (
    SkeletonConv,
    SkeletonLinear,
    create_pooling_list,
    SkeletonPool,
    SkeletonUnpool,
    find_collapse_joints,
    create_degree_list,
    create_expanded_neighbor_list,
    distance_joints,
    calc_distance_mat,
    find_neighbor,
)

__all__ = [
    'SkeletonConv',
    'SkeletonLinear',
    'create_pooling_list',
    'SkeletonPool',
    'SkeletonUnpool',
    'find_collapse_joints',
    'create_degree_list',
    'create_expanded_neighbor_list',
    'distance_joints',
    'calc_distance_mat',
    'find_neighbor',
    'dummy_idx',
    'dummy_idx_initial',
]


_LIVE_MODULE = 'networks.skeleton_layers'
# backward compatibility: old names map to live globals in _LIVE_MODULE
_LIVE_NAMES = {'dummy_idx': 'extra_joint_index', 'dummy_idx_initial': 'extra_joint_index_initial'}


def __getattr__(name):
    # globals are rebound at runtime, so read the live value on each access
    if name in _LIVE_NAMES:
        import importlib
        return getattr(importlib.import_module(_LIVE_MODULE), _LIVE_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

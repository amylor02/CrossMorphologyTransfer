"""Backward-compatible imports for original autoencoder architecture."""

import torch
import torch.nn as nn
from skeleton import (
    SkeletonPool,
    SkeletonUnpool,
    find_neighbor,
    SkeletonConv,
    SkeletonLinear,
    create_pooling_list,
)
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pymotion.rotations.ortho6d_torch as ortho6d
import torch.nn.functional as F
from utils import network_timing

from networks.original.autoencoder import (
    Autoencoder,
)
from networks.original.encoder import (
    Encoder,
)
from networks.original.decoder import (
    Decoder,
)
from networks.original.static_encoder import (
    StaticEncoder,
)
from networks.original.prior import (
    RecurrentPrior,
)
from networks.original.components import (
    num_layers,
    dummy_joint,
    attn_dim,
    plot,
    resolve_vae_latent_dim,
    ConvLayerNorm,
    HeadingConcatAdapter,
    TimingEncoder,
    TagEncoder,
    LatentLSTM,
    RotsFromWaypoints,
    DisplacementPredictor,
)

__all__ = [
    'Autoencoder',
    'Encoder',
    'Decoder',
    'StaticEncoder',
    'RecurrentPrior',
    'num_layers',
    'dummy_joint',
    'attn_dim',
    'plot',
    'resolve_vae_latent_dim',
    'ConvLayerNorm',
    'HeadingConcatAdapter',
    'TimingEncoder',
    'TagEncoder',
    'LatentLSTM',
    'RotsFromWaypoints',
    'DisplacementPredictor',
]

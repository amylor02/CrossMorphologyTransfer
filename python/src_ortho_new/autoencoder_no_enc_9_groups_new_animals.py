"""Backward-compatible imports for new_animals autoencoder architecture."""

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

from networks.new_animals.autoencoder import (
    Autoencoder,
)
from networks.new_animals.encoder import (
    Encoder,
)
from networks.new_animals.decoder import (
    Decoder,
)
from networks.new_animals.static_encoder import (
    StaticEncoder,
)
from networks.new_animals.prior import (
    RecurrentPrior,
)
from networks.new_animals.components import (
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

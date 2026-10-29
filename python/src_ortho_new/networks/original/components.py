"""Architecture identifier "original" (Human and Dog).

The implementation is shared in networks/unified. This module binds the variant
ORIGINAL (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.components import (  # noqa: F401
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

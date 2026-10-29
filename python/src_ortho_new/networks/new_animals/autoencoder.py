"""Architecture identifier "new_animals" (Ostrich).

The implementation is shared in networks/unified. This module binds the variant
NEW_ANIMALS (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.autoencoder import Autoencoder as _UnifiedAutoencoder
from networks.unified.components import (  # noqa: F401
    DisplacementPredictor,
    LatentLSTM,
    RotsFromWaypoints,
    TagEncoder,
    TimingEncoder,
    resolve_vae_latent_dim,
)
from networks.new_animals.decoder import Decoder  # noqa: F401
from networks.new_animals.encoder import Encoder  # noqa: F401
from networks.new_animals.prior import RecurrentPrior  # noqa: F401
from networks.unified.variants import NEW_ANIMALS


class Autoencoder(_UnifiedAutoencoder):
    def __init__(self, param, parents, device, transform_net=False, is_vae=False, is_vq_vae=False):
        super().__init__(param, parents, device, transform_net=transform_net, is_vae=is_vae,
                         is_vq_vae=is_vq_vae, variant=NEW_ANIMALS)


__all__ = ['Autoencoder']

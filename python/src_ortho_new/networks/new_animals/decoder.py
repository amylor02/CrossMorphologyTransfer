"""Architecture identifier "new_animals" (Ostrich).

The implementation is shared in networks/unified. This module binds the variant
NEW_ANIMALS (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.decoder import Decoder as _UnifiedDecoder
from networks.unified.variants import NEW_ANIMALS


class Decoder(_UnifiedDecoder):
    def __init__(self, param, enc, device):
        super().__init__(param, enc, device, variant=NEW_ANIMALS)


__all__ = ['Decoder']

"""Architecture identifier "original" (Human and Dog).

The implementation is shared in networks/unified. This module binds the variant
ORIGINAL (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.decoder import Decoder as _UnifiedDecoder
from networks.unified.variants import ORIGINAL


class Decoder(_UnifiedDecoder):
    def __init__(self, param, enc, device):
        super().__init__(param, enc, device, variant=ORIGINAL)


__all__ = ['Decoder']

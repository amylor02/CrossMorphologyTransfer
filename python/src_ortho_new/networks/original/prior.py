"""Architecture identifier "original" (Human and Dog).

The implementation is shared in networks/unified. This module binds the variant
ORIGINAL (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.prior import RecurrentPrior as _UnifiedRecurrentPrior
from networks.unified.variants import ORIGINAL


class RecurrentPrior(_UnifiedRecurrentPrior):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variant', ORIGINAL)
        super().__init__(*args, **kwargs)


__all__ = ['RecurrentPrior']

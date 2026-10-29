"""Architecture identifier "new_animals" (Ostrich).

The implementation is shared in networks/unified. This module binds the variant
NEW_ANIMALS (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.prior import RecurrentPrior as _UnifiedRecurrentPrior
from networks.unified.variants import NEW_ANIMALS


class RecurrentPrior(_UnifiedRecurrentPrior):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('variant', NEW_ANIMALS)
        super().__init__(*args, **kwargs)


__all__ = ['RecurrentPrior']

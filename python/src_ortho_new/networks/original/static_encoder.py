"""Architecture identifier "original" (Human and Dog).

The implementation is shared in networks/unified. This module binds the variant
ORIGINAL (networks/unified/variants.py) and keeps the historical import path.
"""

from networks.unified.static_encoder import StaticEncoder  # noqa: F401

__all__ = ['StaticEncoder']

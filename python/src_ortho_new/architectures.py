"""Autoencoder implementations selectable by architecture identifier.

Identifiers are stored per morphology in config/skeleton_registry.json. This module only maps
them to module names, so the registry and the registration tool can validate identifiers
without importing torch. generator_architecture.py performs the actual import.
"""

ARCHITECTURE_MODULES = {
    "original": "autoencoder_no_enc_9_groups",
    "new_animals": "autoencoder_no_enc_9_groups_new_animals",
}

# Used when no architecture is supplied (training). This is the implementation that
# generator_architecture.py imported before architectures became selectable.
DEFAULT_ARCHITECTURE = "new_animals"

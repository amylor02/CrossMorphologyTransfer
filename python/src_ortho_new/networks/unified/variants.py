"""Variant settings for the unified autoencoder implementation.

Each ArchitectureSettings instance reproduces one historical implementation exactly: the same operations in the same
order, with the same constants and the same random draws. The values were read from the two former packages
(networks/original and networks/new_animals at commit 0aeb3f7). The architecture identifiers ("original",
"new_animals") are unchanged and still map to modules in architectures.py.

Adding a value here changes the behaviour of that architecture. Do not change a value to "clean up" the code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ArchitectureSettings:
    name: str
    # Control tags, in the order they are concatenated into the prior and displacement control vector.
    control_tags: Tuple[str, ...]
    # Per control tag: None keeps the tensor unchanged; a number multiplies the tensor by that value.
    control_multipliers: Tuple[Optional[float], ...]
    # Tags concatenated into the decoder heading input, in order.
    heading_tags: Tuple[str, ...]
    # Channels of the decoder heading input (HeadingConcatAdapter heading_dim).
    heading_dim: int
    # Decoder output is clamped to +-output_clamp. None: no clamp.
    output_clamp: Optional[float]
    # Dropout probability of the prior's dropout layers. Constructor default of RecurrentPrior for this variant.
    prior_dropout: float
    # Dropout inside the prior GRU. None: the prior's dropout argument is used.
    gru_dropout: Optional[float]
    # Probability of zeroing a control channel per timestep during training.
    ctrl_channel_dropout: float
    teacher_forcing_prob: float
    teacher_forcing_min: float
    # (min, max) of the per-sample velocity scale drawn during training. None: no draw.
    velocity_scale_range: Optional[Tuple[float, float]]
    # "carry": the prior always carries its GRU hidden state between calls.
    # "param": the prior carries it only when param["use_hidden_state"] is true (default False).
    hidden_state: str

    def __post_init__(self):
        assert len(self.control_multipliers) == len(self.control_tags)
        assert len(self.heading_tags) == self.heading_dim
        assert self.hidden_state in ("carry", "param")

    def hidden_state_enabled(self, param):
        """Whether the prior carries its GRU hidden state, given the morphology parameter dict."""
        if self.hidden_state == "carry":
            return True
        return bool(param.get("use_hidden_state", False))

    @property
    def ctrl_dim(self):
        return len(self.control_tags)


# Former networks/original (Human, Dog).
ORIGINAL = ArchitectureSettings(
    name="original",
    control_tags=("ctrl_velocity", "ctrl_acceleration", "ctrl_yaw_rate", "ctrl_yaw_accel",
                  "ctrl_height", "ctrl_head_height"),
    control_multipliers=(None, None, None, None, None, None),
    heading_tags=("ctrl_velocity", "ctrl_acceleration", "ctrl_height", "ctrl_vertical_velocity",
                  "yaw_sin", "yaw_cos"),
    heading_dim=6,
    output_clamp=None,
    prior_dropout=0.1,
    gru_dropout=0.1,
    ctrl_channel_dropout=0.25,
    teacher_forcing_prob=0.5,
    teacher_forcing_min=0.2,
    velocity_scale_range=None,
    hidden_state="carry",
)

# Former networks/new_animals (Ostrich).
NEW_ANIMALS = ArchitectureSettings(
    name="new_animals",
    control_tags=("ctrl_velocity", "ctrl_acceleration", "ctrl_yaw_rate", "ctrl_yaw_accel"),
    control_multipliers=(None, 0, 1, 0),
    heading_tags=("ctrl_velocity", "yaw_sin", "yaw_cos"),
    heading_dim=3,
    output_clamp=10.0,
    prior_dropout=0.2,
    gru_dropout=None,
    ctrl_channel_dropout=0.1,
    teacher_forcing_prob=0.95,
    teacher_forcing_min=0.05,
    velocity_scale_range=(0.7, 1.3),
    hidden_state="param",
)

# backward compatibility
VariantSettings = ArchitectureSettings

VARIANTS = {"original": ORIGINAL, "new_animals": NEW_ANIMALS}

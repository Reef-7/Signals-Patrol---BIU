"""Geometry boundary checks and default coordinates for the 3D array contract.

The corrected DOA contract uses eight microphones across two square layers to
provide vertical aperture for simulation. These coordinates are assumptions,
not verified hardware dimensions.
"""

from math import sqrt

from .config import (
    ARRAY_COUNT,
    DEFAULT_ARRAY_LAYER_HALF_WIDTH_METERS,
    DEFAULT_ARRAY_LAYER_SPACING_METERS,
)
from .models import MicrophoneArrayConfig, Vector3D


def default_simulation_microphone_coordinates(
    half_width_meters: float = DEFAULT_ARRAY_LAYER_HALF_WIDTH_METERS,
    layer_spacing_meters: float = DEFAULT_ARRAY_LAYER_SPACING_METERS,
) -> tuple[Vector3D, ...]:
    """Return the default two-square-layer 3D simulation geometry."""
    lower_z = -layer_spacing_meters / 2.0
    upper_z = layer_spacing_meters / 2.0
    lower_layer = (
        (half_width_meters, half_width_meters, lower_z),
        (-half_width_meters, half_width_meters, lower_z),
        (-half_width_meters, -half_width_meters, lower_z),
        (half_width_meters, -half_width_meters, lower_z),
    )
    upper_radius = half_width_meters * sqrt(2.0)
    upper_layer = (
        (upper_radius, 0.0, upper_z),
        (0.0, upper_radius, upper_z),
        (-upper_radius, 0.0, upper_z),
        (0.0, -upper_radius, upper_z),
    )
    return lower_layer + upper_layer


def validate_array_configurations(
    configurations: tuple[MicrophoneArrayConfig, ...],
) -> None:
    """Validate identity/count properties for a complete system configuration."""
    if len(configurations) != ARRAY_COUNT:
        raise ValueError(f"system configuration must contain {ARRAY_COUNT} arrays")
    array_ids = {configuration.array_id for configuration in configurations}
    if len(array_ids) != ARRAY_COUNT:
        raise ValueError("each microphone array must have a distinct array_id")

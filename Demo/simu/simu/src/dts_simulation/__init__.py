"""Contracts and scaffold components for DTS-Simulation."""

from .config import ARRAY_COUNT, MICROPHONES_PER_ARRAY, SAMPLE_COUNT, SAMPLE_RATE_HZ
from .models import (
    ArraySampleMatrix,
    DirectionEstimate,
    LocalizationResult,
    MicrophoneArrayConfig,
    SystemSampleTensor,
)

__all__ = [
    "ARRAY_COUNT",
    "MICROPHONES_PER_ARRAY",
    "SAMPLE_COUNT",
    "SAMPLE_RATE_HZ",
    "ArraySampleMatrix",
    "DirectionEstimate",
    "LocalizationResult",
    "MicrophoneArrayConfig",
    "SystemSampleTensor",
]

